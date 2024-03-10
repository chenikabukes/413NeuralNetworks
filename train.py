import argparse
import torch
import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='RNA Model Training with optional W&B Logging')
parser.add_argument('--wandb', action='store_true', help='Enable logging to Weights & Biases')
parser.add_argument('--quick_start', action='store_true', help='Use quick start dataset')
parser.add_argument('--epochs', type=int, default=32, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
args = parser.parse_args()

# W&B Integration
if args.wandb:
    wandb.login()
    wandb.init(project='RNA_Translation', entity='rna-fold')
    wandb.config.update({"epochs": args.epochs, "lr": args.lr})

# Use the appropriate dataset file based on -quick_start flag
dataset_file = 'train_data_QUICK_START' if args.quick_start else 'train_data'
print(f"Using dataset: {dataset_file}")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

fname = 'starter_model'
current_time = datetime.now().strftime('%Y%m%d_%H:%M')
PATH = '/h/u6/c4/05/zha11021/CSC413/413NeuralNetworks/dataset'
OUT = '/h/u6/c4/05/zha11021/CSC413/413NeuralNetworks/model_weights'
SEED = 1337
seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)
model_save_path = os.path.join(OUT, f'{fname}_{current_time}.pth')
bs = 256
num_workers = 1 if True else max(1, os.cpu_count() - 1)
nfolds = 4
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4,
                 mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = 206
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        split = list(KFold(n_splits=nfolds, random_state=seed,
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,self.Lmax-len(seq)))

        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])

        return {'seq':torch.from_numpy(seq), 'mask':mask}, \
               {'react':react, 'react_err':react_err,
                'sn':sn, 'mask':mask}

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s,tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            L = max(1,L // 16)
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)

            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RNA_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size,
                                                                   dim_feedforward=4*dim, dropout=0.1,
                                                                   activation='gelu', batch_first=True,
                                                                   norm_first=True) for _ in range(depth)])
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layers[0], num_layers=depth)

        self.proj_out = nn.Linear(dim,2)

    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)

        return x

def custom_loss(pred,target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()

    return loss

parquet_file = os.path.join(PATH, f"{dataset_file}.parquet")

print("Read data start")
if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
else:
    csv_file = os.path.join(PATH, f"{dataset_file}.csv")
    df = pd.read_csv(csv_file)
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    # Drop duplicates based on "sequence_id", "experiment_type"
    df = df.drop_duplicates(subset=["sequence_id", "experiment_type"])
    # Sort the DataFrame based on "sequence_id" and "experiment_type"
    df = df.sort_values(by=["sequence_id", "experiment_type"])
    df.to_parquet(parquet_file)
print("Read data end")

fold = 0
ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
ds_train_len = RNA_Dataset(df, mode='train', fold=fold,
            nfolds=nfolds, mask_only=True)
sampler_train = torch.utils.data.RandomSampler(ds_train_len)
len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
            drop_last=True)
dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,
            batch_sampler=len_sampler_train, num_workers=num_workers,
            persistent_workers=True), device)

ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
ds_val_len = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds,
            mask_only=True)
sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs,
            drop_last=False)
dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val,
            batch_sampler=len_sampler_val, num_workers=num_workers), device)

train_loader = DataLoader(ds_train, batch_sampler=len_sampler_train, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(ds_val, batch_sampler=len_sampler_val, num_workers=num_workers, pin_memory=True)

model = RNA_Model().to(device)
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.05)
scaler = GradScaler()
best_val_loss = float('inf')
gc.collect()
for epoch in range(1, args.epochs+1):
    model.train()
    train_loss_accumulator = 0.0
    num_batches_train = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False):
        inputs, targets = to_device(batch, device)  # Ensure this function correctly moves your batch to the device
        optimizer.zero_grad()

        with autocast():
            predictions = model(inputs)
            loss = custom_loss(predictions, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_accumulator += loss.item()
        num_batches_train += 1

    average_train_loss = train_loss_accumulator / num_batches_train

    # Validation loop
    val_loss_accumulator = 0.0
    num_batches_val = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} Validation", leave=False):
            inputs, targets = to_device(batch, device)
            with autocast():
                predictions = model(inputs)
                val_loss = custom_loss(predictions, targets)

            val_loss_accumulator += val_loss.item()
            num_batches_val += 1

    average_val_loss = val_loss_accumulator / num_batches_val

    if args.wandb:
        wandb.log({"epoch": epoch, "train_loss": average_train_loss, "val_loss": average_val_loss, "lr_0": optimizer.param_groups[0]['lr']})

    # Checkpoint saving logic
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch}: New best model saved with val_loss: {average_val_loss}")

    print(f"Epoch {epoch}: Train Loss: {average_train_loss}, Val Loss: {average_val_loss}")
gc.collect()