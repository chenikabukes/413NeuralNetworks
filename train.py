import argparse
import torch
import pandas as pd
import os, gc
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from src.data.dataset import RNA_Dataset, LenMatchBatchSampler
from src.models.starter import RNA_Model
from RNA_CNN_Transformer import RNA_CNN_Transformer

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
PATH = dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + 'data')
OUT = dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + 'model_weights')
SEED = 1337
seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)
model_save_path = os.path.join(OUT, f'{fname}_{current_time}.pth')
bs = 256
num_workers = 1 if True else max(1, os.cpu_count() - 1)
nfolds = 4
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


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
    df = df.drop_duplicates(subset=["sequence_id", "experiment_type"])
    df = df.sort_values(by=["sequence_id", "experiment_type"])
else:
    raise FileNotFoundError(f"File {parquet_file} not found.")
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

# Uncomment 1 of the 3 models for training
# model = RNA_Model().to(device)
model = RNA_CNN_Transformer(ntoken=10, ninp=512, nhead=8, nhid=2048, nlayers=6, nkmers=64, dropout=0.1).to(device)

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.05)
# lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

scaler = GradScaler()
best_val_loss = float('inf')
early_stopping_counter = 0
patience = 10
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
            # print('The predictions are:', predictions)
            # print('The targets are:', targets)
            loss = custom_loss(predictions, targets)

        scaler.scale(loss).backward()
        # Gradient clipping to avoid overfit
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
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
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Stopping early at epoch {epoch} due to no improvement.")
            break

    print(f"Epoch {epoch}: Train Loss: {average_train_loss}, Val Loss: {average_val_loss}")
gc.collect()