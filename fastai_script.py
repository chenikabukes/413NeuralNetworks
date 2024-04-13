import torch
from fastai.vision.all import *
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
from src.models.RNACNNTransformer import RNA_Model as Simgle_CNN_Transformer
from src.models.Hyena import RNA_Model as RNA_HyenaModel
from src.models.MultiCNN import RNA_Model as Multi_CNN_Transformer
from src.models.starter import RNA_Model as Starter
import wandb
import argparse
from fastai.callback.wandb import *
from fastai.callback.tracker import SaveModelCallback

from src.data.dataset import RNA_Dataset, LenMatchBatchSampler, DeviceDataLoader

# Parse command-line arguments
parser = argparse.ArgumentParser(description='RNA Model Training with optional W&B Logging')
parser.add_argument('--wandb', action='store_true', help='Enable logging to Weights & Biases')
parser.add_argument('--quick_start', action='store_true', help='Use quick start dataset')
parser.add_argument('--epochs', type=int, default=32, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument("--model", type=int, default=1, help="Model to use for training")

args = parser.parse_args()


def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast

@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self):
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property
    def param_groups(self):
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs):
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None

import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

PATH = '/h/u6/c4/05/zha11021/CSC413/413NeuralNetworks/data/'
WEIGHT_OUT = '/h/u6/c4/05/zha11021/CSC413/413NeuralNetworks/fasit_ai_model_weights/'
bs = 256
num_workers = 2
SEED = 2023
nfolds = 4
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = None
if args.model == 1:
    model = Starter()
elif args.model == 2:
    model = Simgle_CNN_Transformer()
elif args.model == 3:
    model = Multi_CNN_Transformer()
elif args.model == 4:
    model = RNA_HyenaModel()
model = model.to(device)
cbs = [GradientClip(3.0), SaveModelCallback(monitor='valid_loss', fname=f"{model.name()}_{datetime.now().strftime('%Y%m%d_%H%M')}")]

if args.wandb:
    wandb.login()
    wandb.init(project='RNA_Translation', entity='rna-fold', name=f"{model.name()}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    wandb.config.update({"epochs": args.epochs, "lr": args.lr})
    cbs.append(WandbCallback())


# class RNA_Dataset(Dataset):
#     def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4,
#                  mask_only=False, **kwargs):
#         self.seq_map = {'A':0,'C':1,'G':2,'U':3}
#         self.Lmax = 206
#         df['L'] = df.sequence.apply(len)
#         df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
#         df_DMS = df.loc[df.experiment_type=='DMS_MaP']
#         split = list(KFold(n_splits=nfolds, random_state=seed,
#                 shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
#         df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
#         df_DMS = df_DMS.iloc[split].reset_index(drop=True)

#         m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
#         df_2A3 = df_2A3.loc[m].reset_index(drop=True)
#         df_DMS = df_DMS.loc[m].reset_index(drop=True)

#         self.seq = df_2A3['sequence'].values
#         self.L = df_2A3['L'].values

#         self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
#                                  'reactivity_0' in c]].values
#         self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
#                                  'reactivity_0' in c]].values
#         self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
#                                  'reactivity_error_0' in c]].values
#         self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
#                                 'reactivity_error_0' in c]].values
#         self.sn_2A3 = df_2A3['signal_to_noise'].values
#         self.sn_DMS = df_DMS['signal_to_noise'].values
#         self.mask_only = mask_only

#     def __len__(self):
#         return len(self.seq)

#     def __getitem__(self, idx):
#         seq = self.seq[idx]
#         if self.mask_only:
#             mask = torch.zeros(self.Lmax, dtype=torch.bool)
#             mask[:len(seq)] = True
#             return {'mask':mask},{'mask':mask}
#         seq = [self.seq_map[s] for s in seq]
#         seq = np.array(seq)
#         mask = torch.zeros(self.Lmax, dtype=torch.bool)
#         mask[:len(seq)] = True
#         seq = np.pad(seq,(0,self.Lmax-len(seq)))

#         react = torch.from_numpy(np.stack([self.react_2A3[idx],
#                                            self.react_DMS[idx]],-1))
#         react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
#                                                self.react_err_DMS[idx]],-1))
#         sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])

#         return {'seq':torch.from_numpy(seq), 'mask':mask}, \
#                {'react':react, 'react_err':react_err,
#                 'sn':sn, 'mask':mask}

# class LenMatchBatchSampler(torch.utils.data.BatchSampler):
#     def __iter__(self):
#         buckets = [[]] * 100
#         yielded = 0

#         for idx in self.sampler:
#             s = self.sampler.data_source[idx]
#             if isinstance(s,tuple): L = s[0]["mask"].sum()
#             else: L = s["mask"].sum()
#             L = max(1,L // 16)
#             if len(buckets[L]) == 0:  buckets[L] = []
#             buckets[L].append(idx)

#             if len(buckets[L]) == self.batch_size:
#                 batch = list(buckets[L])
#                 yield batch
#                 yielded += 1
#                 buckets[L] = []

#         batch = []
#         leftover = [idx for bucket in buckets for idx in bucket]

#         for idx in leftover:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yielded += 1
#                 yield batch
#                 batch = []

#         if len(batch) > 0 and not self.drop_last:
#             yielded += 1
#             yield batch

# def dict_to(x, device='cuda'):
#     return {k:x[k].to(device) for k in x}

# def to_device(x, device='cuda'):
#     return tuple(dict_to(e,device) for e in x)

# class DeviceDataLoader:
#     def __init__(self, dataloader, device='cuda'):
#         self.dataloader = dataloader
#         self.device = device

#     def __len__(self):
#         return len(self.dataloader)

#     def __iter__(self):
#         for batch in self.dataloader:
#             yield tuple(dict_to(x, self.device) for x in batch)

def loss(pred,target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()

    return loss

class MAE(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.x,self.y = [],[]

    def accumulate(self, learn):
        x = learn.pred[learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0,1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss


seed_everything(SEED)
os.makedirs(WEIGHT_OUT, exist_ok=True)
print("DF start")
parquet_file = os.path.join(PATH, f"train_data.parquet")

if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    df = df.drop_duplicates(subset=["sequence_id", "experiment_type"])
    df = df.sort_values(by=["sequence_id", "experiment_type"])
else:
    raise FileNotFoundError(f"File {parquet_file} not found.")
print("DF end")


ds_train = RNA_Dataset(df, SEED, mode='train', nfolds=nfolds)
ds_train_len = RNA_Dataset(df, SEED, mode='train',
            nfolds=nfolds, mask_only=True)
sampler_train = torch.utils.data.RandomSampler(ds_train_len)
len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
            drop_last=True)
dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train,
            batch_sampler=len_sampler_train, num_workers=num_workers,
            persistent_workers=True), device)

ds_val = RNA_Dataset(df, SEED, mode='eval', nfolds=nfolds)
ds_val_len = RNA_Dataset(df, SEED, mode='eval', nfolds=nfolds,
            mask_only=True)
sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs,
            drop_last=False)
dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val,
            batch_sampler=len_sampler_val, num_workers=num_workers), device)
gc.collect()

data = DataLoaders(dl_train,dl_val)

learn = Learner(data, model, loss_func=loss,cbs=cbs,
            metrics=[MAE()]).to_fp16()
#fp16 doesn't help at P100 but gives x1.6-1.8 speedup at modern hardware

learn.fit_one_cycle(args.epochs, lr_max=args.lr, wd=0.05, pct_start=0.02)
gc.collect()