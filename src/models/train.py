import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../data')
from dataset import RNA_Dataset, LenMatchBatchSampler, DeviceDataLoader
from GeneViT import GeneViT 
from fastai.callback.wandb import Learner
from fastai.vision.all import GradientClip, Metric

def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)

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

fname = 'example0'
PATH = '../../data/'
OUT = './'
bs = 256
num_workers = 2
SEED = 2023
nfolds = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(os.path.join(PATH,'train_data.parquet'))

for fold in [0]: # running multiple folds at kaggle may cause OOM
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
    gc.collect()

    data = DataLoader(dl_train, batch_size=1)
    model = GeneViT().to(device)
    learn = Learner(data, model, loss_func=loss,cbs=[GradientClip(3.0)],
                metrics=[MAE()]).to_fp16()

    learn.fit_one_cycle(32, lr_max=5e-4, wd=0.05, pct_start=0.02)
    torch.save(learn.model.state_dict(),os.path.join(OUT,f'{fname}_{fold}.pth'))
    gc.collect()
