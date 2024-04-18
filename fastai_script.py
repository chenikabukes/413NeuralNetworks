import torch
from fastai.vision.all import *
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.models.starter import RNA_Model as Starter
from src.models.RNACNNTransformer import RNA_Model as Single_CNN_Transformer
from src.models.MultiCNN import RNA_Model as Multi_CNN_Transformer
from src.models.Hyena import RNA_MLP_Model as Hyena_MLP_Model
from src.models.Hyena import RNA_Transformer_Model as Hyena_Transformer_Model
import wandb
import argparse
from fastai.callback.wandb import *
from fastai.callback.tracker import SaveModelCallback

from src.data.dataset import RNA_Dataset, LenMatchBatchSampler, DeviceDataLoader, load_data

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

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_WEIGHT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model_weights",
)
bs = 256
num_workers = 2
SEED = 2023
nfolds = 4
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# choose model
model = None
if args.model == 1:
    model = Starter()
elif args.model == 2:
    model = Single_CNN_Transformer()
elif args.model == 3:
    model = Multi_CNN_Transformer()
elif args.model == 4:
    model = Hyena_MLP_Model()
elif args.model == 5:
    model = Hyena_Transformer_Model()
model = model.to(device)
checkpoint_name = f'{model.name()}_{datetime.now().strftime("%Y%m%d_%H%M")}.pth'
cbs = [GradientClip(3.0), SaveModelCallback(monitor='valid_loss', fname=os.path.join(MODEL_WEIGHT_PATH, checkpoint_name))]

if args.wandb:
    wandb.login()
    wandb.init(project='RNA_Translation', entity='rna-fold', name=f"{model.name()}_{datetime.now().strftime('%Y%m%d_%H%M')}", group=f"{model.name()}_{args.epochs}")
    wandb.config.update({"epochs": args.epochs, "lr": args.lr})
    cbs.append(WandbCallback())

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
os.makedirs(MODEL_WEIGHT_PATH, exist_ok=True)
print("DF start")
parquet_file = os.path.join(DATA_PATH, f"train_data.parquet")

df = load_data(parquet_file)

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