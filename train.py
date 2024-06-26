"""
Adapted from https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb.
"""
import argparse
import torch
import pandas as pd
import os, gc
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm
from src.models.starter import RNA_Model
from src.models.MultiCNN import RNA_CNN_Transformer

# from src.models.RNACNNTransformer import RNA_CNN_Transformer
from src.models.Hyena import RNA_HyenaModel
from src.models.GeneViT import GeneViT
from src.data.dataset import RNA_Dataset, LenMatchBatchSampler, DeviceDataLoader

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="RNA Model Training with optional W&B Logging"
)
parser.add_argument(
    "--wandb", action="store_true", help="Enable logging to Weights & Biases"
)
parser.add_argument(
    "--quick_start", action="store_true", help="Use quick start dataset"
)
parser.add_argument("--epochs", type=int, default=32, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--seed", type=int, default=1337, help="Random seed")
parser.add_argument("--toy", action="store_true", help="Use toy dataset")
parser.add_argument(
    "--early_stopping", action="store_true", help="Enable early stopping"
)
# TO SWITCH MODEL, CHANGE THIS ARGUMENT
# 1. starter
# 2. GeneViT
# 3. RNACNNTransformer
parser.add_argument("--model", type=int, default=1, help="Model to use for training")
args = parser.parse_args()

# W&B Integration
if args.wandb:
    wandb.login()
    wandb.init(project="RNA_Translation", entity="rna-fold")
    wandb.config.update({"epochs": args.epochs, "lr": args.lr})

# Use the appropriate dataset file based on -quick_start flag
dataset_file = (
    "train_data_QUICK_START"
    if args.quick_start
    else ("train_data_light" if args.toy else "train_data")
)
print(f"Using dataset: {dataset_file}")


def generate_src_mask(sz, device):
    # Assuming that 'sz' is the sequence length, modify as needed
    # The mask ensures the model doesn't attend to future positions during training
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.unsqueeze(0).to(device)  # Add batch dimension


def one_hot_encode(sequences, device):
    # Assuming sequences is a tensor of sequence IDs with shape [batch_size, sequence_length]
    # and that you have 4 nucleotides (A, C, G, U) represented as 0, 1, 2, 3.
    batch_size, sequence_length = sequences.shape
    one_hot_encoded = torch.zeros(batch_size, 4, sequence_length, device=device)
    for i, nucleotide in enumerate("ACGU"):
        one_hot_encoded[:, i, :] = (sequences == i).to(torch.float32)
    return one_hot_encoded.permute(
        0, 2, 1
    )  # Change to [batch_size, sequence_length, 4] if needed


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


fname = "starter_model"
current_time = datetime.now().strftime("%Y%m%d_%H:%M")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_WEIGHT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model_weights",
)
SEED = args.seed
seed_everything(SEED)
os.makedirs(MODEL_WEIGHT_PATH, exist_ok=True)
model_save_path = os.path.join(MODEL_WEIGHT_PATH, f"{fname}_{current_time}.pth")
bs = 256
num_workers = 1 if True else max(1, os.cpu_count() - 1)
nfolds = 4
device = (
    f"cuda:{torch.cuda.current_device()}"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def dict_to(x, device="cuda"):
    return {k: x[k].to(device) for k in x}


def to_device(x, device="cuda"):
    return tuple(dict_to(e, device) for e in x)


def loss(pred, target):
    p = pred[target["mask"][:, : pred.shape[1]]]
    y = target["react"][target["mask"]].clip(0, 1)
    loss = F.l1_loss(p, y, reduction="none")
    loss = loss[~torch.isnan(loss)].mean()

    return loss


class MAE:
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds, self.targets = [], []

    def accumulate(self, preds, target):
        # Assuming the preds and targets are already filtered by the mask
        self.preds.append(preds.view(-1))
        self.targets.append(target.view(-1))

    def compute(self):
        # Concatenate all accumulated predictions and targets
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        # Calculate the MAE
        loss = F.l1_loss(preds, targets, reduction="none")
        loss = loss[~torch.isnan(loss)].mean()
        return loss.item()


parquet_file = os.path.join(DATA_PATH, f"{dataset_file}.parquet")

print("Read data start")
if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    df = df.drop_duplicates(subset=["sequence_id", "experiment_type"])
    df = df.sort_values(by=["sequence_id", "experiment_type"])
else:
    raise FileNotFoundError(f"File {parquet_file} not found.")
print("Read data end")

fold = 0
ds_train = RNA_Dataset(df, args.seed, mode="train", fold=fold, nfolds=nfolds)
ds_train_len = RNA_Dataset(
    df, args.seed, mode="train", fold=fold, nfolds=nfolds, mask_only=True
)
sampler_train = torch.utils.data.RandomSampler(ds_train_len)
len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs, drop_last=True)
dl_train = DeviceDataLoader(
    torch.utils.data.DataLoader(
        ds_train,
        batch_sampler=len_sampler_train,
        num_workers=num_workers,
        persistent_workers=True,
    ),
    device,
)

ds_val = RNA_Dataset(df, args.seed, mode="eval", fold=fold, nfolds=nfolds)
ds_val_len = RNA_Dataset(
    df, args.seed, mode="eval", fold=fold, nfolds=nfolds, mask_only=True
)
sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs, drop_last=False)
dl_val = DeviceDataLoader(
    torch.utils.data.DataLoader(
        ds_val, batch_sampler=len_sampler_val, num_workers=num_workers
    ),
    device,
)

train_loader = DataLoader(
    ds_train, batch_sampler=len_sampler_train, num_workers=num_workers, pin_memory=True
)
val_loader = DataLoader(
    ds_val, batch_sampler=len_sampler_val, num_workers=num_workers, pin_memory=True
)


model = None
if args.model == 1:
    model = RNA_Model().to(device)
elif args.model == 2:
    model = GeneViT().to(device)
elif args.model == 3:
    model = RNA_CNN_Transformer().to(device)
elif args.model == 4:
    model = RNA_HyenaModel().to(device)

print(f"Now using model: {model}")

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.05)
# lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

scaler = GradScaler()
best_val_loss = float("inf")
early_stopping_counter = 0
patience = 10
gc.collect()
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0

    for batch in tqdm(
        train_loader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False
    ):
        inputs, targets = to_device(batch, device)
        # print(f'inputs shape: {inputs.shape}')
        # print(f'targets shape: {targets.shape}')
        optimizer.zero_grad()
        with autocast():
            predictions = model(inputs)
            loss = loss(predictions, targets)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation loop
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc=f"Epoch {epoch}/{args.epochs} Validation", leave=False
        ):
            inputs, targets = to_device(batch, device)
            with autocast():
                predictions = model(inputs)
                val_loss = loss(predictions, targets)

            val_loss += val_loss.item()

    val_loss /= len(val_loader)

    if args.wandb:
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr_0": optimizer.param_groups[0]["lr"],
            }
        )

    # Checkpoint saving logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch}: New best model saved with val_loss: {val_loss}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if args.early_stopping and early_stopping_counter >= patience:
            print(f"Stopping early at epoch {epoch} due to no improvement.")
            break

    print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}")
gc.collect()
