import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# Ensure your model script is named model.py
from src.models.NucleicTransformer import NucleicTransformer
import numpy as np
import math
import random

# Example dataset class


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(label, dtype=torch.long)


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def main():
    # Hyperparameters and configuration
    ntoken = 1000  # Size of vocabulary; adjust as needed
    ninp = 200  # Embedding dimension
    nhead = 2  # Number of attention heads
    nhid = 200  # Dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nkmers = 64  # Number of k-mers (convolutional filters)
    dropout = 0.2
    batch_size = 32
    epochs = 10
    learning_rate = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming sequences and labels are loaded from your dataset
    sequences = [...]  # Your sequence data here
    labels = [...]  # Your labels here
    dataset = SequenceDataset(sequences, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NucleicTransformer(
        ntoken, ninp, nhead, nhid, nlayers, nkmers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train(model, data_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    print("Training complete")


if __name__ == "__main__":
    main()
