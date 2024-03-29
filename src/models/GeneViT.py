import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
# import SimpleViT   # Simple ViT
from simple_vit_regression import SimpleViTForRegression
from pyDeepInsight import ImageTransformer  # sequence to image


class GeneViT(nn.Module):
    def __init__(self, image_dim=15):
        super().__init__()
        self.image_dim = image_dim
        self.it = ImageTransformer(pixels=(image_dim, image_dim))
        self.vit = SimpleViTForRegression(image_size=image_dim**2, patch_size=25, target_dim=206, dim=256, depth=6, heads=16, mlp_dim=512)

    def forward(self, x0):
        # Extend mask and data to length 225
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        extend = torch.zeros(self.image_dim ** 2 - Lmax, dtype=torch.bool)
        Lextend = len(extend) + Lmax
        mask = mask.cat((mask, extend), dim=0)

        seq = x0['seq']
        seq = np.pad(seq,(0,Lextend-len(seq)))
        x = seq[:,:Lextend]

        # Transform into image format
        x = self.it.transform(x)

        # TODO SimpleViT with motifications for regression
        x = self.vit(x)

        return x
