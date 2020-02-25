"""
author: Wira D K Putra
https://github.com/WiraDKP/pytorch_speaker_embedding_for_diarization
25 Feb 2020
"""

import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, ndim, triplet=False):
        super().__init__()
        self.ndim = ndim
        self.triplet = triplet
        self.conv = nn.Sequential(
            nn.Conv2d(1, ndim, (40, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(),            
            nn.Conv2d(ndim, ndim, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU(),
            nn.Conv2d(ndim, ndim, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ndim),            
            nn.ReLU()            
        )
        self.fc = nn.Sequential(
            nn.Linear(2*ndim, ndim),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv(x)
        mean, std = x.mean(-1), x.std(-1)
        x = torch.cat([mean, std], dim=1).squeeze(-1)
        x = self.fc(x)
            
        if self.triplet:
            return x.view(3, -1, self.ndim)
        else:
            return x.view(-1, self.ndim)