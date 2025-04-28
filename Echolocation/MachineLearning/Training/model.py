#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        mask = ~torch.isnan(targets)
        outputs = outputs[mask]
        targets = targets[mask]
        loss = self.mse_loss(outputs, targets)
        return loss.mean()

class AudioLidarDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MLPRegressor, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.regression_head = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))
        self.classification_head = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        shared = self.regression_head[:-1](x)
        regression_output = self.regression_head[-1](shared)
        classification_output = self.classification_head(shared)
        return regression_output, classification_output
