#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from MachineLearning.Training.TrainingConfig import DISTANCE_THRESHOLD_ENABLED

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        # Mask values where targets are NaN or greater than distance_threshold
        if DISTANCE_THRESHOLD_ENABLED:
            mask = ~torch.isnan(targets)
            outputs = outputs[mask]
            targets = targets[mask]
        loss = self.mse_loss(outputs, targets)
        return loss.mean()


class AudioLidarDataset(Dataset):
    def __init__(self, X, Y):
        """
            Initializes the dataset with input features (X) and targets (Y).
            Expects X and Y as NumPy arrays.
        """
        # Convert X and Y to PyTorch tensors with type float32
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        # Returns the total number of samples in the dataset
        return self.X.shape[0]

    def __getitem__(self, index):
        # Retrieves the input (X) and target (Y) at the specified index
        return self.X[index], self.Y[index]

# ---------------------------
# PyTorch Model Definition
# ---------------------------


class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, layer_type):
        """
            A simple feedforward neural network for regression.

            Parameters:
              - input_dim: number of input features.
              - hidden_dim: size of the hidden layer.
              - output_dim: dimensionality of the target (LiDAR scan length).
              - num_layers: number of hidden layers (default=2).
        """
        super().__init__()
        print(f"regressor layer_type: {layer_type}")
        if layer_type == "Tanh":
            layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        elif layer_type == "Sigmoid":
            layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
             
        for _ in range(num_layers):
            if layer_type == "Tanh":
                layers += [nn.Linear(hidden_dim, int(hidden_dim)), nn.Tanh()]
            elif layer_type == "Sigmoid":
                layers += [nn.Linear(hidden_dim, int(hidden_dim)), nn.Sigmoid()]
            else:
                layers += [nn.Linear(hidden_dim, int(hidden_dim)), nn.ReLU()]
            #layers.append(nn.Dropout(0.3))
            #hidden_dim = int(hidden_dim*2)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, layer_type):
        super().__init__()
        print(f"classifier layer_type: {layer_type}")
        if layer_type == "Tanh":
            layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        elif layer_type == "Sigmoid":
            layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
             
        for _ in range(num_layers):
            if layer_type == "Tanh":
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            elif layer_type == "Sigmoid":
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()]
            else:
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            #layers.append(nn.Dropout(0.2))
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)