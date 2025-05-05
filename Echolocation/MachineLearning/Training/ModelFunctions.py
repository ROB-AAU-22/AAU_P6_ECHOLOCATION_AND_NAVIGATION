#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from MachineLearning.Training.TrainingConfig import DISTANCE_THRESHOLD_ENABLED

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

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
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
            A simple feedforward neural network for regression and classification.

            Parameters:
              - input_dim: number of input features.
              - hidden_dim: size of the hidden layer.
              - output_dim: dimensionality of the target (LiDAR scan length).
              - num_layers: number of hidden layers (default=2).
        """
        super(MLPRegressor, self).__init__()
        layers = []
        # Add the first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Add additional hidden layers based on num_layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Define the regression head
        self.regression_head = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))
        # Define the classification head with sigmoid activation for probabilities
        self.classification_head = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        # Pass input through the shared regression layers excluding the last layer
        shared = self.regression_head[:-1](x)
        # Compute regression output using the final layer of regression head
        regression_output = self.regression_head[-1](shared)
        # Compute classification output
        classification_output = self.classification_head(shared)
        # Return both regression and classification outputs
        return regression_output, classification_output


class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)