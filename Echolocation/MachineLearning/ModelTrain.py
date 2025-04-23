#!/usr/bin/env python3
import os
import json
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


# ---------------------------
# Data Preparation Functions
# ---------------------------
def build_dataset_from_csv(csv_file, dataset_root):
    """
    Build a dataset for ML using normalized audio features from a CSV file and LiDAR scans from dataset_2 structure.

    Parameters:
      - csv_file: Path to features_all_normalized.csv.
      - dataset_root: Root folder containing dataset_2.

    Returns:
      - X: NumPy array of shape (n_samples, n_audio_features)
      - Y: NumPy array of shape (n_samples, n_lidar_points)
      - sample_ids: List of sample indices (timestamps)
    """
    df = pd.read_csv(csv_file)  # Read the CSV file containing features
    df.sort_values("filename", inplace=True)  # Sort dataframe based on filenames
    print("DataFrame preview:")
    print(df.head())  # Preview the first few rows of the dataframe

    X_list = []  # List to store feature vectors
    Y_list = []  # List to store LiDAR vectors
    sample_ids = []  # List to track sample IDs
    feature_names_full = []  # Feature names for the dataset

    # Exclude 'filename' column to capture feature columns
    feature_cols = [col for col in df.columns if col != "filename"]

    # Set of sample numbers to skip due to potential errors or issues
    skip_ids = {"1", "10", "50", "53", "69", "97", "114", "128", "129", "149", "157", "181", "199", "200", "234", "250",
                "263", "283", "396", "441", "465", "472", "477", "502", "527", "538", "645", "668", "686", "697",
                "713"}  # Add more as needed

    for idx, row in df.iterrows():
        filename = row["filename"]  # Get the filename for current row

        # Skip files based on defined skip IDs
        if any(f"_{sid}_" in filename for sid in skip_ids):
            print(f"Skipping file due to skip list: {filename}")
            continue

        # Path to the LiDAR JSON file associated with the audio sample
        lidar_filename = f"{filename}_distance_data.json"
        lidar_file = os.path.join(dataset_root, filename, lidar_filename)

        print(f"Processing file: {filename}")

        feature_vector = []  # Create a feature vector for the current row
        feature_names_row = []  # Feature names for this specific row
        for col in feature_cols:
            cell = row[col]  # Extract cell value
            # Check if the cell contains a string representation of a list
            if isinstance(cell, str) and cell.strip().startswith('[') and cell.strip().endswith(']'):
                try:
                    parsed_value = ast.literal_eval(cell)  # Parse list-like strings
                    if isinstance(parsed_value, list):
                        feature_vector.extend(
                            [float(x) for x in parsed_value])  # Extend feature vector with parsed list
                        feature_names_row.extend(
                            [f"{col}_{i}" for i in range(len(parsed_value))])  # Create detailed feature names
                    else:
                        feature_vector.append(float(parsed_value))  # Append single value if not a list
                        feature_names_row.append(col)
                except Exception as e:
                    # Handle parsing error if the string cannot be converted
                    print(f"Error parsing column '{col}' with value '{cell}': {e}")
                    continue
            else:
                try:
                    feature_vector.append(float(cell))  # Append numeric value
                    feature_names_row.append(col)  # Append feature name
                except Exception as e:
                    # Handle error if non-numeric value cannot be converted
                    print(f"Error converting cell in column '{col}' with value '{cell}' to float: {e}")
                    continue

        # Save the column names for the first row as full feature names
        if idx == 0:
            feature_names_full = feature_names_row

        # Check for existence of LiDAR file
        if not os.path.exists(lidar_file):
            print(f"LiDAR file {lidar_file} not found. Skipping sample {filename}.")
            continue

        # Read and process the LiDAR file
        try:
            with open(lidar_file, "r") as f:
                lidar_data = json.load(f)  # Load JSON file
            lidar_vector = np.array(lidar_data["LiDAR_distance"], dtype=float)  # Convert distances to NumPy array
        except Exception as e:
            # Handle errors in reading/processing the LiDAR file
            print(f"Error loading LiDAR file {lidar_file}: {e}. Skipping sample {filename}.")
            continue

        # Append processed feature and target vectors to their respective lists
        X_list.append(feature_vector)
        Y_list.append(lidar_vector)
        sample_ids.append(filename)

    # Convert lists to NumPy arrays and return
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, sample_ids, feature_names_full


class AudioLidarDataset(Dataset):
    def __init__(self, X, Y):
        """
        PyTorch Dataset for combining audio features (X) and LiDAR outputs (Y).
        
        Parameters:
          - X: Input audio feature data as a NumPy array.
          - Y: Output LiDAR distance data as a NumPy array.
        """
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert X to PyTorch tensor
        self.Y = torch.tensor(Y, dtype=torch.float32)  # Convert Y to PyTorch tensor

    def __len__(self):
        # Return the number of samples in the dataset
        return self.X.shape[0]

    def __getitem__(self, index):
        # Return the data sample and its corresponding target at the given index
        return self.X[index], self.Y[index]


# ---------------------------
# PyTorch Model Definition
# ---------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        A simple feedforward neural network for regression.
        
        Parameters:
          - input_dim: number of input features.
          - hidden_dim: size of the hidden layer.
          - output_dim: dimensionality of the target (LiDAR scan length).
          - num_layers: number of hidden layers (default=2).
        """
        super(MLPRegressor, self).__init__()
        layers = []
        # Add input layer and its activation
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Add hidden layers with activations
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Add output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)  # Combine layers into a sequential model

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)


# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs):
    """
    Train a PyTorch model with the given parameters.

    Parameters:
      - model: The PyTorch model to train.
      - train_loader: DataLoader for training data.
      - val_loader: DataLoader for validation data.
      - optimizer: Optimizer for parameter updates.
      - loss_fn: Loss function to optimize.
      - device: Device to use ('cpu' or 'cuda').
      - num_epochs: Number of epochs to train for.

    Returns:
      - best_val_loss: Lowest validation loss encountered during training.
      - best_model_state: State dictionary of the model with lowest validation loss.
    """
    best_val_loss = float("inf")  # Initialize the best validation loss to infinity
    best_model_state = None  # Variable to store the best model state

    for epoch in range(1, num_epochs + 1):
        model.train()  # Set model to training mode
        train_losses = []  # List to store training losses
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)  # Move input data to the device
            Y_batch = Y_batch.to(device)  # Move target data to the device
            optimizer.zero_grad()  # Reset gradients
            outputs = model(X_batch)  # Forward pass
            loss = loss_fn(outputs, Y_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_losses.append(loss.item())  # Append batch loss

        avg_train_loss = np.mean(train_losses)  # Compute average training loss

        model.eval()  # Set model to evaluation mode
        val_losses = []  # List to store validation losses
        with torch.no_grad():  # No gradient computation in evaluation
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)  # Move input data to the device
                Y_batch = Y_batch.to(device)  # Move target data to the device
                outputs = model(X_batch)  # Forward pass
                loss = loss_fn(outputs, Y_batch)  # Compute loss
                val_losses.append(loss.item())  # Append batch loss
        avg_val_loss = np.mean(val_losses)  # Compute average validation loss

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Update the best model state if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    return best_val_loss, best_model_state
