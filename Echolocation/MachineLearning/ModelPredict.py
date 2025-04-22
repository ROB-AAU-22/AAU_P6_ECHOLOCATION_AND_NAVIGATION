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
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class AudioLidarDataset(Dataset):
    def __init__(self, X, Y):
        """
        Expects X and Y as NumPy arrays.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def load_model(epochs, layers):
    model_file = os.path.join("Extracted features", "models", f"lidar_prediction_model_tuned_torch_{epochs}_{layers}.pth")
    model_checkpoint = torch.load(model_file)
    
    hyperparams = model_checkpoint['hyperparameters']
    input_dim = hyperparams['input_dim']
    output_dim = hyperparams['output_dim']
    hidden_size = hyperparams['hidden_size']
    
    model = MLPRegressor(input_dim, hidden_size, output_dim, layers)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    
    return model, hyperparams

def load_data(file_path, dataset_root):
    df = pd.read_csv(file_path)
    df.sort_values("filename", inplace=True)
    
    X_list = []
    Y_list = []
    sample_ids = []
    feature_names_full = []
    
    feature_cols = [col for col in df.columns if col != "filename"]
    
    for idx, row in df.iterrows():
        filename = row["filename"]
        try:
            # Extract timestamp (e.g., 1744104210) from the filename
            timestamp = filename.split('_')[0]
        except Exception as e:
            print(f"Error parsing timestamp from filename '{filename}': {e}. Skipping row.")
            continue

        feature_vector = []
        feature_names_row = []
        for col in feature_cols:
            cell = row[col]
            if isinstance(cell, str) and cell.strip().startswith('[') and cell.strip().endswith(']'):
                try:
                    parsed_value = ast.literal_eval(cell)
                    if isinstance(parsed_value, list):
                        feature_vector.extend([float(x) for x in parsed_value])
                        feature_names_row.extend([f"{col}_{i}" for i in range(len(parsed_value))])
                    else:
                        feature_vector.append(float(parsed_value))
                        feature_names_row.append(col)
                except Exception as e:
                    print(f"Error parsing column '{col}' with value '{cell}': {e}")
                    continue
            else:
                try:
                    feature_vector.append(float(cell))
                    feature_names_row.append(col)
                except Exception as e:
                    print(f"Error converting cell in column '{col}' with value '{cell}' to float: {e}")
                    continue

        if idx == 0:
            feature_names_full = feature_names_row

        lidar_file = os.path.join(dataset_root, timestamp, f"{timestamp}_distance_data.json")
        if not os.path.exists(lidar_file):
            print(f"LiDAR file {lidar_file} not found. Skipping sample {timestamp}.")
            continue
        try:
            with open(lidar_file, "r") as f:
                lidar_data = json.load(f)
            lidar_vector = np.array(lidar_data["LiDAR_distance"], dtype=float)
        except Exception as e:
            print(f"Error loading LiDAR file {lidar_file}: {e}. Skipping sample {timestamp}.")
            continue

        X_list.append(feature_vector)
        Y_list.append(lidar_vector)
        sample_ids.append(timestamp)

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, sample_ids, feature_names_full

def predict_model(model, data_loader, device):
    model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = model(X_batch)
            #loss = nn.MSELoss(outputs, Y_batch)
            #test_losses.append(loss.item())
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())
    avg_loss = np.mean(test_losses)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return avg_loss, all_predictions, all_targets

def main():
    dataset_csv = os.path.join("Extracted features", "features_all_normalized.csv")
    dataset_root = os.path.join("dataset_2")
    
    X, Y, sample_ids, feature_names_full = load_data(dataset_csv, dataset_root)
    
    if X.shape[0] == 0:
        print("No valid data found.")
        return
    print(f"Dataset: {dataset_csv} loaded with {X.shape[0]} samples, {X.shape[1]} features, and LiDAR scan length: {Y.shape[1]}.")
    
    batch_size = 16
    test_dataset = AudioLidarDataset(X, Y)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    num_epochs = 100
    num_layers = 2
    
    model, hyperparams = load_model(num_epochs, num_layers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    avg_loss, predictions, targets = predict_model(model, test_loader, device)
    print(f"Average test loss: {avg_loss:.4f}")
    
    for i in range(len(targets)):
        plt.figure(figsize=(10, 6))
        plt.plot(targets[i], label="Ground Truth LiDAR", marker="o")
        plt.plot(predictions[i], label="Predicted LiDAR", linestyle="--", marker="x")
        plt.xlabel("Scan Index")
        plt.ylabel("Distance (m)")
        plt.title(f"LiDAR Scan {i} : {num_epochs} epochs : {num_layers} layers")
        plt.legend()
        plt.grid(True)
        
        lidar_plots_folder = os.path.join("Extracted features", f"lidar_plots_{num_epochs}_{num_layers}")
        os.makedirs(lidar_plots_folder, exist_ok=True)
        lidar_plot_file = os.path.join(lidar_plots_folder, f"lidar_prediction_{i}.png")
        plt.savefig(lidar_plot_file)
        plt.close()
        print(f"LiDAR prediction plot saved to {lidar_plot_file}")

if __name__ == '__main__':
    main()