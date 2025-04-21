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
    df = pd.read_csv(csv_file)
    df.sort_values("filename", inplace=True)
    print("DataFrame preview:")
    print(df.head())
    
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

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs):
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                outputs = model(X_batch)
                loss = loss_fn(outputs, Y_batch)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
    
    return best_val_loss, best_model_state

def compute_permutation_importance(model, X_val, Y_val, loss_fn, device):
    model.eval()
    base_preds = model(torch.tensor(X_val, dtype=torch.float32).to(device))
    base_loss = loss_fn(base_preds, torch.tensor(Y_val, dtype=torch.float32).to(device)).item()

    importances = []
    for i in range(X_val.shape[1]):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, i])  # Shuffle i-th feature column
        permuted_preds = model(torch.tensor(X_permuted, dtype=torch.float32).to(device))
        permuted_loss = loss_fn(permuted_preds, torch.tensor(Y_val, dtype=torch.float32).to(device)).item()
        importance = permuted_loss - base_loss
        importances.append(importance)

    return importances

# ---------------------------
# Main Function (with Hyperparameter Tuning)
# ---------------------------
def main():
    # Paths for dataset
    csv_file = os.path.join("Extracted features", "features_all_normalized.csv")
    dataset_root = os.path.join("dataset_2")
    
    # Build dataset (X: audio features, Y: LiDAR scan)
    X, Y, sample_ids, feature_names_full = build_dataset_from_csv(csv_file, dataset_root)
    if X.shape[0] == 0:
        print("No valid samples found. Exiting.")
        return
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, LiDAR scan length: {Y.shape[1]}")
    
    # Split dataset into train (70%), validation (15%), and test (15%)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.1765, random_state=42)  # 0.1765*0.85 ~ 15%
    
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Create PyTorch datasets and dataloaders
    batch_size = 16
    train_dataset = AudioLidarDataset(X_train, Y_train)
    val_dataset = AudioLidarDataset(X_val, Y_val)
    test_dataset = AudioLidarDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameter grid for tuning
    learning_rates = [0.001, 0.005, 0.01]
    hidden_sizes = [64, 128, 256]
    num_epochs = 100  # You may adjust epochs
    num_layers = 2  # Number of hidden layers
    
    best_overall_val_loss = float("inf")
    best_hyperparams = None
    best_model_state = None
    
    # Grid search over hyperparameters.
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            print(f"\nTraining with learning rate={lr}, hidden_size={hidden_size}")
            model = MLPRegressor(input_dim, hidden_size, output_dim, num_layers=num_layers).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            
            val_loss, model_state = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs)
            
            print(f"Hyperparams (lr={lr}, hidden_size={hidden_size}) - Best Val Loss: {val_loss:.4f}")
            if val_loss < best_overall_val_loss:
                best_overall_val_loss = val_loss
                best_hyperparams = {"input_dim": input_dim, "output_dim": output_dim, "lr": lr, "hidden_size": hidden_size, "num_epochs": num_epochs, "num_layers": num_layers}
                best_model_state = model_state
    
    print("\nBest hyperparameters found:")
    print(best_hyperparams)
    print(f"Best validation loss: {best_overall_val_loss:.4f}")
    
    # Build the best model and load best state
    best_model = MLPRegressor(input_dim, best_hyperparams["hidden_size"], output_dim, num_layers=num_layers).to(device)
    best_model.load_state_dict(best_model_state)
    
    # Evaluate on the test set.
    best_model.eval()
    test_losses = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = best_model(X_batch)
            loss = loss_fn(outputs, Y_batch)
            test_losses.append(loss.item())
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())
    avg_test_loss = np.mean(test_losses)
    print(f"\nMean Squared Error on test set: {avg_test_loss:.4f}")
    
    # Concatenate predictions for plotting.
    Y_pred = np.concatenate(all_preds, axis=0)
    Y_true = np.concatenate(all_targets, axis=0)
    
    # Plot an example LiDAR scan: ground truth vs. prediction.
    
    for i in range(len(Y_true)):
        plt.figure(figsize=(10, 6))
        plt.plot(Y_true[i], label="Ground Truth LiDAR", marker="o")
        plt.plot(Y_pred[i], label="Predicted LiDAR", linestyle="--", marker="x")
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
    
    """ sample_idx = 10  # Change this index as desired.
    plt.figure(figsize=(10, 6))
    plt.plot(Y_true[sample_idx], label="Ground Truth LiDAR", marker="o")
    plt.plot(Y_pred[sample_idx], label="Predicted LiDAR", linestyle="--", marker="x")
    plt.xlabel("Scan Index")
    plt.ylabel("Distance (m)")
    plt.title("Example LiDAR Scan: Ground Truth vs. Predicted")
    plt.legend()
    plt.grid(True)
    
    plots_folder = os.path.join("Extracted features", "plots")
    os.makedirs(plots_folder, exist_ok=True)
    example_plot_file = os.path.join(plots_folder, "lidar_prediction_example_torch.png")
    plt.savefig(example_plot_file)
    plt.close()
    print(f"Example prediction plot saved to {example_plot_file}") """
    
    # compute and plot feature importance
    importances = compute_permutation_importance(best_model, X_val, Y_val, loss_fn, device)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances)
    plt.xticks(ticks=range(len(importances)), labels=feature_names_full, rotation=90, ha='right')
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Permutation Importance of Features")
    plt.grid()
    plt.show()
        
    # save feature importance to csv
    importance_df = pd.DataFrame({
        "Feature": feature_names_full,
        "Importance": importances
    })
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    
    base_file_dir = os.path.join("Extracted features", "feature_importance")
    os.makedirs(base_file_dir, exist_ok=True)
    
    base_filename = f"feature_importance_{num_epochs}_{num_layers}"
    counter = 1
    file_extension = ".csv"
    importance_file = os.path.join(base_file_dir, f"{base_filename}{file_extension}")
    
    while os.path.exists(importance_file):
        importance_file = os.path.join(base_file_dir, f"{base_filename}_{counter}{file_extension}")
        counter += 1
        
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importances saved to {importance_file}")
    
    # Save the best model.
    models_folder = os.path.join("Extracted features", "models")
    os.makedirs(models_folder, exist_ok=True)
    model_file = os.path.join(models_folder, f"lidar_prediction_model_tuned_torch_{num_epochs}_{num_layers}.pth")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "hyperparameters": best_hyperparams
    }, model_file)
    print(f"Best model saved to {model_file}")

if __name__ == '__main__':
    main()
