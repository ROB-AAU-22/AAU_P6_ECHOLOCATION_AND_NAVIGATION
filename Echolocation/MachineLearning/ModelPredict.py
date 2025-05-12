#!/usr/bin/env python3
import os
import json
import ast
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from FeatureExtraction import FeatureExtractionScript
from FeatureExtraction import NormalizeFeatures
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

def polar_to_cartesian(distances, angle_range=(-3*np.pi/4, 3*np.pi/4)):
    angles = np.linspace(angle_range[0], angle_range[1], num=len(distances))
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return x, y

def load_model_regressor(model_path):
    """ Load the trained model from a path. Epochs and layers are used to construct the file name. """
    print(f"Loading model from {model_path}")
    model_checkpoint = torch.load(model_path)
    
    hyperparams = model_checkpoint['hyperparameters']
    input_dim = hyperparams['input_dim']
    output_dim = hyperparams['output_dim']
    hidden_size = hyperparams['hidden_size']
    hidden_layer_count = hyperparams['num_layers']
    
    model = MLPRegressor(input_dim, hidden_size, output_dim, hidden_layer_count)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    
    return model, hyperparams

def load_model_classifier(model_path):
    """ Load the trained model from a path. Epochs and layers are used to construct the file name. """
    print(f"Loading model from {model_path}")
    model_checkpoint = torch.load(model_path)
    
    hyperparams = model_checkpoint['hyperparameters']
    input_dim = hyperparams['input_dim']
    output_dim = hyperparams['output_dim']
    hidden_size = hyperparams['hidden_size']
    hidden_layer_count = hyperparams['num_layers']
    
    model = Classifier(input_dim, hidden_size, output_dim, hidden_layer_count)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    
    return model, hyperparams

def predict_single_input(model, input_tensor):
    """
    Predict the output for a single input tensor using the trained model.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        input_tensor (torch.Tensor): A single input tensor (1D or 2D).

    Returns:
        np.ndarray: The predicted output as a NumPy array.
    """
    # Ensure the model is on the correct device
    device = next(model.parameters()).device
    model.eval()

    # Move the input tensor to the same device as the model
    input_tensor = input_tensor.to(device)

    # Add a batch dimension if the input tensor is 1D
    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        # Perform the prediction
        output = model(input_tensor)

    # Convert the output to a NumPy array and return
    return output.cpu()

def load_single_row_from_csv(csv_path, row_index):
    """
    Load a single row from a CSV file and return its values as a list.

    Parameters:
        csv_path (str): Path to the CSV file.
        row_index (int): Index of the row to load (0-based).

    Returns:
        list: The selected row's values as a list.
    """
    # Load the CSV file
    data = pd.read_csv(csv_path)
    
    # Check if the row index is valid
    if row_index < 0 or row_index >= len(data):
        raise IndexError(f"Row index {row_index} is out of bounds for the CSV file with {len(data)} rows.")
    
    # Extract the specified row and convert it to a list
    row_values = data.iloc[row_index].tolist()
    return row_values

def plot_cartisian(original_gt_i, predictions, classifications_i, distance, DPI, time_stamp):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)
    #print(f"Worker {worker_id} plotting cartesian LiDAR for sample {i}...")
    gt_x, gt_y = polar_to_cartesian(original_gt_i)
    pred_x, pred_y = polar_to_cartesian(predictions)
    ignored_gt = original_gt_i > distance

    ignored_gt_x, ignored_gt_y = polar_to_cartesian(original_gt_i)
    ax.scatter(ignored_gt_x[ignored_gt], ignored_gt_y[ignored_gt], color='red', marker='o', label='Ignored GT', alpha=0.7, zorder=1)
    ax.plot(gt_x[~ignored_gt], gt_y[~ignored_gt], label="Ground Truth LiDAR", marker='o', linestyle='-', alpha=0.7, zorder=2)

    robot_circle = plt.Circle((0, 0), 0.2, color='gray', fill=True, alpha=0.5, label='Robot', zorder=2)
    ax.add_patch(robot_circle)

    # draw a line from origin to first scan point
    plt.plot([0, gt_x[0]], [0, gt_y[0]], color='blue', linestyle='--', alpha=0.5, zorder=3)
    plt.plot([0, pred_x[0]], [0, pred_y[0]], color='red', linestyle='--', alpha=0.5, zorder=3)
    # draw a line from origin to last scan point
    plt.plot([0, gt_x[-1]], [0, gt_y[-1]], color='blue', linestyle='--', alpha=0.5, zorder=3)
    plt.plot([0, pred_x[-1]], [0, pred_y[-1]], color='red', linestyle='--', alpha=0.5, zorder=3)
    # draw an arrow vector from origin to middle point(s)
    plt.arrow(0, 0, gt_x[540], gt_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black', alpha=1, zorder=4)
    plt.arrow(0, 0, pred_x[540], pred_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black', alpha=1, zorder=4)
    #print(f"Classifications_i: {classifications_i}")
    classified_as_object = classifications_i > 0.5
    classified_as_no_object = ~classified_as_object
    ax.scatter(pred_x[classified_as_object], pred_y[classified_as_object], color='green', marker='o', s=30, label='Object', zorder=6)
    ax.scatter(pred_x[classified_as_no_object], pred_y[classified_as_no_object], color='orange', marker='o', s=30, label='Not Object', zorder=5)

    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"prediction cartisian {time_stamp}")
    ax.grid(True)
    ax.legend()

    #plot_type = 'cartesian' if task_type == 'cartesian' else 'scan_index'
    filename = f"prediction_cartisian_{time_stamp}.png"
    fig.savefig(os.path.join("/home/volle/Desktop/plots/cartisian", filename), bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    

def plot_scan_index(original_gt_i, predictions, classifications_i, distance, DPI, time_stamp):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    #print(f"Worker {worker_id} plotting scan index LiDAR for sample {i}...")
    ax.plot(original_gt_i, label="Ground Truth LiDAR", marker="o")
    #ax.plot(Y_pred_i, label="Predicted LiDAR", linestyle="--", marker="x")
    ignored_gt = original_gt_i > distance
    ax.scatter(np.arange(len(original_gt_i))[ignored_gt], original_gt_i[ignored_gt], color='red', marker='o', label='Ignored GT')
    classified_as_object = classifications_i > 0.5
    classified_as_no_object = ~classified_as_object
    ax.scatter(np.arange(len(predictions))[classified_as_object], predictions[classified_as_object], color='green', marker='o', s=50, label='Object')
    ax.scatter(np.arange(len(predictions))[classified_as_no_object], predictions[classified_as_no_object], color='orange', marker='o', s=50, label='Not Object')
    ax.set_xlabel(f"Scan Index {time_stamp}")
    ax.set_ylabel("Distance (m)")
    ax.set_title(f"{id}")
    ax.grid(True)
    ax.legend()

    #plot_type = 'cartesian' if task_type == 'cartesian' else 'scan_index'
    filename = f"prediction_cartisian_{time_stamp}.png"
    fig.savefig(os.path.join("/home/volle/Desktop/plots/scan_index", filename), bbox_inches='tight', dpi=DPI)
    plt.close(fig)


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

def main_predict():
    DPI = 200
    time_stamp = time.strftime("%d-%m_%H-%M-%S")
    model_lidar, hyperparams_lidar = load_model_regressor(r"Echolocation/Models/2.0m_threshold/echolocation-wide-long-all_200_2_model_regressor.pth")
    model_classifier, hyperparams_classifier = load_model_classifier(r"Echolocation/Models/2.0m_threshold/echolocation-wide-long-all_200_model_classifier.pth")
    #input_features = load_single_row_from_csv(r"Echolocation\FeatureExtraction\ExtractedFeatures\echolocation-wide-long-all\features_all_normalized.csv", 96)
    features = list(FeatureExtractionScript.exstract_single_features_from_wav(r"/home/volle/Desktop/wav").values())
    lidar_file = "/home/volle/Desktop/dist.json"
    with open(lidar_file, "r") as f:
        lidar_data = json.load(f)
    lidar_vector = np.array(lidar_data, dtype=float)
    print(f"Extracted features: {features}")
    print(f"length of features: {len(features)}")
    features = NormalizeFeatures.normalize_single_feature_vector(features, r"Echolocation/FeatureExtraction/ExtractedFeatures/echolocation-wide-long-all/mean_std.csv")
    print(f"Normalized features: {features}")
    #print(input_features.pop(0))  # Remove the first element (sample ID)

    input_tensor = torch.tensor(features, dtype=torch.float32)
    print(f"Input tensor: {input_tensor}")
    print(f"Input tensor shape: {input_tensor.shape}")

    predictions = predict_single_input(model_lidar, input_tensor)
    predictions_classefied = predict_single_input(model_classifier, predictions)
    print(f"Predicted output: {predictions}")
    predicrions_classefied = predictions_classefied.numpy()
    predictions = predictions.numpy()
    predictions = predictions[0]
    print("Predictions:",predictions)
    print("Predictions class:",predicrions_classefied)

    plot_cartisian(lidar_vector, predictions, predicrions_classefied[0], 2.0, DPI, time_stamp)
    plot_scan_index(lidar_vector, predictions, predicrions_classefied[0], 2.0, DPI, time_stamp)
    return
    x, y = polar_to_cartesian(predictions)
    print("x:",x)
    print("y:",y)
    
    # Plot the predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(x,y)
    #plt.plot(predictions[0], label="Predicted Output", marker="o")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Predicted Output Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

    