#!/usr/bin/env python3
import os
import json
import ast
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
from scipy.io.wavfile import write
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from FeatureExtraction import FeatureExtractionScript
from FeatureExtraction import NormalizeFeatures
from MachineLearning.Training.ModelFunctions import Regressor, Classifier
from MachineLearning.Training.EvaluationPlots import plot_confusion_matrix_all


def mean_abserror(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

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


def polar_to_cartesian(distances, angle_range=(-3*np.pi/4, 3*np.pi/4)):
    angles = np.linspace(angle_range[0], angle_range[1], num=len(distances))
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return x, y

def load_model_regressor(device, model_path):
    """ Load the trained model from a path. Epochs and layers are used to construct the file name. """
    print(f"Loading model from {model_path}")
    model_checkpoint = torch.load(model_path,map_location=device)
    
    hyperparams = model_checkpoint['hyperparameters']
    input_dim = hyperparams['input_dim']
    output_dim = hyperparams['output_dim']
    hidden_size = hyperparams['hidden_dim']
    hidden_layer_count = hyperparams['num_layers']
    layer_type = hyperparams['layer_type']
    
    model = Regressor(input_dim, hidden_size, output_dim, hidden_layer_count,layer_type)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    
    return model, hyperparams

def load_model_classifier(device, model_path):
    """ Load the trained model from a path. Epochs and layers are used to construct the file name. """
    print(f"Loading model from {model_path}")
    model_checkpoint = torch.load(model_path,map_location=device)
    
    hyperparams = model_checkpoint['hyperparameters']
    input_dim = hyperparams['input_dim']
    output_dim = hyperparams['output_dim']
    hidden_size = hyperparams['hidden_dim']
    hidden_layer_count = hyperparams['num_layers']
    layer_type = hyperparams['layer_type']
    
    model = Classifier(input_dim, hidden_size, output_dim, hidden_layer_count, layer_type)
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
    filename = f"plot_cartisian.png"
    fig.savefig(os.path.join("Echolocation/MachineLearning/Prediction/", time_stamp, "plots", filename), bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    

def plot_scan_index(original_gt_i, predictions, classifications_i, distance, DPI, time_stamp):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    ignored_gt = original_gt_i > distance
    #print(f"Worker {worker_id} plotting scan index LiDAR for sample {i}...")
    ax.plot(np.arange(len(original_gt_i))[~ignored_gt], original_gt_i[~ignored_gt], label="Ground Truth LiDAR", marker="o")
    #ax.plot(Y_pred_i, label="Predicted LiDAR", linestyle="--", marker="x")
    
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
    filename = f"plot_scan_index.png"
    fig.savefig(os.path.join("Echolocation/MachineLearning/Prediction/", time_stamp, "plots", filename), bbox_inches='tight', dpi=DPI)
    plt.close(fig)



def main_predict():
    DPI = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_stamp = time.strftime("%d-%m_%H-%M-%S")
    model_lidar, hyperparams_lidar = load_model_regressor(device, r"C:/GitHub/AAU_P6_ECHOLOCATION_AND_NAVIGATION/Echolocation/Models/2.0m_threshold/echolocation-wide-long-all_200_2_model_regressor.pth")
    model_classifier, hyperparams_classifier = load_model_classifier(device, r"C:/GitHub/AAU_P6_ECHOLOCATION_AND_NAVIGATION/Echolocation/Models/2.0m_threshold/echolocation-wide-long-all_200_model_classifier.pth")
    #input_features = load_single_row_from_csv(r"Echolocation\FeatureExtraction\ExtractedFeatures\echolocation-wide-long-all\features_all_normalized.csv", 96)
    features = list(FeatureExtractionScript.exstract_single_features_from_wav(r"E:/all test results/distance estimation test/1 - 0.6 meters/14-05_09-07-10/data/wav").values())
    #features = list(FeatureExtractionScript.exstract_single_features_from_wav(r"/home/volle/catkin_ws/Echolocation/Data/dataset/echolocation-wide-long-all/1744893392_33_Wide_Long/1744893392_33_Wide_Long_sound.wav").values())
    lidar_file = "E:/all test results/distance estimation test/1 - 0.6 meters/14-05_09-07-10/LiDAR/LiDAR_true.json"
    #lidar_file = "/home/volle/catkin_ws/Echolocation/Data/dataset/echolocation-wide-long-all/1744893392_33_Wide_Long/1744893392_33_Wide_Long_distance_data.json"
    with open(lidar_file, "r") as f:
        lidar_data = json.load(f)
    
    print("\n\n")
    lidar_vector = np.array(lidar_data, dtype=float)
    #lidar_vector = np.array(lidar_data["LiDAR_distance"], dtype=float)
    print(f"Length of LiDAR vector: {len(lidar_vector)}")
    print(f"LiDAR vector: {lidar_vector}")
    gt_classified = (lidar_vector < 2.0).astype(int)
    #print(f"Extracted features: {features}")
    print(f"length of features: {len(features)}")
    features = NormalizeFeatures.normalize_single_feature_vector(features, r"Echolocation/FeatureExtraction/ExtractedFeatures/echolocation-wide-long-all/mean_std.csv")
    print(f"Normalized features: {features}")
    #print(input_features.pop(0))  # Remove the first element (sample ID)

    input_tensor = torch.tensor(features, dtype=torch.float32)
    #print(f"Input tensor: {input_tensor}")
    #print(f"Input tensor shape: {input_tensor.shape}")

    predictions = predict_single_input(model_lidar, input_tensor)
    predictions_classefied = predict_single_input(model_classifier, predictions)
    
    #print(f"Predicted output: {predictions}")
    predicrions_classefied = predictions_classefied.numpy()[0]
    predictions_classefied_thresholded = (predicrions_classefied > 0.5).astype(int)
    predictions = predictions.numpy()
    predictions = predictions[0]
    
    print("Predictions:",predictions)
    print("Predictions class:",predicrions_classefied)

    output_path = os.path.join("Echolocation/MachineLearning/Prediction/", time_stamp)
    os.makedirs(os.path.join(output_path, "plots"), exist_ok=True)

    lidar_path = os.path.join(output_path, "LiDAR", "LiDAR_true.json")
    os.makedirs(os.path.join(output_path, "LiDAR"), exist_ok=True)
    lidar_pred_path = os.path.join(output_path, "LiDAR", "LiDAR_prediction.json")
    os.makedirs(os.path.join(output_path, "LiDAR"), exist_ok=True)
    class_path = os.path.join(output_path, "classifications", "classifications_true.json")
    os.makedirs(os.path.join(output_path, "classifications"), exist_ok=True)
    class_pred_path = os.path.join(output_path, "classifications", "classifications_prediction.json")
    os.makedirs(os.path.join(output_path, "classifications"), exist_ok=True)
    # save wav, lidar to new folder
    with open(lidar_path, "w") as f:
        json.dump(lidar_data, f, indent=4)
    with open(lidar_pred_path, "w") as f:
        json.dump(predictions.tolist(), f, indent=4)
    with open(class_path, "w") as f:
        json.dump(gt_classified.tolist(), f)
    with open(class_pred_path, "w") as f:
        json.dump(predictions_classefied_thresholded.tolist(), f)

    # wav
    wav_path = os.path.join(output_path, "data", "wav.wav")
    os.makedirs(os.path.join(output_path, "data"), exist_ok=True)
    stereo_signal, sr = sf.read("E:/all test results/distance estimation test/1 - 0.6 meters/14-05_09-07-10/data/wav", dtype="float32")
    write(wav_path, sr, stereo_signal)
    
    plot_cartisian(lidar_vector, predictions, predicrions_classefied, 2.0, DPI, time_stamp)
    plot_scan_index(lidar_vector, predictions, predicrions_classefied, 2.0, DPI, time_stamp)

    
    confusion_path = os.path.join(output_path, "plots", f"confusion_matrix")
    plot_confusion_matrix_all(gt_classified, predicrions_classefied, save_path=confusion_path)

    # metrics regressor
    print("Regressor metrics:")
    root_mean_squared_errors = root_mean_squared_error(lidar_vector, predictions)
    print(f"    RMSE: {root_mean_squared_errors}")
    r2 = r2_score(lidar_vector, predictions)
    print(f"    R-squared value = {r2}")
    meanABS = mean_abserror(lidar_vector, predictions)
    print(f"    Mean Absolute Error = {meanABS}")

    regressor_error_metrics = {
        "RMSE" : root_mean_squared_errors,
        "R-squared" : r2,
        "Means Absolute Error" : meanABS,
    }
    regressor_metrics = os.path.join(output_path, "metrics", "metrics_regressor.json")
    os.makedirs(os.path.join(output_path, "metrics"), exist_ok=True)
    with open(regressor_metrics, "w") as file:
        json.dump(regressor_error_metrics, file, indent=4)

    # metrics classifier
    print("Classifier metrics:")
    class_accuracy = accuracy_score(gt_classified, predictions_classefied_thresholded)
    print(f"    Accuracy: {class_accuracy}")
    class_precision = precision_score(gt_classified, predictions_classefied_thresholded, zero_division=0)
    print(f"    Precision: {class_precision}")
    class_recall = recall_score(gt_classified, predictions_classefied_thresholded, zero_division=0)
    print(f"    recall: {class_recall}")
    class_f1 = f1_score(gt_classified, predictions_classefied_thresholded, zero_division=0)
    print(f"    f1: {class_f1}")

    tn, fp, fn, tp = confusion_matrix(gt_classified, predictions_classefied_thresholded).ravel()
    fp_rate = fp/(fp+tn)
    print(f"    FPR: {fp_rate}")
    fn_rate = fn/(fn+tp)
    print(f"    FNR: {fn_rate}")

    classifier_error_metrics = {
        "accuracy" : class_accuracy,
        "precision": class_precision,
        "recall" : class_recall,
        "f1" : class_f1,
        "false positive rate" : fp_rate,
        "false negative rate" : fn_rate,
    }
    classifier_metrics = os.path.join(output_path, "metrics", "metrics_classifier.json")
    with open(classifier_metrics, "w") as file:
        json.dump(classifier_error_metrics, file, indent=4)




    