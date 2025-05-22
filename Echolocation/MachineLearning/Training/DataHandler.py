#!/usr/bin/env python3
import os
import json
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from MachineLearning.Training.TrainingConfig import DISTANCE_THRESHOLD, DISTANCE_THRESHOLD_ENABLED


def build_dataset_from_csv(csv_file, dataset_root, distance_threshold=DISTANCE_THRESHOLD):
    """
    Builds a dataset from a CSV file and corresponding LiDAR JSON files.
    This function reads a CSV file containing feature data and associates each sample with a LiDAR distance data JSON file.
    It processes feature columns (including list-like string values), loads and thresholds LiDAR distance data, and returns
    arrays suitable for machine learning tasks.
    Args:
        csv_file (str): Path to the CSV file containing feature data. The CSV must include a 'filename' column.
        dataset_root (str): Root directory where sample folders and LiDAR JSON files are located.
        distance_threshold (float, optional): Maximum allowed LiDAR distance. Distances above this value are replaced with NaN.
            Defaults to DISTANCE_THRESHOLD.
    Returns:
        tuple:
            - X (np.ndarray): Array of feature vectors for each sample.
            - Y (np.ndarray): Array of LiDAR distance vectors (labels) for each sample.
            - sample_ids (list): List of sample identifiers (filenames).
            - feature_names_full (list): List of feature names corresponding to columns in X.
            - original_distances (list): List of original (unthresholded) LiDAR distance arrays for each sample.
    Notes:
        - The function skips samples if the corresponding LiDAR JSON file is missing or cannot be loaded.
        - Feature columns containing list-like strings (e.g., "[1.0, 2.0]") are expanded into multiple features.
        - Requires global variables DISTANCE_THRESHOLD and DISTANCE_THRESHOLD_ENABLED to be defined.
    """

    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Sort DataFrame by 'filename' to maintain order
    df.sort_values("filename", inplace=True)
    #print("DataFrame preview:")
    #print(df.head())

    # Initialize lists to hold features, labels, metadata, and feature names
    X_list = []
    Y_list = []
    original_distances = []
    sample_ids = []
    feature_names_full = []

    # Identify all feature columns except 'filename'
    feature_cols = [col for col in df.columns if col != "filename"]

    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():
        # Get file details
        filename = row["filename"]
        lidar_filename = f"{filename}_distance_data.json"
        lidar_file = os.path.join(dataset_root, filename, lidar_filename)

        # Prepare the feature vector and corresponding names
        feature_vector = []
        feature_names_row = []
        for col in feature_cols:
            cell = row[col]
            # Handle list-like string values
            if isinstance(cell, str) and cell.strip().startswith('[') and cell.strip().endswith(']'):
                try:
                    parsed_value = ast.literal_eval(cell)  # Parse string to list
                    if isinstance(parsed_value, list):
                        # Extend feature vector with parsed list values
                        feature_vector.extend([float(x) for x in parsed_value])
                        # Generate names for each feature in the list
                        feature_names_row.extend([f"{col}_{i}" for i in range(len(parsed_value))])
                    else:
                        # Handle single scalar value
                        feature_vector.append(float(parsed_value))
                        feature_names_row.append(col)
                except Exception as e:
                    print(f"Error parsing column '{col}' with value '{cell}': {e}")
                    continue
            else:
                try:
                    # Handle single scalar value
                    feature_vector.append(float(cell))
                    feature_names_row.append(col)
                except Exception as e:
                    print(f"Error converting cell in column '{col}' with value '{cell}' to float: {e}")
                    continue

        # Set feature names for the first sample
        if idx == 0:
            feature_names_full = feature_names_row

        # Check if the corresponding LiDAR file exists
        if not os.path.exists(lidar_file):
            print(f"LiDAR file {lidar_file} not found. Skipping sample {filename}.")
            continue

        try:
            # Load LiDAR data from JSON file
            with open(lidar_file, "r") as f:
                lidar_data = json.load(f)
            # Extract LiDAR distances
            lidar_vector = np.array(lidar_data["LiDAR_distance"], dtype=float)
            # Save original distances for later analysis
            original_distances.append(lidar_vector.copy())
            # Apply distance threshold, replacing values above it with NaN
            if DISTANCE_THRESHOLD_ENABLED:
                lidar_vector[lidar_vector > (distance_threshold)] = np.nan
                #lidar_vector[lidar_vector > (distance_threshold)] = 2.1

        except Exception as e:
            print(f"Error loading LiDAR file {lidar_file}: {e}. Skipping sample {filename}.")
            continue

        # Collect processed feature and label vectors
        X_list.append(feature_vector)
        Y_list.append(lidar_vector)
        sample_ids.append(filename)

    # Convert lists to NumPy arrays
    X = np.array(X_list)
    Y = np.array(Y_list)

    # Return features, labels, sample IDs, feature names, and original distances
    return X, Y, sample_ids, feature_names_full, original_distances


class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, predicted_distances, original_distances):
        #print(f"train preds: {predicted_distances}")
        #print(f"train targets: {original_distances}")
        self.X = np.float32(predicted_distances)#.float()
        self.Y = np.float32(~np.isnan(original_distances))#.float()
        #self.Y = original_distances  # .float()


    def __len__(self):
        return len(self.X)#.shape[0]

    def __getitem__(self, idx):
        #print(f"train preds: {np.float32(self.X[idx])}")
        #print(f"train targets: {np.float32(self.Y[idx])}")
        #self.Y = ~np.isnan(self.Y[idx])
        return self.X[idx], self.Y[idx]