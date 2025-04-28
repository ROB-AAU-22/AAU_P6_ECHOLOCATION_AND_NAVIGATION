#!/usr/bin/env python3
import os
import json
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from MachineLearning.Training.TrainingConfig import DISTANCE_THRESHOLD


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
          - original_distances: List of original LiDAR distances for plotting
        """
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Sort DataFrame by 'filename' to maintain order
    df.sort_values("filename", inplace=True)
    print("DataFrame preview:")
    print(df.head())

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
            lidar_vector[lidar_vector > DISTANCE_THRESHOLD] = np.nan

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
