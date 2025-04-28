#!/usr/bin/env python3
import os
import json
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from TrainingConfig import DISTANCE_THRESHOLD

def build_dataset_from_csv(csv_file, dataset_root):
    df = pd.read_csv(csv_file)
    df.sort_values("filename", inplace=True)
    print("DataFrame preview:")
    print(df.head())

    X_list = []
    Y_list = []
    original_distances = []
    sample_ids = []
    feature_names_full = []

    feature_cols = [col for col in df.columns if col != "filename"]

    for idx, row in df.iterrows():
        filename = row["filename"]
        lidar_filename = f"{filename}_distance_data.json"
        lidar_file = os.path.join(dataset_root, filename, lidar_filename)

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

        if not os.path.exists(lidar_file):
            print(f"LiDAR file {lidar_file} not found. Skipping sample {filename}.")
            continue

        try:
            with open(lidar_file, "r") as f:
                lidar_data = json.load(f)
            lidar_vector = np.array(lidar_data["LiDAR_distance"], dtype=float)
            original_distances.append(lidar_vector.copy())
            lidar_vector[lidar_vector > DISTANCE_THRESHOLD] = np.nan

        except Exception as e:
            print(f"Error loading LiDAR file {lidar_file}: {e}. Skipping sample {filename}.")
            continue

        X_list.append(feature_vector)
        Y_list.append(lidar_vector)
        sample_ids.append(filename)

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, sample_ids, feature_names_full, original_distances
