#!/usr/bin/env python3
import os
import json
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from queue import Queue
from threading import Thread

from sklearn.model_selection import train_test_split

distance_threshold = 2

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
      - original_distances: List of original LiDAR distances for plotting
    """
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

    # Set of sample numbers to skip
    skip_ids = {"1", "10", "50", "53", "69", "97", "114", "128", "129", "149", "157", "181", "199", "200", "234", "250", "263", "283", "396", "441", "465", "472" , "477", "502", "527", "538", "645", "668", "686", "697", "713"}  # Add more as needed

    for idx, row in df.iterrows():
        filename = row["filename"]

        if any(f"_{sid}_" in filename for sid in skip_ids):
            print(f"Skipping file due to skip list: {filename}")
            continue

        lidar_filename = f"{filename}_distance_data.json"
        lidar_file = os.path.join(dataset_root, filename, lidar_filename)

        #print(f"Processing file: {filename}")

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
            original_distances.append(lidar_vector.copy())  # Store original distances for plotting

            # Mark distances above distance_threshold as NaN
            lidar_vector[lidar_vector > distance_threshold] = np.nan

        except Exception as e:
            print(f"Error loading LiDAR file {lidar_file}: {e}. Skipping sample {filename}.")
            continue

        X_list.append(feature_vector)
        Y_list.append(lidar_vector)
        sample_ids.append(filename)

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, sample_ids, feature_names_full, original_distances

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        # Mask values where targets are NaN or greater than distance_threshold
        mask = ~torch.isnan(targets) & (targets <= distance_threshold)
        outputs = outputs[mask]
        targets = targets[mask]
        loss = self.mse_loss(outputs, targets)
        return loss.mean()

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
        A simple feedforward neural network for regression and classification.

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
        self.regression_head = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))
        self.classification_head = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        shared = self.regression_head[:-1](x)  # Shared layers
        regression_output = self.regression_head[-1](shared)
        classification_output = self.classification_head(shared)
        return regression_output, classification_output

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, optimizer, regression_loss_fn, classification_loss_fn, device, num_epochs):
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            regression_outputs, classification_outputs = model(X_batch)
            regression_loss = regression_loss_fn(regression_outputs, Y_batch)
            classification_targets = (Y_batch <= distance_threshold).float()
            classification_loss = classification_loss_fn(classification_outputs, classification_targets)
            loss = regression_loss + classification_loss
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
                regression_outputs, classification_outputs = model(X_batch)
                regression_loss = regression_loss_fn(regression_outputs, Y_batch)
                classification_targets = (Y_batch <= distance_threshold).float()
                classification_loss = classification_loss_fn(classification_outputs, classification_targets)
                loss = regression_loss + classification_loss
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    return best_val_loss, best_model_state

def compute_permutation_importance(model, X_val, Y_val, loss_fn, device):
    model.eval()
    base_preds, _ = model(torch.tensor(X_val, dtype=torch.float32).to(device))
    base_loss = loss_fn(base_preds, torch.tensor(Y_val, dtype=torch.float32).to(device)).item()

    importances = []
    for i in range(X_val.shape[1]):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, i])  # Shuffle i-th feature column
        permuted_preds, _ = model(torch.tensor(X_permuted, dtype=torch.float32).to(device))
        permuted_loss = loss_fn(permuted_preds, torch.tensor(Y_val, dtype=torch.float32).to(device)).item()
        importance = permuted_loss - base_loss
        importances.append(importance)

    return importances

def save_feature_importance(chosen_dataset, best_model, X_val, Y_val, loss_fn, device, feature_names_full, num_epochs, num_layers):
    # compute feature importance
    importances = compute_permutation_importance(best_model, X_val, Y_val, loss_fn, device)

    # save feature importance to csv
    importance_df = pd.DataFrame({
        "Feature": feature_names_full,
        "Importance": importances
    })
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)

    base_file_dir = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", "feature_importance", chosen_dataset)
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

def polar_to_cartesian(distances, angle_range=(-3*np.pi/4, 3*np.pi/4)):  # 270 degrees centered forward
    angles = np.linspace(angle_range[0], angle_range[1], num=len(distances))
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return x, y

def model_training(dataset_root_directory, chosen_dataset):
    # Placeholder function for training the model.
    # This should be replaced with the actual training logic.
    csv_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset + "_features_all_normalized.csv")

    X, Y, sample_ids, feature_names_full, original_distances = build_dataset_from_csv(csv_file, dataset_root_directory)
    if X.shape[0] == 0:
        print("No valid samples found. Exiting.")
        return
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, LiDAR scan length: {Y.shape[1]}")

    print("Checking for NaNs or Infs...")
    print("X contains NaNs:", np.isnan(X).any())
    print("X contains Infs:", np.isinf(X).any())
    print("Y contains NaNs:", np.isnan(Y).any())
    print("Y contains Infs:", np.isinf(Y).any())

    print("X stats: min", np.min(X), "max", np.max(X))
    print("Y stats: min", np.min(Y), "max", np.max(Y))

    # Split dataset into train (70%), validation (15%), and test (15%)
    X_train_val, X_test, Y_train_val, Y_test, original_distances_train_val, original_distances_test = train_test_split(X, Y, original_distances, test_size=0.15, random_state=42)
    X_train, X_val, Y_train, Y_val, original_distances_train, original_distances_val = train_test_split(X_train_val, Y_train_val, original_distances_train_val, test_size=0.1765, random_state=42)  # 0.1765*0.85 ~ 15%
    print("Training samples: ", X_train.shape[0], "Validation samples: ", X_val.shape[0], "Test samples: ", X_test.shape[0])

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

    # Hyperparameters
    learning_rates = [0.001]
    hidden_sizes = [256]
    num_epochs = 100  # You may adjust epochs
    num_layers = 2  # Number of hidden layers
    classification_thresholds = [0.5]  # Thresholds to try

    best_overall_val_loss = 0
    best_hyperparams = None
    best_model_state = None
    best_threshold = None

    # Define the global custom loss function
    regression_loss_fn = MaskedMSELoss()
    classification_loss_fn = nn.BCELoss()

    # Grid search over hyperparameters.
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for threshold in classification_thresholds:
                print(f"\nTraining with learning rate={lr}, hidden_size={hidden_size}, classification_threshold={threshold}")
                model = MLPRegressor(input_dim, hidden_size, output_dim, num_layers=num_layers).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                val_loss, model_state = train_model(model, train_loader, val_loader, optimizer, regression_loss_fn, classification_loss_fn, device, num_epochs)

                # Evaluate classification accuracy on the validation set
                model.eval()
                val_classification_accuracies = []
                with torch.no_grad():
                    for X_batch, Y_batch in val_loader:
                        X_batch = X_batch.to(device)
                        Y_batch = Y_batch.to(device)
                        regression_outputs, classification_outputs = model(X_batch)
                        classification_targets = (Y_batch <= distance_threshold).float()
                        classification_preds = (classification_outputs > threshold).float()
                        accuracy = (classification_preds == classification_targets).float().mean().item()
                        val_classification_accuracies.append(accuracy)
                avg_val_classification_accuracy = np.mean(val_classification_accuracies)

                print(f"Hyperparams (lr={lr}, hidden_size={hidden_size}, threshold={threshold}) - Best Val Loss: {val_loss:.4f}, Val Classification Accuracy: {avg_val_classification_accuracy:.4f}")

                print("avg val classification accuracy:", avg_val_classification_accuracy, "best overall val loss:", best_overall_val_loss, "best threshold:", best_threshold)
                # Update best hyperparameters if the current configuration is better
                if avg_val_classification_accuracy > best_overall_val_loss:
                    print("New best hyperparameters found!")
                    best_overall_val_loss = avg_val_classification_accuracy
                    best_hyperparams = {
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                        "lr": lr,
                        "hidden_size": hidden_size,
                        "num_epochs": num_epochs,
                        "num_layers": num_layers
                    }
                    best_model_state = model_state
                    best_threshold = threshold

    # Check if best_hyperparams is not None before accessing its elements
    if best_hyperparams is None:
        print("Error: No valid hyperparameters found during grid search.")
        return

    print("\nBest hyperparameters found:")
    print(best_hyperparams)
    print(f"Best classification threshold: {best_threshold}")
    print(f"Best validation classification accuracy: {best_overall_val_loss:.4f}")

    # Build the best model and load best state
    best_model = MLPRegressor(input_dim, best_hyperparams["hidden_size"], output_dim, num_layers=num_layers).to(device)
    best_model.load_state_dict(best_model_state)

    # Evaluate on the test set.
    best_model.eval()
    test_losses = []
    all_preds = []
    all_targets = []
    all_classifications = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            regression_outputs, classification_outputs = best_model(X_batch)
            regression_loss = regression_loss_fn(regression_outputs, Y_batch)
            classification_targets = (Y_batch <= distance_threshold).float()
            classification_loss = classification_loss_fn(classification_outputs, classification_targets)
            loss = regression_loss + classification_loss
            test_losses.append(loss.item())
            all_preds.append(regression_outputs.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())
            all_classifications.append(classification_outputs.cpu().numpy())
    avg_test_loss = np.mean(test_losses)
    print(f"\nMean Squared Error on test set: {avg_test_loss:.4f}")

    # Concatenate predictions for plotting.
    Y_pred = np.concatenate(all_preds, axis=0)
    Y_true = np.concatenate(all_targets, axis=0)
    classifications = np.concatenate(all_classifications, axis=0)



    # currently broken - will fix later
    #save_feature_importance(chosen_dataset, best_model, X_val, Y_val, loss_fn, device, feature_names_full, num_epochs, num_layers)

    # Save the best model.
    models_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", "models")
    os.makedirs(models_folder, exist_ok=True)
    model_file = os.path.join(models_folder, f"{chosen_dataset}_{num_epochs}_{num_layers}_model.pth")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "hyperparameters": best_hyperparams,
        "classification_threshold": best_threshold
    }, model_file)

    print("Saving comparison plots...")
    # Create a queue for plot tasks
    plot_queue = Queue()
    # Adjust automatically based on your CPU cores
    num_workers = int(os.cpu_count()/2)

    def plot_worker(worker_id):
        while True:
            task = plot_queue.get()
            if task is None:  # Sentinel value to stop worker
                print(f"Worker {worker_id} shutting down")
                plot_queue.task_done()
                break
            try:
                task_type, data = task
                i, Y_true_i, Y_pred_i, classifications_i, original_gt_i, num_epochs, num_layers, lidar_plots_folder = data

                if task_type == 'cartesian':
                    print(f"Worker {worker_id} saving cartesian plot for sample {i}")
                    gt_x, gt_y = polar_to_cartesian(Y_true_i)
                    pred_x, pred_y = polar_to_cartesian(Y_pred_i)
                    ignored_gt = original_gt_i > distance_threshold

                    plt.figure(figsize=(8, 8))
                    ignored_gt_x, ignored_gt_y = polar_to_cartesian(original_gt_i)
                    plt.scatter(ignored_gt_x[ignored_gt], ignored_gt_y[ignored_gt], color='red', marker='o',
                                label='Ignored GT', alpha=0.7, zorder=1)
                    plt.plot(gt_x, gt_y, label="Ground Truth LiDAR", marker='o', linestyle='-', alpha=0.7, zorder=2)
                    plt.plot(pred_x, pred_y, label="Predicted LiDAR", marker='x', linestyle='--', alpha=0.7, zorder=3)

                    robot_circle = plt.Circle((0, 0), 0.2, color='gray', fill=True, alpha=0.5, label='Robot', zorder=2)
                    plt.gca().add_patch(robot_circle)

                    plt.plot([0, gt_x[0]], [0, gt_y[0]], color='blue', linestyle='--', alpha=0.5, zorder=3)
                    plt.plot([0, pred_x[0]], [0, pred_y[0]], color='red', linestyle='--', alpha=0.5, zorder=3)
                    plt.plot([0, gt_x[-1]], [0, gt_y[-1]], color='blue', linestyle='--', alpha=0.5, zorder=3)
                    plt.plot([0, pred_x[-1]], [0, pred_y[-1]], color='red', linestyle='--', alpha=0.5, zorder=3)

                    plt.arrow(0, 0, gt_x[540], gt_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black',
                              alpha=1, zorder=4)
                    plt.arrow(0, 0, pred_x[540], pred_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black',
                              alpha=1, zorder=4)

                    classified_as_object = classifications_i > best_threshold
                    classified_as_no_object = ~classified_as_object

                    plt.scatter(pred_x[classified_as_object], pred_y[classified_as_object], color='green', marker='o',
                                s=50, zorder=5, label='Classified as Object')
                    plt.scatter(pred_x[classified_as_no_object], pred_y[classified_as_no_object], color='orange',
                                marker='o', s=50, zorder=5, label='Classified as No Object')

                    plt.axis('equal')
                    plt.xlabel("X (m)")
                    plt.ylabel("Y (m)")
                    plt.title(f"LiDAR 2D View - Scan {i} : {num_epochs} epochs : {num_layers} layers")
                    plt.grid(True)
                    plt.legend()

                    lidar_plot_file = os.path.join(lidar_plots_folder, f"lidar_prediction_cartesian_{i}.png")
                    plt.savefig(lidar_plot_file)
                    plt.close()

                elif task_type == 'scan_index':
                    print(f"Worker {worker_id} saving scan index plot for sample {i}")
                    plt.figure(figsize=(10, 6))
                    plt.plot(Y_true_i, label="Ground Truth LiDAR", marker="o")
                    plt.plot(Y_pred_i, label="Predicted LiDAR", linestyle="--", marker="x")

                    ignored_gt = original_gt_i > distance_threshold
                    plt.scatter(np.arange(len(original_gt_i))[ignored_gt], original_gt_i[ignored_gt], color='red',
                                marker='o', label='Ignored GT')

                    classified_as_object = classifications_i > best_threshold
                    classified_as_no_object = ~classified_as_object

                    plt.scatter(np.arange(len(Y_pred_i))[classified_as_object], Y_pred_i[classified_as_object],
                                color='green', marker='o', s=50, zorder=5, label='Classified as Object')
                    plt.scatter(np.arange(len(Y_pred_i))[classified_as_no_object], Y_pred_i[classified_as_no_object],
                                color='orange', marker='o', s=50, zorder=5, label='Classified as No Object')

                    plt.xlabel("Scan Index")
                    plt.ylabel("Distance (m)")
                    plt.title(f"LiDAR Scan {i} : {num_epochs} epochs : {num_layers} layers")
                    plt.legend()
                    plt.grid(True)

                    lidar_plot_file = os.path.join(lidar_plots_folder, f"lidar_prediction_{i}.png")
                    plt.savefig(lidar_plot_file)
                    plt.close()

            except Exception as e:
                print(f"Worker {worker_id} encountered error with sample {i}: {e}")
            finally:
                plot_queue.task_done()

    # Start worker threads with IDs
    workers = []
    for worker_id in range(num_workers):
        t = Thread(target=plot_worker, args=(worker_id,))
        t.start()
        workers.append(t)
        print(f"Started worker thread {worker_id}")

    print("Saving comparison plots using multi-threading...")

    # Create folders
    cartesian_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures",
                                    f"{chosen_dataset}_lidar_plots_{num_epochs}_{num_layers}", "cartesian")
    scan_index_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures",
                                     f"{chosen_dataset}_lidar_plots_{num_epochs}_{num_layers}", "scan_index")
    os.makedirs(cartesian_folder, exist_ok=True)
    os.makedirs(scan_index_folder, exist_ok=True)

    # Add plot tasks to queue
    for i in range(len(Y_true)):
        plot_queue.put(('cartesian',
                        (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i],
                         num_epochs, num_layers, cartesian_folder)))
        plot_queue.put(('scan_index',
                        (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i],
                         num_epochs, num_layers, scan_index_folder)))

    # Wait for all tasks to complete
    plot_queue.join()

    # Stop workers
    for _ in range(num_workers):
        plot_queue.put(None)
    for t in workers:
        t.join()

    print(f"Best model saved to {model_file}")

    # Evaluate classification accuracy
    classification_accuracy = (classifications.round() == (Y_true <= distance_threshold)).mean()
    print(f"Classification Accuracy: {classification_accuracy:.4f}")
