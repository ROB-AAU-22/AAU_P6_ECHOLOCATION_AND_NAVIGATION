#!/usr/bin/env python3
import os
import json
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for multiprocessing
import matplotlib.pyplot as plt
import time
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Process, Queue, cpu_count
from sklearn.model_selection import train_test_split

DISTANCE_THRESHOLD = 2


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
    # skip_ids = {"1", "10", "29", "50", "53", "69", "97", "114", "128", "129", "149", "157", "181", "199", "200", "234", "250",
    #            "263", "283", "396", "441", "465", "472", "477", "502", "522", "527", "538", "645", "668", "686", "697",
    #            "713", "876"}  # Add more as needed

    #skip_ids = {"188", "203" ,"1196", "1214", "1243"}

    for idx, row in df.iterrows():
        filename = row["filename"]

        #if any(f"_{sid}_" in filename for sid in skip_ids):
        #    print(f"Skipping file due to skip list: {filename}")
        #    continue

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

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        # Mask values where targets are NaN or greater than distance_threshold
        mask = ~torch.isnan(targets)# & (targets <= DISTANCE_THRESHOLD)
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
def train_model(model, train_loader, val_loader, optimizer, regression_loss_fn, classification_loss_fn,
                device, num_epochs, patience=10, scheduler=None):
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()
            regression_outputs, classification_outputs = model(X_batch)
            regression_loss = regression_loss_fn(regression_outputs, Y_batch)
            classification_targets = (Y_batch <= DISTANCE_THRESHOLD).float()
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
                classification_targets = (Y_batch <= DISTANCE_THRESHOLD).float()
                classification_loss = classification_loss_fn(classification_outputs, classification_targets)
                loss = regression_loss + classification_loss
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break

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

def plot_worker(queue, worker_id):
    import matplotlib.pyplot as plt
    while True:
        task = queue.get()
        if task is None:
            print(f"Worker {worker_id} shutting down")
            break
        try:
            task_type, data = task
            i, Y_true_i, Y_pred_i, classifications_i, original_gt_i, num_epochs, num_layers, lidar_plots_folder, best_threshold, chosen_dataset = data

            fig, ax = plt.subplots(figsize=(8, 8) if task_type == 'cartesian' else (10, 6))
            if task_type == 'cartesian':
                print(f"Worker {worker_id} plotting cartesian LiDAR for sample {i}...")
                gt_x, gt_y = polar_to_cartesian(Y_true_i)
                pred_x, pred_y = polar_to_cartesian(Y_pred_i)
                ignored_gt = original_gt_i > DISTANCE_THRESHOLD

                ignored_gt_x, ignored_gt_y = polar_to_cartesian(original_gt_i)
                ax.scatter(ignored_gt_x[ignored_gt], ignored_gt_y[ignored_gt], color='red', marker='o', label='Ignored GT', alpha=0.7, zorder=1)
                ax.plot(gt_x, gt_y, label="Ground Truth LiDAR", marker='o', linestyle='-', alpha=0.7, zorder=2)
                #ax.plot(pred_x, pred_y, label="Predicted LiDAR", linestyle='--', alpha=0.7, zorder=3)

                robot_circle = plt.Circle((0, 0), 0.2, color='gray', fill=True, alpha=0.5, label='Robot', zorder=2)
                ax.add_patch(robot_circle)

                # draw a line from origin to first scan point

                plt.plot([0, gt_x[0]], [0, gt_y[0]], color='blue', linestyle='--', alpha=0.5, zorder=3)

                plt.plot([0, pred_x[0]], [0, pred_y[0]], color='red', linestyle='--', alpha=0.5, zorder=3)

                # draw a line from origin to last scan point

                plt.plot([0, gt_x[-1]], [0, gt_y[-1]], color='blue', linestyle='--', alpha=0.5, zorder=3)

                plt.plot([0, pred_x[-1]], [0, pred_y[-1]], color='red', linestyle='--', alpha=0.5, zorder=3)

                # draw an arrow vector from origin to middle point(s)

                plt.arrow(0, 0, gt_x[540], gt_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black', alpha=1,
                          zorder=4)

                plt.arrow(0, 0, pred_x[540], pred_y[540], head_width=0.1, head_length=0.2, fc='black', ec='black',
                          alpha=1, zorder=4)

                classified_as_object = classifications_i > best_threshold
                classified_as_no_object = ~classified_as_object
                ax.scatter(pred_x[classified_as_object], pred_y[classified_as_object],
                           color='green', marker='o', s=30, label='Object')
                ax.scatter(pred_x[classified_as_no_object], pred_y[classified_as_no_object],
                           color='orange', marker='o', s=30, label='Not Object')

                ax.set_aspect('equal')
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_title(f"{chosen_dataset}-{i}")
                ax.grid(True)
                ax.legend()
            else:
                print(f"Worker {worker_id} plotting scan index LiDAR for sample {i}...")
                ax.plot(Y_true_i, label="Ground Truth LiDAR", marker="o")
                ax.plot(Y_pred_i, label="Predicted LiDAR", linestyle="--", marker="x")
                ignored_gt = original_gt_i > DISTANCE_THRESHOLD
                ax.scatter(np.arange(len(original_gt_i))[ignored_gt], original_gt_i[ignored_gt], color='red', marker='o', label='Ignored GT')
                classified_as_object = classifications_i > best_threshold
                classified_as_no_object = ~classified_as_object
                ax.scatter(np.arange(len(Y_pred_i))[classified_as_object], Y_pred_i[classified_as_object], color='green', marker='o', s=50, label='Object')
                ax.scatter(np.arange(len(Y_pred_i))[classified_as_no_object], Y_pred_i[classified_as_no_object], color='orange', marker='o', s=50, label='Not Object')
                ax.set_xlabel("Scan Index")
                ax.set_ylabel("Distance (m)")
                ax.set_title(f"{chosen_dataset}-{i} \n Epochs: {num_epochs}, Layers: {num_layers}")
                ax.grid(True)
                ax.legend()

            plot_type = 'cartesian' if task_type == 'cartesian' else 'scan_index'
            filename = f"lidar_prediction_{plot_type}_{i}.png"
            fig.savefig(os.path.join(lidar_plots_folder, filename), bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"Error in worker {worker_id} for task {i}: {e}")

def start_multiprocessing_plotting(Y_true, Y_pred, classifications, original_distances_test, num_epochs, num_layers, cartesian_folder, scan_index_folder, best_threshold, chosen_dataset):
    start_time = time.time()
    num_workers = int(cpu_count())
    print(f"Using {num_workers} multiprocessing workers for plotting")

    task_queue = Queue()
    workers = [Process(target=plot_worker, args=(task_queue, i)) for i in range(num_workers)]
    for w in workers:
        w.start()

    for i in range(len(Y_true)):
        task_queue.put(('cartesian', (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i], num_epochs, num_layers, cartesian_folder, best_threshold, chosen_dataset)))
        task_queue.put(('scan_index', (i, Y_true[i], Y_pred[i], classifications[i], original_distances_test[i], num_epochs, num_layers, scan_index_folder, best_threshold, chosen_dataset)))

    for _ in range(num_workers):
        task_queue.put(None)

    for w in workers:
        w.join()

    print(f"Multiprocessing plotting completed in {time.time() - start_time:.2f} seconds")

def compute_error_metrics(Y_true, Y_pred):
    range_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    results = {f"{low}_{high}": {'y_true': [], 'y_pred': []} for low, high in range_bins}
    
    for y_true, y_pred in zip(Y_true, Y_pred):
        for low, high in range_bins:
            if low <= y_true < high:
                key = f"{low}_{high}"
                results[key]['y_true'].append(y_true)
                results[key]['y_pred'].append(y_pred)
                break  # Exit the loop once the correct bin is found
    
    error_metrics_results = {}
    
    for (low, high) in range_bins:
        y_true_arr = np.array(results[f"{low}_{high}"]['y_true'])
        y_pred_arr = np.array(results[f"{low}_{high}"]['y_pred'])
        
        if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
            continue
            
        error_metrics_results[f"{low}_{high}"] = {
            'mae': np.mean(np.abs(y_true_arr - y_pred_arr)),
            'rmse': np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)),
            'mre': np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-10))) if np.any(y_true_arr) else 0,
        }
    
    return error_metrics_results

def model_training(dataset_root_directory, chosen_dataset):
    # Placeholder function for training the model.
    # This should be replaced with the actual training logic.
    
    csv_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, "features_all_normalized.csv")
    
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

    train_dataset = AudioLidarDataset(X_train, Y_train)
    val_dataset = AudioLidarDataset(X_val, Y_val)
    test_dataset = AudioLidarDataset(X_test, Y_test)

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    learning_rates = [0.01]
    hidden_sizes = [128, 256, 512, 1024]
    batch_sizes = [32, 64, 128, 256]
    num_epochs = 200
    num_layers_list = [2,3]
    classification_thresholds = [0.5]

    best_overall_val_loss = 0
    best_hyperparams = None
    best_model_state = None
    best_threshold = None

    # Define the global custom loss function
    regression_loss_fn = MaskedMSELoss()
    classification_loss_fn = nn.BCELoss()

    # Grid search over hyperparameters
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                for num_layers in num_layers_list:
                    for threshold in classification_thresholds:
                        print(
                            f"\nTraining with: lr={lr}, hidden_size={hidden_size}, batch_size={batch_size}, layers={num_layers}, threshold={threshold}")

                        # Create dataloaders with current batch size
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                        model = MLPRegressor(input_dim, hidden_size, output_dim, num_layers=num_layers).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                        # Add scheduler for learning rate
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

                        # Call with early stopping and learning rate scheduler
                        val_loss, model_state = train_model(
                            model,
                            train_loader,
                            val_loader,
                            optimizer,
                            regression_loss_fn,
                            classification_loss_fn,
                            device,
                            num_epochs,
                            patience=10,
                            scheduler=scheduler
                        )

                        # Evaluate classification accuracy on the validation set
                        model.eval()
                        val_classification_accuracies = []
                        with torch.no_grad():
                            for X_batch, Y_batch in val_loader:
                                X_batch = X_batch.to(device)
                                Y_batch = Y_batch.to(device)
                                regression_outputs, classification_outputs = model(X_batch)
                                classification_targets = (Y_batch <= DISTANCE_THRESHOLD).float()
                                classification_preds = (classification_outputs > threshold).float()
                                accuracy = (classification_preds == classification_targets).float().mean().item()
                                val_classification_accuracies.append(accuracy)
                        avg_val_classification_accuracy = np.mean(val_classification_accuracies)

                        print(
                            f"Hyperparams (lr={lr}, hidden={hidden_size}, batch={batch_size}, layers={num_layers}, thresh={threshold})")
                        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {avg_val_classification_accuracy:.4f}")

                        # Update best hyperparameters if the current configuration is better
                        if avg_val_classification_accuracy > best_overall_val_loss:
                            print("New best hyperparameters found!")
                            best_overall_val_loss = avg_val_classification_accuracy
                            best_hyperparams = {
                                "input_dim": input_dim,
                                "output_dim": output_dim,
                                "lr": lr,
                                "hidden_size": hidden_size,
                                "batch_size": batch_size,
                                "num_epochs": num_epochs,
                                "num_layers": num_layers
                            }
                            best_model_state = model_state
                            best_threshold = threshold

                            # Create test loader with best batch size for final evaluation
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if best_hyperparams is None:
        print("Error: No valid hyperparameters found during grid search.")
        return

    print("\nBest hyperparameters found:")
    print(best_hyperparams)
    print(f"Best classification threshold: {best_threshold}")
    print(f"Best validation classification accuracy: {best_overall_val_loss:.4f}")

    # Build the best model with all best hyperparameters
    best_model = MLPRegressor(input_dim,
                              best_hyperparams["hidden_size"],
                              output_dim,
                              num_layers=best_hyperparams["num_layers"]).to(device)
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
            classification_targets = (Y_batch <= DISTANCE_THRESHOLD).float()
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

    mae_array = []
    rmse_array = []
    mre_array = []
    # {"0_1": {"mae": [], "rmse": [], "mre": [], "corr": []}, "1_2": {"mae": [], "rmse": [], "mre": [], "corr": []}}
    # setup a collection like this for ranges 0-1, 1-2, 2-3, 3-4, and 4-5
    # for each range, calculate the metrics and append to the collection
    # then save each range to different csv files
    
    range_metrics = []
    
    for i in range(len(Y_true)):
        # add all non nan numbers to array
        mae_array.append(np.nanmean(np.abs(Y_true[i] - Y_pred[i])))
        rmse_array.append(np.sqrt(np.nanmean((Y_true[i] - Y_pred[i]) ** 2)))
        mre_array.append(np.nanmean(np.abs((Y_true[i] - Y_pred[i]) / (Y_true[i] + 1e-10))) if np.any(Y_true[i]) else 0)
        
        range_metrics.append(compute_error_metrics(Y_true[i], Y_pred[i]))

    mean_absolute_error = np.nanmean(mae_array)
    root_mean_square_error = np.nanmean(rmse_array)
    mean_relative_error = np.nanmean(mre_array)
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"Root Mean Square Error: {root_mean_square_error:.4f}")
    print(f"Mean Relative Error: {mean_relative_error:.4f}")
    
    range_metric_accum = {}
    for results in range_metrics:
        for range_bin, metrics in results.items():
            if range_bin not in range_metric_accum:
                range_metric_accum[range_bin] = {'mae': [], 'rmse': [], 'mre': []}
            for metric, value in metrics.items():
                range_metric_accum[range_bin][metric].append(value)
    
    range_metrics_average = {
        range_bin: {
            metric: float(np.mean(values))
            for metric, values in metrics.items()
        }
        for range_bin, metrics in range_metric_accum.items()
    }
    
    range_metrics_average["all"] = {
        'mae': mean_absolute_error,
        'rmse': root_mean_square_error,
        'mre': mean_relative_error
    }
    
    # currently broken - will fix later
    #save_feature_importance(chosen_dataset, best_model, X_val, Y_val, loss_fn, device, feature_names_full, num_epochs, num_layers)

    # Save the best model.
    models_folder = os.path.join("./Echolocation", "Models")
    os.makedirs(models_folder, exist_ok=True)
    model_file = os.path.join(models_folder, f"{chosen_dataset}_{num_epochs}_{num_layers}_model.pth")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "hyperparameters": best_hyperparams,
        "classification_threshold": best_threshold
    }, model_file)

    print("Saving comparison plots...")
    # Create folders
    cartesian_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, f"cartesian_plots_{num_epochs}_{num_layers}")
    scan_index_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, f"scan_index_plots_{num_epochs}_{num_layers}")
    os.makedirs(cartesian_folder, exist_ok=True)
    os.makedirs(scan_index_folder, exist_ok=True)


    # Create and start workers
    start_multiprocessing_plotting(
        Y_true, Y_pred, classifications, original_distances_test,
        num_epochs, num_layers, cartesian_folder, scan_index_folder,
        best_threshold, chosen_dataset
    )

    print(f"Best model saved to {model_file}")

    # Evaluate classification accuracy
    classification_accuracy = (classifications.round() == (Y_true <= DISTANCE_THRESHOLD)).mean()
    print(f"Classification Accuracy: {classification_accuracy:.4f}")
    print("\nBest hyperparameters found:")
    print(best_hyperparams)
    print(f"Best classification threshold: {best_threshold}")
    print(f"Best validation classification accuracy: {best_overall_val_loss:.4f}")
    print(f"\nMean Squared Error on test set: {avg_test_loss:.4f}")
    print(f"Best model saved to {model_file}")
    
    # Saving error metrics to seperate CSV for each range
    header = ['chirp', 'range', 'best_validation_loss', 'mean_absolute_error', 'root_mean_square_error', 'mean_relative_error']
    error_metrics_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", "error_metrics.csv")
    for range_bin in range_metrics_average:
        row = [chosen_dataset, range_bin, best_overall_val_loss, range_metrics_average[range_bin]['mae'], range_metrics_average[range_bin]['rmse'], range_metrics_average[range_bin]['mre']]
        
        try:
            with open(error_metrics_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(row)
        except FileExistsError:
            with open(error_metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        
    print("Error metrics saved")
    print("Model training and evaluation complete.")
