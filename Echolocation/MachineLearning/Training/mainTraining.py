#!/usr/bin/env python3
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from TrainingConfig import LEARNING_RATES, HIDDEN_SIZES, BATCH_SIZES, NUM_EPOCHS, NUM_LAYERS_LIST, CLASSIFICATION_THRESHOLDS, DISTANCE_THRESHOLD
from data_preparation import build_dataset_from_csv
from model import MaskedMSELoss, AudioLidarDataset, MLPRegressor
from training import train_model, compute_error_metrics
from plotting import start_multiprocessing_plotting
def model_training(dataset_root_directory, chosen_dataset):
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

    X_train_val, X_test, Y_train_val, Y_test, original_distances_train_val, original_distances_test = train_test_split(X, Y, original_distances, test_size=0.15, random_state=42)
    X_train, X_val, Y_train, Y_val, original_distances_train, original_distances_val = train_test_split(X_train_val, Y_train_val, original_distances_train_val, test_size=0.1765, random_state=42)
    print("Training samples: ", X_train.shape[0], "Validation samples: ", X_val.shape[0], "Test samples: ", X_test.shape[0])

    train_dataset = AudioLidarDataset(X_train, Y_train)
    val_dataset = AudioLidarDataset(X_val, Y_val)
    test_dataset = AudioLidarDataset(X_test, Y_test)

    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_overall_val_loss = 0
    best_hyperparams = None
    best_model_state = None
    best_threshold = None

    regression_loss_fn = MaskedMSELoss()
    classification_loss_fn = nn.BCELoss()

    for lr in LEARNING_RATES:
        for hidden_size in HIDDEN_SIZES:
            for batch_size in BATCH_SIZES:
                for num_layers in NUM_LAYERS_LIST:
                    for threshold in CLASSIFICATION_THRESHOLDS:
                        print(f"\nTraining with: lr={lr}, hidden_size={hidden_size}, batch_size={batch_size}, layers={num_layers}, threshold={threshold}")

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                        model = MLPRegressor(input_dim, hidden_size, output_dim, num_layers=num_layers).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

                        val_loss, model_state = train_model(
                            model,
                            train_loader,
                            val_loader,
                            optimizer,
                            regression_loss_fn,
                            classification_loss_fn,
                            device,
                            NUM_EPOCHS,
                            scheduler=scheduler
                        )

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

                        print(f"Hyperparams (lr={lr}, hidden={hidden_size}, batch={batch_size}, layers={num_layers}, thresh={threshold})")
                        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {avg_val_classification_accuracy:.4f}")

                        if avg_val_classification_accuracy > best_overall_val_loss:
                            print("New best hyperparameters found!")
                            best_overall_val_loss = avg_val_classification_accuracy
                            best_hyperparams = {
                                "input_dim": input_dim,
                                "output_dim": output_dim,
                                "lr": lr,
                                "hidden_size": hidden_size,
                                "batch_size": batch_size,
                                "num_epochs": NUM_EPOCHS,
                                "num_layers": num_layers
                            }
                            best_model_state = model_state
                            best_threshold = threshold

                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if best_hyperparams is None:
        print("Error: No valid hyperparameters found during grid search.")
        return

    print("\nBest hyperparameters found:")
    print(best_hyperparams)
    print(f"Best classification threshold: {best_threshold}")
    print(f"Best validation classification accuracy: {best_overall_val_loss:.4f}")

    best_model = MLPRegressor(input_dim, best_hyperparams["hidden_size"], output_dim, num_layers=best_hyperparams["num_layers"]).to(device)
    best_model.load_state_dict(best_model_state)

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

    Y_pred = np.concatenate(all_preds, axis=0)
    Y_true = np.concatenate(all_targets, axis=0)
    classifications = np.concatenate(all_classifications, axis=0)

    mae_array = []
    rmse_array = []
    mre_array = []
    range_metrics = []

    for i in range(len(Y_true)):
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

    models_folder = os.path.join("./Echolocation", "Models")
    os.makedirs(models_folder, exist_ok=True)
    model_file = os.path.join(models_folder, f"{chosen_dataset}_{NUM_EPOCHS}_{best_hyperparams['num_layers']}_model.pth")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "hyperparameters": best_hyperparams,
        "classification_threshold": best_threshold
    }, model_file)

    print("Saving comparison plots...")
    cartesian_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, f"cartesian_plots_{NUM_EPOCHS}_{best_hyperparams['num_layers']}")
    scan_index_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, f"scan_index_plots_{NUM_EPOCHS}_{best_hyperparams['num_layers']}")
    os.makedirs(cartesian_folder, exist_ok=True)
    os.makedirs(scan_index_folder, exist_ok=True)

    start_multiprocessing_plotting(
        Y_true, Y_pred, classifications, original_distances_test,
        NUM_EPOCHS, best_hyperparams['num_layers'], cartesian_folder, scan_index_folder,
        best_threshold, chosen_dataset
    )

    print(f"Best model saved to {model_file}")

    classification_accuracy = (classifications.round() == (Y_true <= DISTANCE_THRESHOLD)).mean()
    print(f"Classification Accuracy: {classification_accuracy:.4f}")
    print("\nBest hyperparameters found:")
    print(best_hyperparams)
    print(f"Best classification threshold: {best_threshold}")
    print(f"Best validation classification accuracy: {best_overall_val_loss:.4f}")
    print(f"\nMean Squared Error on test set: {avg_test_loss:.4f}")
    print(f"Best model saved to {model_file}")

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
