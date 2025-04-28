#!/usr/bin/env python3
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from MachineLearning.Training.TrainingConfig import PATIENCE, DISTANCE_THRESHOLD
from MachineLearning.Training.ModelFunctions import MaskedMSELoss, AudioLidarDataset, MLPRegressor

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, optimizer, regression_loss_fn, classification_loss_fn, device, num_epochs, patience=PATIENCE, scheduler=None):
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

def compute_error_metrics(Y_true, Y_pred):
    range_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    results = {f"{low}_{high}": {'y_true': [], 'y_pred': []} for low, high in range_bins}

    for y_true, y_pred in zip(Y_true, Y_pred):
        for low, high in range_bins:
            if low <= y_true < high:
                key = f"{low}_{high}"
                results[key]['y_true'].append(y_true)
                results[key]['y_pred'].append(y_pred)
                break

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
