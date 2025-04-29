#!/usr/bin/env python3
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Echolocation.MachineLearning.Training.TrainingConfig import PATIENCE, DISTANCE_THRESHOLD
from Echolocation.MachineLearning.Training.ModelFunctions import MaskedMSELoss, AudioLidarDataset, MLPRegressor

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, optimizer, regression_loss_fn, classification_loss_fn,
                device, num_epochs, patience=PATIENCE, scheduler=None,
                reg_weight=1.0, class_weight=1.0):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    best_combined_score = float("inf")
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

            loss = reg_weight * regression_loss + class_weight * classification_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --------------------
        # Validation Phase
        # --------------------
        model.eval()
        val_regression_losses = []
        val_classification_losses = []
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                regression_outputs, classification_outputs = model(X_batch)

                regression_loss = regression_loss_fn(regression_outputs, Y_batch)
                classification_targets = (Y_batch <= DISTANCE_THRESHOLD).float()
                classification_loss = classification_loss_fn(classification_outputs, classification_targets)

                val_regression_losses.append(regression_loss.item())
                val_classification_losses.append(classification_loss.item())

                all_targets.extend(classification_targets.cpu().numpy().flatten())
                all_preds.extend(classification_outputs.cpu().numpy().flatten())

        avg_reg_loss = np.mean(val_regression_losses)
        avg_class_loss = np.mean(val_classification_losses)

        # Apply sigmoid threshold for binary classification
        thresholded_preds = (np.array(all_preds) > 0.5).astype(int)
        all_targets = np.array(all_targets).astype(int)

        precision = precision_score(all_targets, thresholded_preds, zero_division=0)
        recall = recall_score(all_targets, thresholded_preds, zero_division=0)
        f1_classifier = f1_score(all_targets, thresholded_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            roc_auc = float('nan')

        combined_loss = reg_weight * avg_reg_loss + class_weight * (1 - f1_classifier)  # minimizing

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Regression Loss: {avg_reg_loss:.4f}, Val Classification Loss: {avg_class_loss:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_classifier:.4f}, ROC AUC: {roc_auc:.4f}")
        print(f"  Combined loss: {combined_loss:.4f}")

        if scheduler:
            scheduler.step(combined_loss)
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        if combined_loss < best_combined_score:
            best_combined_score = combined_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break

    return best_combined_score, best_model_state


def compute_permutation_importance(model, X_val, Y_val, loss_fn, device):
    model.eval()
    base_preds, _ = model(torch.tensor(X_val, dtype=torch.float32).to(device))
    base_loss = loss_fn(base_preds, torch.tensor(Y_val, dtype=torch.float32).to(device)).item()

    importances = []
    for i in range(X_val.shape[1]):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_preds, _ = model(torch.tensor(X_permuted, dtype=torch.float32).to(device))
        permuted_loss = loss_fn(permuted_preds, torch.tensor(Y_val, dtype=torch.float32).to(device)).item()
        importance = permuted_loss - base_loss
        importances.append(importance)

    return importances

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
