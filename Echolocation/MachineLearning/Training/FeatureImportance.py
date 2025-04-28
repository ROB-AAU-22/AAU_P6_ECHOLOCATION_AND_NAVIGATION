#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        preds, _ = self.base_model(x)  # Only return predictions
        return preds

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

    wrapped_model = WrappedModel(model)
    ig = IntegratedGradients(wrapped_model)
    attributions = ig.attribute(
        torch.tensor(X_val, dtype=torch.float32).to(device),
        target=0
    )

    return importances, attributions

def save_feature_importance(chosen_dataset, best_model, X_val, Y_val, loss_fn, device, feature_names_full, num_epochs, num_layers):
    # compute feature importance
    importances, attributions = compute_permutation_importance(best_model, X_val, Y_val, loss_fn, device)

    ig_mean = attributions.detach().cpu().numpy()
    ig_mean = np.mean(np.abs(ig_mean), axis=0)

    # save feature importance to csv
    importance_df = pd.DataFrame({
        "Feature": feature_names_full,
        "PermutationImportance": importances,
        "IntegratedGradientsImportance": ig_mean
    })
    importance_df.sort_values(by="PermutationImportance", ascending=False, inplace=True)

    base_file_dir = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset)
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