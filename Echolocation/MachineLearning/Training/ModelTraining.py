## ModelTraining.py
#!/usr/bin/env python3
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Echolocation.MachineLearning.Training.TrainingConfig import PATIENCE, DISTANCE_THRESHOLD
from Echolocation.MachineLearning.Training.ModelFunctions import MaskedMSELoss, AudioLidarDataset, MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------
# Training Functions
# ---------------------------
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

def train_regressor(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, scheduler=None, patience=PATIENCE):
    model.to(device)
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        preds_train_list = []
        targets_train_list = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            preds_train = model(X_batch)
            #print(f"len preds_train: {len(preds_train.cpu().detach().numpy()[0])}")
            preds_train_list.append(preds_train.cpu().detach().numpy())

            targets_train_list.append(Y_batch.cpu().detach().numpy())

            loss = loss_fn(preds_train, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_training_loss = np.mean(train_losses)
        #print(f"lengtsss preds_train len: {len(np.concatenate(preds_train_list)[0])}")
        preds_train_list = np.concatenate(preds_train_list)
        targets_train_list = np.concatenate(targets_train_list)

        # validation
        model.eval()
        preds_val_list, targets_val_list = [], []
        val_loss = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds_val = model(X_batch).cpu()
                preds_val_list.append(preds_val)
                targets_val_list.append(Y_batch)
                val_loss.append(loss_fn(preds_val, Y_batch).item())
        preds_val_list = np.concatenate(preds_val_list)
        targets_val_list = np.concatenate(targets_val_list)

        avg_val_loss = np.mean(val_loss)
        print(f"[Regressor] Epoch {epoch + 1}/{epochs}, Train Loss: {avg_training_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        combined_loss = avg_training_loss + avg_val_loss

        # Adjust the learning rate using the scheduler if provided
        if scheduler:
            scheduler.step(avg_val_loss)
            print(f"    Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"    No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model, avg_val_loss, preds_train_list, preds_val_list, targets_train_list, targets_val_list

def evaluate_regressor(model, dataloader, device):
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()
            preds_list.append(preds)
            targets_list.append(Y_batch)
    return torch.cat(preds_list), torch.cat(targets_list)


def train_classifier(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, scheduler, patience=PATIENCE):
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):  # run for specified epochs
        train_loss = []
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, Yb)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        avg_training_loss = np.mean(train_loss)

        # validation
        model.eval()
        val_loss = []
        val_preds, val_labels = [], []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.to(device)
                pred = model(Xb).cpu()
                val_preds.append(pred)
                val_labels.append(Yb)
                loss = loss_fn(pred, Yb)
                val_loss.append(loss.item())
        val_preds = np.concatenate(val_preds).flatten()
        val_labels = np.concatenate(val_labels).flatten()
        avg_val_loss = np.mean(val_loss)
        avg_loss = avg_training_loss + avg_val_loss

        print(
            f"[Classifier] Epoch {epoch + 1}/{epochs}, Train Loss: {avg_training_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        #print(f"val preds: {val_preds}")
        #print(f"val labels: {val_labels}")
        y_val_preds_thresholded = (val_preds > 0.5).astype(int)
        #print(f"y_val_preds_thresholded: {y_val_preds_thresholded}")
        # Compute classification metrics
        precision = precision_score(val_labels, y_val_preds_thresholded, zero_division=0)
        recall = recall_score(val_labels, y_val_preds_thresholded, zero_division=0)
        f1_classifier = f1_score(val_labels, y_val_preds_thresholded, zero_division=0)
        print(
            f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_classifier:.4f}"
        )
        # Adjust the learning rate using the scheduler if provided
        if scheduler:
            scheduler.step(avg_val_loss)
            print(f"    Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"    No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model, avg_val_loss

def evaluate_classifier(model, classifier_loader_val, device,predicted_val, y_val_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(predicted_val.to(device)).cpu().numpy()
    y_val_preds = outputs.flatten()
    y_val_labels_flat = y_val_labels.flatten().numpy()

    y_val_preds_thresholded = (y_val_preds > 0.5).astype(int)
    # Compute classification metrics
    precision = precision_score(y_val_labels_flat, y_val_preds_thresholded, zero_division=0)
    recall = recall_score(y_val_labels_flat, y_val_preds_thresholded, zero_division=0)
    f1_classifier = f1_score(y_val_labels_flat, y_val_preds_thresholded, zero_division=0)
    print(
        f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_classifier:.4f}\n"
    )

    return y_val_preds, y_val_labels_flat

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