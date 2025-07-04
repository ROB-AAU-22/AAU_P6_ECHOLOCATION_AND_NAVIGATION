## ModelTraining.py
#!/usr/bin/env python3
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from MachineLearning.Training.TrainingConfig import PATIENCE, CLASSIFICATION_THRESHOLD
from MachineLearning.Training.ModelFunctions import MaskedMSELoss, AudioLidarDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

# ---------------------------
# Training Functions
# ---------------------------
def train_one_epoch_regressor(model, train_loader, optimizer, loss_fn, device):
    """
    Trains a regression model for one epoch.
    Args:
        model (torch.nn.Module): The regression model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        loss_fn (callable): Loss function to compute the training loss.
        device (torch.device): Device on which the computation will be performed (e.g., 'cpu' or 'cuda').
    Returns:
        tuple: A tuple containing:
            - avg_training_loss (float): The average training loss for the epoch.
            - preds_train_list (numpy.ndarray): Concatenated predictions for the training set.
            - targets_train_list (numpy.ndarray): Concatenated true targets for the training set.
    """
    model.train()  # Set the model to training mode
    train_losses = []  # List to store training losses for the epoch
    preds_train_list = []  # List to store predictions for the training set
    targets_train_list = []  # List to store true targets for the training set

    # Iterate over the training data
    for X_batch, Y_batch in train_loader:
        # Move data to the specified device
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()  # Reset gradients
        preds_train = model(X_batch)  # Forward pass
        preds_train_list.append(preds_train.cpu().detach().numpy())  # Store predictions
        targets_train_list.append(Y_batch.cpu().detach().numpy())  # Store true targets

        loss = loss_fn(preds_train, Y_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        train_losses.append(loss.item())  # Store the loss

    # Compute average training loss for the epoch
    avg_training_loss = np.mean(train_losses)
    preds_train_list = np.concatenate(preds_train_list)  # Concatenate all training predictions
    targets_train_list = np.concatenate(targets_train_list)  # Concatenate all training targets

    return avg_training_loss, preds_train_list, targets_train_list


def validate_one_epoch_regressor(model, val_loader, loss_fn, device):
    """
    Validates a regression model for one epoch using the provided validation data loader.
    Args:
        model (torch.nn.Module): The regression model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        loss_fn (callable): Loss function to compute the validation loss.
        device (torch.device): Device to run the validation on (e.g., 'cpu' or 'cuda').
    Returns:
        tuple: A tuple containing:
            - avg_val_loss (float): The average validation loss for the epoch.
            - preds_val_list (numpy.ndarray): Concatenated predictions for the entire validation set.
            - targets_val_list (numpy.ndarray): Concatenated true targets for the entire validation set.
    """
    model.eval()  # Set the model to evaluation mode
    preds_val_list, targets_val_list = [], []  # Lists to store validation predictions and targets
    val_loss = []  # List to store validation losses

    with torch.no_grad():  # Disable gradient computation for validation
        for X_batch, Y_batch in val_loader:
            # Move data to the specified device
            X_batch = X_batch.to(device)
            preds_val = model(X_batch).cpu()  # Forward pass
            preds_val_list.append(preds_val)  # Store predictions
            targets_val_list.append(Y_batch)  # Store true targets
            val_loss.append(loss_fn(preds_val, Y_batch).item())  # Compute and store loss

    # Concatenate all validation predictions and targets
    preds_val_list = np.concatenate(preds_val_list)
    targets_val_list = np.concatenate(targets_val_list)

    # Compute average validation loss for the epoch
    avg_val_loss = np.mean(val_loss)

    return avg_val_loss, preds_val_list, targets_val_list

def train_regressor(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, scheduler=None, patience=PATIENCE):
    """
    Trains a regression model using the provided training and validation data loaders.
    Args:
        model (torch.nn.Module): The regression model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (callable): Loss function to compute the training and validation loss.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        epochs (int): Number of epochs to train the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
        patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to PATIENCE.
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model.
            - avg_val_loss (float): The average validation loss of the last epoch.
            - preds_train_list (list): List of predictions from the training dataset.
            - preds_val_list (list): List of predictions from the validation dataset.
            - targets_train_list (list): List of ground truth targets from the training dataset.
            - targets_val_list (list): List of ground truth targets from the validation dataset.
    """
    # Move the model to the specified device (GPU or CPU)
    model.to(device)
    best_loss = float("inf")  # Initialize the best loss to infinity
    patience_counter = 0  # Counter for early stopping

    for epoch in range(epochs):
        # Training phase
        avg_training_loss, preds_train_list, targets_train_list = train_one_epoch_regressor(
            model, train_loader, optimizer, loss_fn, device
        )

        # Validation phase
        avg_val_loss, preds_val_list, targets_val_list = validate_one_epoch_regressor(
            model, val_loader, loss_fn, device
        )

        print(f"[Regressor] Epoch {epoch + 1}/{epochs}, Train Loss: {avg_training_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        #print(f"Predictions: {np.concatenate(preds_val_list).flatten()}")
        # calculate mean absolute error
        #mae = np.mean(np.abs(np.concatenate(preds_val_list).flatten() - np.concatenate(targets_val_list).flatten()))
        # calculate mean squared error
        #mse = np.mean((np.concatenate(preds_val_list).flatten() - np.concatenate(targets_val_list).flatten()) ** 2)
        # calculate root mean squared error
        #rmse = np.sqrt(mse)
        #print(f"    Mean Absolute Error: {mae:.4f}, Mean Squared Error: {mse:.4f}, Root Mean Squared Error: {rmse:.4f}")


        # Adjust the learning rate using the scheduler if provided
        if scheduler:
            scheduler.step(avg_val_loss)
            print(f"    Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping logic
        if round(avg_val_loss,4) < round(best_loss,4):
            best_loss = avg_val_loss  # Update the best loss
            patience_counter = 0  # Reset the patience counter
        else:
            patience_counter += 1  # Increment the patience counter
            print(f"    No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")  # Stop training if patience is exceeded
                break

    return model, avg_val_loss, preds_train_list, preds_val_list, targets_train_list, targets_val_list






def train_one_epoch_classifier(model, train_loader, optimizer, loss_fn, device):
    """
    Trains a classification model for one epoch.
    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        loss_fn (torch.nn.Module): Loss function to compute the training loss.
        device (torch.device): Device on which to perform computations (e.g., 'cpu' or 'cuda').
    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
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
    return avg_training_loss


def validate_one_epoch_classifier(model, val_loader, loss_fn, device):
    """
    Validates a classifier model for one epoch using the provided validation data loader.
    Args:
        model (torch.nn.Module): The classifier model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        loss_fn (callable): Loss function to compute the validation loss.
        device (torch.device): The device (CPU or GPU) to perform computations on.
    Returns:
        tuple: A tuple containing:
            - avg_val_loss (float): The average validation loss over the epoch.
            - val_preds (numpy.ndarray): Flattened array of predicted values for the validation set.
            - val_labels (numpy.ndarray): Flattened array of true labels for the validation set.
    """
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
    return avg_val_loss, val_preds, val_labels


def train_classifier(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, scheduler, patience=PATIENCE):
    """
    Trains a classification model using the provided training and validation data loaders.
    Args:
        model (torch.nn.Module): The classification model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (callable): Loss function to compute the training and validation loss.
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
        epochs (int): Number of epochs to train the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): Learning rate scheduler to adjust the learning rate during training.
        patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to PATIENCE.
    Returns:
        torch.nn.Module: The trained model.
        float: The final validation loss after training.
    Notes:
        - The function performs early stopping if the validation loss does not improve for a specified number of epochs (patience).
        - During training, it prints the training loss, validation loss, classification accuracy, and other metrics (precision, recall, F1-score, accuracy).
        - If a scheduler is provided, it adjusts the learning rate based on the validation loss.
    """
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):  # run for specified epochs
        # Training phase
        avg_training_loss = train_one_epoch_classifier(model, train_loader, optimizer, loss_fn, device)

        # Validation phase
        avg_val_loss, val_preds, val_labels = validate_one_epoch_classifier(model, val_loader, loss_fn, device)
        y_val_preds_thresholded = (val_preds > CLASSIFICATION_THRESHOLD).astype(int)

        classification_accuracy = ((y_val_preds_thresholded) == (val_labels)).mean()

        print(
            f"[Classifier] Epoch {epoch + 1}/{epochs}, Train Loss: {avg_training_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Classification Accuracy: {classification_accuracy:.4f}")

        # Compute classification metrics
        precision = precision_score(val_labels, y_val_preds_thresholded, zero_division=0)
        recall = recall_score(val_labels, y_val_preds_thresholded, zero_division=0)
        f1_classifier = f1_score(val_labels, y_val_preds_thresholded, zero_division=0)
        accuracy = accuracy_score(val_labels, y_val_preds_thresholded)
        print(
            f"      Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_classifier:.4f}, Accuracy: {accuracy:.4f}"
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


def evaluate_regressor(model, dataloader, device):
    """
    Evaluates a regression model on a given dataset.

    Args:
        model (torch.nn.Module): The regression model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset for evaluation.
        device (torch.device): The device (CPU or GPU) on which the model and data should be processed.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Concatenated predictions from the model for the entire dataset.
            - torch.Tensor: Concatenated ground truth targets from the dataset.
    """
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()
            preds_list.append(preds)
            targets_list.append(Y_batch)
    return torch.cat(preds_list), torch.cat(targets_list)

def evaluate_classifier(model, device, predicted_val, y_val_labels, batch_size):
    """
    Evaluates the performance of a classifier model on validation data.
    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        predicted_val (torch.Tensor): The predicted values from the model.
        y_val_labels (torch.Tensor): The ground truth labels for the validation data.
        batch_size (int): The batch size to use for the DataLoader.
    Returns:
        tuple: A tuple containing:
            - y_val_preds_thresholded (numpy.ndarray): Thresholded predictions for the validation data.
            - y_val_labels_thresholded (numpy.ndarray): Thresholded ground truth labels for the validation data.
            - concatenated_preds (numpy.ndarray): Concatenated predictions from all batches.
            - concatenated_targets (torch.Tensor): Concatenated ground truth labels from all batches.
    Prints:
        Precision, Recall, and F1 score of the classifier on the validation data.
    """
    model.eval()
    #print(f"predicted_val: {predicted_val}")
    #print(f"y_val_labels: {y_val_labels}")
    data = [predicted_val, y_val_labels]
    dataloader = DataLoader(data, batch_size, shuffle=False)
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for Xb, Yb in dataloader:
            Xb = Xb.to(device)
            outputs = model(Xb).cpu()
            preds_list.append(outputs)
            targets_list.append(Yb)
    y_val_preds = outputs.numpy()
    #y_val_labels_flat = ~np.isnan(y_val_labels.cpu().numpy())
    #y_val_labels_thresholded = (y_val_labels_flat).astype(int)
    y_val_labels_thresholded = (y_val_labels.cpu().numpy() < 2.0).astype(int)

    y_val_preds_thresholded = (y_val_preds>CLASSIFICATION_THRESHOLD).astype(int)
    
    #print(f"y_val_preds_thresholded: {y_val_preds_thresholded}")
    #print(f"y_val_labels_thresholded: {y_val_labels_thresholded}")
    # concenate all predictions and targets
    y_val_preds_thresholded = np.concatenate(y_val_preds_thresholded)
    y_val_labels_thresholded = np.concatenate(y_val_labels_thresholded)
    #print(f"preds_list: {y_val_preds_thresholded}")
    #print(f"targets_list: {y_val_labels_thresholded}")
    # Compute classification metrics
    precision = precision_score(y_val_labels_thresholded, y_val_preds_thresholded, zero_division=0)
    recall = recall_score(y_val_labels_thresholded, y_val_preds_thresholded, zero_division=0)
    f1_classifier = f1_score(y_val_labels_thresholded, y_val_preds_thresholded, zero_division=0)
    print(
        f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_classifier:.4f}\n"
    )

    return y_val_preds_thresholded, y_val_labels_thresholded, torch.cat(preds_list).numpy(), torch.cat(targets_list), y_val_preds

def compute_error_metrics(Y_true, Y_pred, distance):
    """
    Compute error metrics (MAE, RMSE, MRE) between true and predicted values within a specified distance range.
    Parameters:
        Y_true (array-like): Ground truth (actual) values.
        Y_pred (array-like): Predicted values.
        distance (float): Upper bound for the range bin (lower bound is 0).
    Returns:
        dict: A dictionary where each key is a string representing the range bin (e.g., "0_10"), and each value is another dictionary containing:
            - 'mae': Mean Absolute Error for the bin.
            - 'rmse': Root Mean Squared Error for the bin.
            - 'mre': Mean Relative Error for the bin.
    Notes:
        - The function currently supports only a single range bin from 0 to `distance`.
        - If there are no samples in the bin, that bin is omitted from the results.
        - A small epsilon (1e-10) is added to the denominator in MRE calculation to avoid division by zero.
    """
    
    range_bins = [(0, distance)]
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
    """
    Computes permutation feature importance for a given model and validation dataset.
    Permutation importance measures the increase in the loss function when the values of each feature are randomly shuffled,
    which breaks the relationship between the feature and the target. A higher increase in loss indicates a more important feature.
    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        X_val (np.ndarray): Validation input features of shape (n_samples, n_features).
        Y_val (np.ndarray): Validation target values.
        loss_fn (callable): Loss function used to evaluate model performance.
        device (torch.device): Device on which computations are performed (e.g., 'cpu' or 'cuda').
    Returns:
        list: A list of importance scores, one for each feature, representing the increase in loss when that feature is permuted.
    """
    
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