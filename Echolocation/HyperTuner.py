"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from MachineLearning.Training.mainTraining import split_data, prepare_dataset
from MachineLearning.Training.ModelFunctions import MaskedMSELoss

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants for batch size, number of classes, and dataset paths
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()  # Current working directory
EPOCHS = 100  # Number of training epochs
N_TRAIN_EXAMPLES = BATCHSIZE * 30  # Limit training examples for faster training
N_VALID_EXAMPLES = BATCHSIZE * 10  # Limit validation examples for faster evaluation


def define_model(trial,input_dim, output_dim):
    """
    Define the neural network model with hyperparameters optimized by Optuna.

    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter optimization.

    Returns:
        nn.Sequential: A PyTorch sequential model.
    """
    # Optimize the number of layers, hidden units, and dropout ratio
    n_layers = trial.suggest_int("n_layers", 2, 3)
    layers = []

    in_features = input_dim  # Input size for FashionMNIST (28x28 images)
    #print(f"Input features: {in_features}")
    for i in range(n_layers):
        # Suggest the number of units in the current layer
        out_features = trial.suggest_categorical("hidden_dim_{}".format(i), [16, 32, 64, 128, 256, 512, 1024])
        #print(f"Layer {i}: {in_features} -> {out_features}")
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())  # Activation function

        # Suggest the dropout rate for the current layer
        #p = trial.suggest_float("dropout_l{}".format(i), 0.0, 0.2)
        #p = trial.suggest_categorical("dropout_l{}".format(i), [0.0, 0.1, 0.2, 0.3])

        #layers.append(nn.Dropout(p))

        in_features = out_features  # Update input size for the next layer

    # Add the output layer
    layers.append(nn.Linear(in_features, output_dim))
    #layers.append(nn.LogSoftmax(dim=1))  # Log-Softmax for classification
    #print(f"Output features: {in_features} -> {output_dim}")

    return nn.Sequential(*layers)


def get_mnist(trial):
    """
    Load the FashionMNIST dataset.

    Returns:
        tuple: Training and validation data loaders.
    """
    # split data into training and validation sets
    csv_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures/echolocation-wide-long-all/features_all_normalized.csv")
    dataset_root_directory = os.path.join("./Echolocation/Data/dataset/echolocation-wide-long-all")
    distance = 2.0  # Distance threshold for filtering data
    data = prepare_dataset(csv_file, dataset_root_directory, distance)

    X, Y, sample_ids, original_distances, feature_names = data
    splits = split_data(X, Y, original_distances, sample_ids)
    train_dataset, val_dataset, test_dataset, *split_info = splits
    
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128])
    # Load training data
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, True
    )
    # Load validation data
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size, True
    )
    
    #print(f"\nX : {X}\n")
    #print(f"X shape: {X.shape}")
    #print(f"X shape 0: {X.shape[0]}")
    #print(f"X shape 0: {X.shape[1]}")
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    return train_loader, valid_loader, input_dim, output_dim


def objective(trial):
    """
    Objective function for Optuna to optimize.

    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter optimization.

    Returns:
        float: Validation accuracy of the model.
    """
    
    # Get the FashionMNIST dataset
    train_loader, valid_loader, input_dim, output_dim = get_mnist(trial)
    # Generate the model with trial-specific hyperparameters
    model = define_model(trial,input_dim, output_dim).to(DEVICE)

    # Suggest the optimizer and learning rate
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = 0.01
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-5, 0.0]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, 3)
    loss_fn = MaskedMSELoss()  # Loss function

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            # Limit training data for faster epochs
            #if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
            #    break

            # Flatten the input data and move to the appropriate device
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            # Perform forward and backward passes
            optimizer.zero_grad()
            output = model(X_batch)
            
            loss = loss_fn(output, Y_batch)  # Negative log-likelihood loss
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        preds_val_list, targets_val_list = [], []
        val_loss = []  # List to store validation losses
        with torch.no_grad():
            for batch_idx, (X_batch, Y_batch) in enumerate(valid_loader):
                # Limit validation data
                #if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                #    break
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                
                output = model(X_batch)
                # Get the index of the max log-probability
                #print(f"Output shape: {output.shape}")
                #print(f"Y_batch shape: {Y_batch.shape}")
                val_loss.append(loss_fn(output, Y_batch).item())  # Compute and store loss

        # Calculate validation accuracy
        avg_val_loss = round(np.mean(val_loss), 4)
        #print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {avg_val_loss:.4f}")

        # Report intermediate results to Optuna
        trial.report(avg_val_loss, epoch)
        
        scheduler.step(avg_val_loss)
        #print(f"    Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Prune the trial if it is not promising
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss


if __name__ == "__main__":
    # Create an Optuna study to maximize validation accuracy
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=2400, show_progress_bar=True)  # Run optimization for 100 trials or 600 seconds

    # Get statistics about the study
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Print study statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)  # Best validation accuracy

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))  # Best hyperparameters