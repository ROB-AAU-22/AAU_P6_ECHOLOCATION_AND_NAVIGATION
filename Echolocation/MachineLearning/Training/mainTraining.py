import os
import time
import csv
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from MachineLearning.Training.TrainingConfig import *
from MachineLearning.Training.DataHandler import build_dataset_from_csv, ClassifierDataset
from MachineLearning.Training.ModelFunctions import MaskedMSELoss, AudioLidarDataset, Regressor, Classifier
from MachineLearning.Training.ModelTraining import compute_error_metrics, train_regressor, evaluate_regressor, train_classifier, evaluate_classifier
from MachineLearning.Training.Plotting import start_multiprocessing_plotting
from MachineLearning.Training.EvaluationPlots import plot_precision_recall_curve, plot_confusion_matrix_all, plot_roc_curve

def prepare_dataset(csv_file, dataset_root_directory, distance):
    """
    Prepares the dataset for training by loading data from a CSV file and processing it.

    Args:
        csv_file (str): Path to the CSV file containing dataset metadata.
        dataset_root_directory (str): Root directory where dataset files are stored.
        distance (float): The distance parameter used to filter or process the dataset.

    Returns:
        tuple or None: A tuple containing the following elements if the dataset is successfully prepared:
            - X (numpy.ndarray): Feature matrix with shape (n_samples, n_features).
            - Y (numpy.ndarray): Target matrix with shape (n_samples, lidar_scan_length).
            - sample_ids (list): List of sample identifiers.
            - original_distances (list): List of original distances corresponding to the samples.
            - feature_names_full (list): List of feature names.
          Returns None if the dataset contains no samples.

    Prints:
        Information about the dataset, including the number of samples, features, LiDAR scan length,
        and statistics (min and max values) for both X and Y.
    """
    X, Y, sample_ids, feature_names_full, original_distances = build_dataset_from_csv(csv_file, dataset_root_directory, distance)
    if X.shape[0] == 0:
        return None
    #print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, LiDAR scan length: {Y.shape[1]}")
    #print("X stats: min", np.min(X), "max", np.max(X))
    #print("Y stats: min", np.min(Y), "max", np.max(Y))
    return X, Y, sample_ids, original_distances

def split_data(X, Y, original_distances, sample_ids):
    """
    Splits the input data into training, validation, and test sets.
    This function takes the input features, labels, original distances, and sample IDs,
    and splits them into training, validation, and test datasets. The test set is split
    first, followed by a split of the remaining data into training and validation sets.
    Args:
        X (numpy.ndarray): The input features.
        Y (numpy.ndarray): The target labels.
        original_distances (numpy.ndarray): The original distances corresponding to the samples.
        sample_ids (numpy.ndarray): The unique identifiers for the samples.
    Returns:
        tuple: A tuple containing:
            - AudioLidarDataset: Training dataset.
            - AudioLidarDataset: Validation dataset.
            - AudioLidarDataset: Test dataset.
            - numpy.ndarray: Sample IDs for the test set.
            - numpy.ndarray: Original distances for the test set.
            - numpy.ndarray: Target labels for the test set.
    Notes:
        - The test set size is 15% of the total data.
        - The validation set size is 15% of the remaining data after splitting the test set.
        - The random_state is fixed at 42 for reproducibility.
    """
    X_train_val, X_test, Y_train_val, Y_test, od_train_val, od_test, sid_train_val, sid_test = train_test_split(
        X, Y, original_distances, sample_ids, test_size=0.15, random_state=42)

    X_train, X_val, Y_train, Y_val, od_train, od_val, sid_train, sid_val = train_test_split(
        X_train_val, Y_train_val, od_train_val, sid_train_val, test_size=0.15, random_state=42)

    #print("Training samples:", X_train.shape[0], "Validation:", X_val.shape[0], "Test:", X_test.shape[0])
    return (AudioLidarDataset(X_train, Y_train),
            AudioLidarDataset(X_val, Y_val),
            AudioLidarDataset(X_test, Y_test),
            sid_test, od_test, Y_test, od_train, od_val)

def run_regressor_grid_search(train_dataset, val_dataset, test_dataset, input_dim, output_dim, device, dataset_iter, distance_folder, chosen_dataset):
    """
    Perform a grid search to find the best hyperparameters for a regressor model.
    This function trains a regressor model using various combinations of hyperparameters
    and evaluates their performance on a validation dataset. The best-performing model
    and its corresponding hyperparameters are returned.
    Args:
        train_dataset (torch.utils.data.Dataset): The dataset used for training the model.
        val_dataset (torch.utils.data.Dataset): The dataset used for validating the model.
        test_dataset (torch.utils.data.Dataset): The dataset used for testing the model.
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output predictions.
        device (torch.device): The device (CPU or GPU) to run the computations on.
        dataset_iter (str): Identifier for the current dataset iteration.
        distance_folder (str): Folder name for storing results based on distance.
        chosen_dataset (str): Name of the dataset being used.
    Returns:
        dict: A dictionary containing the following keys:
            - model (torch.nn.Module): The best-trained regressor model.
            - hyperparams (dict): The best hyperparameters for the model.
            - val_loss (float): The validation loss of the best model.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            - preds_train (torch.Tensor): Predictions on the training dataset.
            - preds_val (torch.Tensor): Predictions on the validation dataset.
            - targets_train (torch.Tensor): Ground truth targets for the training dataset.
            - targets_val (torch.Tensor): Ground truth targets for the validation dataset.
    Notes:
        - The function logs the hyperparameters and validation loss of the best model
          to a file named "regressor_results.txt" in the specified directory.
        - The function uses a predefined set of hyperparameter values for grid search,
          including learning rates, hidden dimensions, batch sizes, number of layers,
          weight decays, and layer types.
        - The training process uses the MaskedMSELoss as the loss function and the
          Adam optimizer with a learning rate scheduler.
    """
    best_loss = float("inf")
    best_params, best_state = None, None
    best_preds_train, best_preds_val = None, None
    best_targets_train, best_targets_val = None, None
    test_loader = None
    loss_fn = MaskedMSELoss()

    for lr in REGRESSOR_LEARNING_RATES:
        for hd in REGRESSOR_HIDDEN_DIMS:
            nl = len(hd)
            for bs in REGRESSOR_BATCH_SIZES:
                for wd in REGRESSOR_WEIGHT_DECAYS:
                    for lt in REGRESSOR_LAYER_TYPE:
                        for ot in REGRESSOR_OPTIMIZER:
                            print(f"\nTraining Regressor: hidden={hd}, batch={bs}, layers={nl}, type={lt}, decay={wd}, optimizer={ot}")
                            model = Regressor(input_dim, hd, output_dim, num_layers=nl, layer_type=lt).to(device)

                            opt = getattr(optim, ot)(model.parameters(), lr=lr, weight_decay=wd, fused=True)
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)

                            model, val_loss, pt_train, pt_val, tt_train, tt_val = train_regressor(
                                model, DataLoader(train_dataset, bs, True), #DataLoader(train_data, bs, True)
                                DataLoader(val_dataset, bs, True),  #DataLoader(val_dataset, bs, False)
                                opt, loss_fn, device, NUM_EPOCHS, scheduler)
                            
                            params = {"input_dim": input_dim, "output_dim": output_dim, "lr": lr,
                                                "hidden_dim": hd, "batch_size": bs, "num_epochs": NUM_EPOCHS,
                                                "num_layers": nl, "layer_type": lt, "weight_decay": wd, "optimizer": ot}
                            
                            

                            print(f"Val Loss: {val_loss:.4f}")
                            if val_loss < best_loss:
                                print(f"New best loss: {val_loss:.4f}")
                                best_loss = val_loss
                                best_state = model.state_dict()
                                best_params = params
                                best_preds_train, best_preds_val = pt_train, pt_val
                                best_targets_train, best_targets_val = tt_train, tt_val
                                test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

                            path = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, distance_folder, "regressor_results.txt")
                            log = f"Regressor Hyperparameters: {params}\n  Best Regressor Loss: {val_loss:.4f}\n\n"
                            with open(path, 'a') as f:
                                f.write(log)

    if best_params is None:
        return None

    model = Regressor(**{k: best_params[k] for k in ("input_dim", "hidden_dim", "output_dim")},
                      num_layers=best_params["num_layers"], layer_type=best_params["layer_type"]).to(device)
    model.load_state_dict(best_state)

    return dict(model=model, hyperparams=best_params, val_loss=best_loss, test_loader=test_loader,
                preds_train=best_preds_train, preds_val=best_preds_val,
                targets_train=best_targets_train, targets_val=best_targets_val)

def run_classifier_grid_search(reg_results, output_dim, device, dataset_iter, distance_folder, chosen_dataset):
    """
    Perform a grid search to find the best hyperparameters for a classifier model.
    This function trains a classifier model using various combinations of hyperparameters
    and evaluates their performance on a validation dataset. The best-performing model
    and its hyperparameters are saved and returned.
    Args:
        reg_results (dict): A dictionary containing training and validation predictions 
            and targets. Keys include:
            - "preds_train": Predictions for the training set.
            - "targets_train": Targets for the training set.
            - "preds_val": Predictions for the validation set.
            - "targets_val": Targets for the validation set.
        output_dim (int): The dimensionality of the output layer of the classifier.
        device (torch.device): The device (CPU or GPU) to use for training the model.
        dataset_iter (str): Identifier for the current dataset iteration.
        distance_folder (str): Folder name corresponding to the distance-based dataset.
        chosen_dataset (str): Name of the dataset being used.
    Returns:
        dict: A dictionary containing:
            - "model": The best-trained classifier model.
            - "hyperparams": A dictionary of the best hyperparameters.
            - "val_loss": The validation loss of the best model.
        None: If no valid hyperparameters were found.
    Notes:
        - The function logs the hyperparameters and validation loss of the best model
          to a file named "classifier_results.txt" in the specified directory.
        - The hyperparameter search space includes learning rates, hidden dimensions,
          batch sizes, number of layers, layer types, and weight decays.
    """
    best_loss = float("inf")
    best_params, best_state = None, None
    loss_fn = nn.BCELoss()

    train_data = ClassifierDataset(reg_results["preds_train"], reg_results["targets_train"])
    val_data = ClassifierDataset(reg_results["preds_val"], reg_results["targets_val"])

    for lr in CLASSIFIER_LEARNING_RATES:
        for hd in CLASSIFIER_HIDDEN_DIMS:
            for bs in CLASSIFIER_BATCH_SIZES:
                for nl in CLASSIFIER_NUM_LAYERS_LIST:
                    for wd in CLASSIFIER_WEIGHT_DECAYS:
                        for lt in CLASSIFIER_LAYER_TYPE:
                            for ot in CLASSIFIER_OPTIMIZER:
                                print(f"\nTraining Classifier: hidden={hd}, batch={bs}, layers={nl}, type={lt}, decay={wd}, optimizer={ot}")
                                model = Classifier(input_dim=output_dim, hidden_dim=hd, output_dim=output_dim,
                                                num_layers=nl, layer_type=lt).to(device)
                                if ot == "SGD":
                                    lr = 0.1
                                opt = getattr(optim, ot)(model.parameters(), lr=lr, weight_decay=wd, fused=True)
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)

                                model, val_loss = train_classifier(
                                    model, DataLoader(train_data, bs, True), DataLoader(val_data, bs, True), #DataLoader(val_data, bs, False)
                                    opt, loss_fn, device, NUM_EPOCHS, scheduler)

                                params = {"input_dim": output_dim, "output_dim": output_dim, "lr": lr,
                                                "hidden_dim": hd, "batch_size": bs, "num_epochs": NUM_EPOCHS,
                                                "num_layers": nl, "layer_type": lt, "weight_decay": wd, "optimizer": ot}
                                
                                print(f"Val loss: {val_loss:.4f}")
                                if val_loss < best_loss:
                                    print(f"New best loss: {val_loss:.4f}")
                                    best_loss = val_loss
                                    best_params = params
                                    best_state = model.state_dict()

                                path = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, distance_folder, "classifier_results.txt")
                                log = f"Classifier Hyperparameters: {params}\n  Best classifier Loss: {val_loss:.4f}\n\n"
                                with open(path, 'a') as f:
                                    f.write(log)

    if best_params is None:
        return None

    model = Classifier(**{k: best_params[k] for k in ("input_dim", "hidden_dim", "output_dim")},
                       num_layers=best_params["num_layers"], layer_type=best_params["layer_type"]).to(device)
    model.load_state_dict(best_state)

    return dict(model=model, hyperparams=best_params, val_loss=best_loss)

def evaluate_and_save_results(reg_results, cls_results, test_dataset, split_info, distance, distance_folder, dataset_iter, chosen_dataset, device):
    """
    Evaluates regression and classification models, generates evaluation metrics, 
    saves plots, and stores model states and metrics to disk.
    Args:
        reg_results (dict): Dictionary containing the regression model and its test data loader.
            - 'model': Trained regression model.
            - 'test_loader': DataLoader for the test dataset.
            - 'hyperparams': Hyperparameters used for training the regression model.
        cls_results (dict): Dictionary containing the classification model and its hyperparameters.
            - 'model': Trained classification model.
            - 'hyperparams': Hyperparameters used for training the classification model.
            - 'val_loss': Best validation loss achieved during training.
        test_dataset (Dataset): The test dataset used for evaluation.
        split_info (tuple): Tuple containing test sample IDs, original distances, and ground truth labels.
            - sample_ids_test: IDs of the test samples.
            - original_distances_test: Original distances of the test samples.
            - Y_test: Ground truth labels for the test samples.
        distance (float): Distance threshold for classification evaluation.
        distance_folder (str): Folder name corresponding to the distance threshold.
        dataset_iter (str): Identifier for the current dataset iteration.
        chosen_dataset (str): Name of the dataset being used.
        device (torch.device): Device on which the models are evaluated (e.g., 'cpu' or 'cuda').
    Saves:
        - Precision-recall curve, confusion matrix, and ROC curve plots in the evaluation metrics folder.
        - Regression and classification model states in the models folder.
        - Cartesian and scan index plots in their respective folders.
        - Error metrics summary in a CSV file.
    Outputs:
        - Prints evaluation metrics such as MAE, RMSE, MRE, and classification accuracy.
        - Logs the completion of model evaluation and saving process.
    """
    sample_ids_test, original_distances_test, Y_test, original_distances_train, original_distances_val = split_info
    best_threshold = CLASSIFICATION_THRESHOLD
    predicted_test, ground_truth_test = evaluate_regressor(reg_results['model'], reg_results['test_loader'], device)
    
    class_test_dataset = ClassifierDataset(predicted_test, ground_truth_test)
    classifications, classifications_true, classifcation_list, classifcation_true_list, classifciation_preds_score = evaluate_classifier(
        cls_results['model'], device, DataLoader(class_test_dataset, cls_results['hyperparams']['batch_size'], True), distance)

    metrics_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, distance_folder, "evaluation_metrics")
    os.makedirs(metrics_folder, exist_ok=True)

    y_class_targets = classifications_true.flatten()
    y_class_probs = classifciation_preds_score.flatten()

    prec_score, rec_score, f1, accuracy = plot_precision_recall_curve(y_class_targets, y_class_probs,
                                save_path=os.path.join(metrics_folder, "precision_recall_curve.png"))
    cm = plot_confusion_matrix_all(y_class_targets, y_class_probs, threshold=best_threshold,
                              save_path=os.path.join(metrics_folder, "confusion_matrix.png"))
    roc_auc, thresholds = plot_roc_curve(y_class_targets, y_class_probs, save_path=os.path.join(metrics_folder, "roc_curve.png"))

    print("Saved evaluation plots.")

    mae, rmse, mre, range_metrics = [], [], [], []
    for i in range(len(ground_truth_test)):
        gt, pred = ground_truth_test.numpy()[i], predicted_test.numpy()[i]
        mae.append(np.nanmean(np.abs(gt - pred)))
        rmse.append(np.sqrt(np.nanmean((gt - pred) ** 2)))
        mre.append(np.nanmean(np.abs(gt - pred) / (gt + 1e-10)) if np.any(gt) else 0)
        range_metrics.append(compute_error_metrics(gt, pred, distance))

    mean_mae, mean_rmse, mean_mre = np.nanmean(mae), np.nanmean(rmse), np.nanmean(mre)
    print(f"MAE: {mean_mae:.4f}, RMSE: {mean_rmse:.4f}, MRE: {mean_mre:.4f}")

    model_dir = os.path.join("./Echolocation", "Models", distance_folder)
    os.makedirs(model_dir, exist_ok=True)
    torch.save({"model_state_dict": reg_results['model'].state_dict(), "hyperparameters": reg_results['hyperparams'], "classification_threshold": best_threshold},
               os.path.join(model_dir, f"{chosen_dataset}_{NUM_EPOCHS}_{reg_results['hyperparams']['num_layers']}_model_regressor.pth"))
    torch.save({"model_state_dict": cls_results['model'].state_dict(), "hyperparameters": cls_results['hyperparams'], "classification_threshold": best_threshold},
               os.path.join(model_dir, f"{chosen_dataset}_{NUM_EPOCHS}_model_classifier.pth"))

    cartesian_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, distance_folder, "cartesian_plots")
    scan_index_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, distance_folder, "scan_index_plots")
    os.makedirs(cartesian_folder, exist_ok=True)
    os.makedirs(scan_index_folder, exist_ok=True)
    
    # write best hyperparameters for this distance to file with the other results
    # calculate fpr and fnr using cm
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    # calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    with open(os.path.join(metrics_folder, "best_hyperparameters.txt"), 'a') as f:
        f.write(f"\nDataset iteration: {dataset_iter}\n")
        f.write(f"\nBest Regressor Hyperparameters: {reg_results['hyperparams']}")
        f.write(f"\n  Best Validation Loss: {round(reg_results['val_loss'],4)}")
        f.write(f"\n  MAE: {mean_mae:.4f}, RMSE: {mean_rmse:.4f}, MRE: {mean_mre:.4f}")
        f.write(f"\n\nBest Classifier Hyperparameters: {cls_results['hyperparams']}")
        f.write(f"\n  Best Validation Loss: {round(cls_results['val_loss'],4)}")
        f.write(f"\n  Precision: {round(prec_score,4)}, Recall: {round(rec_score,4)}, F1: {round(f1,4)}, Accuracy: {round(accuracy,4)}")
        f.write(f"\n  FPR: {fpr:.4f}, FNR: {fnr:.4f}, AUC: {roc_auc:.4f}")
        f.write(f"\n  Best Threshold: {thresholds}\n")
    

    if False:
        start_multiprocessing_plotting(ground_truth_test, predicted_test, classifcation_list, original_distances_test,
                                    NUM_EPOCHS, reg_results['hyperparams']['num_layers'], cartesian_folder,
                                    scan_index_folder, CLASSIFICATION_THRESHOLD, dataset_iter, sample_ids_test, distance)

    #acc = ((classifications > best_threshold) == (classifications_true <= distance)).mean()
    print(f"Classification Accuracy: {round(accuracy,4)}")

    # Save metrics summary
    header = ['chirp', 'range', 'best_validation_loss', 'mean_absolute_error', 'root_mean_square_error', 'mean_relative_error']
    error_metrics_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, distance_folder, "error_metrics.csv")
    with open(error_metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([chosen_dataset, "all", cls_results['val_loss'], mean_mae, mean_rmse, mean_mre])
    
    # save best regressor and classifier loss to json file
    
    # Calculate test loss for regressor
    regressor_test_loss = MaskedMSELoss()(predicted_test, ground_truth_test).item()

    # Calculate test loss for classifier
    class_test_loader = DataLoader(class_test_dataset, cls_results['hyperparams']['batch_size'], shuffle=False)
    cls_results['model'].eval()
    with torch.no_grad():
        test_class_loss = 0.0
        total = 0
        for x, y in class_test_loader:
            x, y = x.to(device), y.to(device)
            outputs = cls_results['model'](x)
            loss = nn.BCELoss()(outputs, y)
            test_class_loss += loss.item() * x.size(0)
            total += x.size(0)
        classifier_test_loss = test_class_loss / total if total > 0 else 0.0

    # Save test losses to JSON
    test_losses = {
        "regressor_test_loss": regressor_test_loss,
        "classifier_test_loss": classifier_test_loss,
        "classifier_accuracy": accuracy,
    }
    test_losses_file = os.path.join(metrics_folder, "test_losses.json")
    with open(test_losses_file, 'w') as f:
        json.dump(test_losses, f, indent=4)

    print("Model evaluation and saving complete.")


def main_training(dataset_root_directory, chosen_dataset):
    """
    Trains machine learning models using a specified dataset and evaluates their performance.
    This function performs the following steps:
    1. Prepares the dataset by loading and processing features from a CSV file.
    2. Splits the dataset into training, validation, and testing subsets.
    3. Performs hyperparameter grid search for a regressor model.
    4. Performs hyperparameter grid search for a classifier model using the regressor's results.
    5. Evaluates the models and saves the results to the appropriate directories.
    Args:
        dataset_root_directory (str): The root directory containing the dataset files.
        chosen_dataset (str): The name of the dataset to be used for training.
    Returns:
        None: The function saves evaluation metrics and model results to disk.
              If no valid samples or hyperparameters are found, the function exits early.
    """
    csv_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset,
                            "features_all_normalized.csv")
    time_stamp = time.strftime("%d-%m_%H-%M-%S")
    dataset_iter = chosen_dataset + "_" + time_stamp

    for distance in DISTANCE_THRESHOLDS:
        distance_folder = f"{distance}m_threshold"
        distance_path = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset,
                                     dataset_iter, distance_folder, "evaluation_metrics")
        os.makedirs(distance_path, exist_ok=True)

        data = prepare_dataset(csv_file, dataset_root_directory, distance)
        if data is None:
            print("No valid samples found. Exiting.")
            return

        X, Y, sample_ids, original_distances = data
        splits = split_data(X, Y, original_distances, sample_ids)
        train_dataset, val_dataset, test_dataset, *split_info = splits

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        input_dim = X.shape[1]
        output_dim = Y.shape[1]

        regressor_results = run_regressor_grid_search(
            train_dataset, val_dataset, test_dataset,
            input_dim, output_dim, device,
            dataset_iter, distance_folder, chosen_dataset
        )

        if regressor_results is None:
            print("Error: No valid regressor hyperparameters found during grid search.")
            return

        classifier_results = run_classifier_grid_search(
            regressor_results, output_dim, device,
            dataset_iter, distance_folder, chosen_dataset
        )

        if classifier_results is None:
            print("Error: No valid classifier hyperparameters found during grid search.")
            return

        print("\nBest Regressor Hyperparameters:", regressor_results['hyperparams'])
        print("Best Classifier Hyperparameters:", classifier_results['hyperparams'],"\n")
        
        evaluate_and_save_results(
            regressor_results, classifier_results, test_dataset,
            split_info, distance, distance_folder,
            dataset_iter, chosen_dataset, device
        )
