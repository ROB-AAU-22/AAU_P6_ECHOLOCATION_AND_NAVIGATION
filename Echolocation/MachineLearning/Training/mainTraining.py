## mainTraining.py
#!/usr/bin/env python3
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from MachineLearning.Training.TrainingConfig import NUM_EPOCHS, CLASSIFICATION_THRESHOLDS, DISTANCE_THRESHOLD, DISTANCE_THRESHOLDS, REGRESSOR_LEARNING_RATES, REGRESSOR_HIDDEN_SIZES, REGRESSOR_BATCH_SIZES, REGRESSOR_NUM_LAYERS_LIST, REGRESSOR_WEIGHT_DECAYS, REGRESSOR_LAYER_TYPE, CLASSIFIER_LEARNING_RATES, CLASSIFIER_HIDDEN_SIZES, CLASSIFIER_BATCH_SIZES, CLASSIFIER_NUM_LAYERS_LIST, CLASSIFIER_WEIGHT_DECAYS, CLASSIFIER_LAYER_TYPE
from MachineLearning.Training.DataHandler import build_dataset_from_csv, ClassifierDataset
from MachineLearning.Training.ModelFunctions import MaskedMSELoss, AudioLidarDataset, Regressor, Classifier
from MachineLearning.Training.ModelTraining import compute_error_metrics, train_regressor, evaluate_regressor, train_classifier, evaluate_classifier
from MachineLearning.Training.Plotting import start_multiprocessing_plotting
from MachineLearning.Training.EvaluationPlots import plot_precision_recall_curve, plot_confusion_matrix_all, plot_roc_curve


def model_training(dataset_root_directory, chosen_dataset):
    csv_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset,
                            "features_all_normalized.csv")

    time_stamp = time.strftime("%d-%m_%H-%M-%S")
    # add mm-dd-hh:mm:ss to dataset
    dataset_iter = chosen_dataset + "_" + time_stamp
    
    for distance in DISTANCE_THRESHOLDS:
        distance_folder = f"{distance}m_threshold"
        distance_path = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset,  dataset_iter,
                                      distance_folder, "evaluation_metrics")
        os.makedirs(distance_path, exist_ok=True)
        X, Y, sample_ids, feature_names_full, original_distances = build_dataset_from_csv(csv_file, dataset_root_directory, distance)
        if X.shape[0] == 0:
            print("No valid samples found. Exiting.")
            return
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, LiDAR scan length: {Y.shape[1]}")

        print("X stats: min", np.min(X), "max", np.max(X))
        print("Y stats: min", np.min(Y), "max", np.max(Y))

        # Split dataset into train (70%), validation (15%), and test (15%)
        X_train_val, X_test, Y_train_val, Y_test, original_distances_train_val, original_distances_test, sample_ids_train_val, sample_ids_test = train_test_split(
            X, Y, original_distances, sample_ids, test_size=0.15, random_state=42)

        X_train, X_val, Y_train, Y_val, original_distances_train, original_distances_val, sample_ids_train, sample_ids_val = train_test_split(
            X_train_val, Y_train_val, original_distances_train_val, sample_ids_train_val, test_size=0.15, random_state=42)
        
        print("Training samples: ", X_train.shape[0], "Validation samples: ", X_val.shape[0], "Test samples: ",
              X_test.shape[0])

        train_dataset = AudioLidarDataset(X_train, Y_train)
        val_dataset = AudioLidarDataset(X_val, Y_val)
        test_dataset = AudioLidarDataset(X_test, Y_test)

        input_dim = X.shape[1]
        output_dim = Y.shape[1]

        # Use GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        best_regressor_val_loss = float("inf")
        best_classifier_val_loss = float("inf")
        best_classifier_hyperparams = None
        best_model_state = None
        best_threshold = 0.5

        # Define the global custom loss function
        regression_loss_fn = MaskedMSELoss()
        classification_loss_fn = nn.BCELoss()
        


        # Grid search over regression hyperparameters
        for lr in REGRESSOR_LEARNING_RATES:
            for hidden_size in REGRESSOR_HIDDEN_SIZES:
                for batch_size in REGRESSOR_BATCH_SIZES:
                    for num_layers in REGRESSOR_NUM_LAYERS_LIST:
                        for decay in REGRESSOR_WEIGHT_DECAYS:
                            for layer_type in REGRESSOR_LAYER_TYPE:

                                print(
                                    f"\nTraining Regressor: hidden={hidden_size}, batch={batch_size}, layers={num_layers}, type={layer_type}, decay={decay}")

                                # Initialize model and optimizer
                                regressor = Regressor(input_dim, hidden_size, output_dim, num_layers=num_layers, layer_type=layer_type).to(
                                    device)
                                optimizer_r = optim.Adam(regressor.parameters(), lr=lr, weight_decay=decay)

                                # scheduler
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_r, mode='min', factor=0.1, patience=5)

                                # Train regressor
                                regressor, avg_val_loss, preds_train_list, preds_val_list, targets_train_list, targets_val_list = train_regressor(
                                    regressor,
                                    DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                                    DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
                                    optimizer_r,
                                    regression_loss_fn,
                                    device,
                                    NUM_EPOCHS,
                                    scheduler
                                )

                                print(f"Val Accuracy: {avg_val_loss:.4f}")
                                regressor_hyperparams = {
                                        "input_dim": input_dim,
                                        "output_dim": output_dim,
                                        "lr": lr,
                                        "hidden_size": hidden_size,
                                        "batch_size": batch_size,
                                        "num_epochs": NUM_EPOCHS,
                                        "num_layers": num_layers,
                                        "layer_type": layer_type,
                                        "weight_decay": decay
                                    }
                                if avg_val_loss < best_regressor_val_loss:
                                    print("New best regressor model found.")
                                    best_regressor_val_loss = avg_val_loss
                                    best_regressor_hyperparams = regressor_hyperparams
                                    best_regressor_state = regressor.state_dict()
                                    best_regressor_train_preds = preds_train_list
                                    best_regressor_val_preds = preds_val_list
                                    best_regressor_train_targets = targets_train_list
                                    best_regressor_val_targets = targets_val_list
                                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                                
                                regressor_info_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, distance_folder, "regressor_results.txt")
                                regressor_info_text = (
                                    f"Regressor Hyperparameters: {regressor_hyperparams}\n"
                                    f"  Best Regressor Loss: {avg_val_loss:.4f}\n\n\n"
                                )

                                # Write or append to the file
                                try:
                                    with open(regressor_info_file, 'x') as f:
                                        f.write(regressor_info_text)
                                except FileExistsError:
                                    with open(regressor_info_file, 'a') as f:
                                        f.write("\n" + regressor_info_text)

        if best_regressor_hyperparams is None:
            print("Error: No valid regressor hyperparameters found during grid search.")
            return
        print("Best regressor hyperparams:", best_regressor_hyperparams)
        best_regressor = Regressor(best_regressor_hyperparams["input_dim"], best_regressor_hyperparams["hidden_size"], best_regressor_hyperparams["output_dim"],
                                   num_layers=best_regressor_hyperparams["num_layers"], layer_type=best_regressor_hyperparams["layer_type"]).to(device)
        best_regressor.load_state_dict(best_regressor_state)


        #y_val_labels = (original_distances <= DISTANCE_THRESHOLD).float()
        print("\nStarting classification grid search with best regressor model...")
        # Create classifier training dataset from all regressor predictions
        #print(f"train preds: {best_regressor_train_preds}")
        #print(f"train targets: {best_regressor_train_targets}")
        classifier_dataset_train = ClassifierDataset(best_regressor_train_preds, best_regressor_train_targets)
        classifier_dataset_val = ClassifierDataset(best_regressor_val_preds, best_regressor_val_targets)

        # Grid search over classification hyperparameters
        for lr in CLASSIFIER_LEARNING_RATES:
            for hidden_size in CLASSIFIER_HIDDEN_SIZES:
                for batch_size in CLASSIFIER_BATCH_SIZES:
                    for num_layers in CLASSIFIER_NUM_LAYERS_LIST:
                        for decay in CLASSIFIER_WEIGHT_DECAYS:
                            for layer_type in CLASSIFIER_LAYER_TYPE:
                                classifier = Classifier(input_dim=output_dim, hidden_dim=hidden_size, output_dim=output_dim, num_layers=num_layers, layer_type=layer_type).to(device)
                                optimizer_c = optim.Adam(classifier.parameters(), lr=lr, weight_decay=decay)

                                print(
                                    f"\nTraining Classifier: hidden={hidden_size}, batch={batch_size}, layers={num_layers}, type={layer_type}, decay={decay}")

                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, mode='min', factor=0.1,
                                                                                    patience=5)

                                classifier, avg_val_loss = train_classifier(
                                    classifier,
                                    DataLoader(classifier_dataset_train, batch_size=batch_size, shuffle=True),
                                    DataLoader(classifier_dataset_val, batch_size=batch_size, shuffle=False),
                                    optimizer_c,
                                    classification_loss_fn,
                                    device,
                                    NUM_EPOCHS,
                                    scheduler
                                )


                                print(f"Val Accuracy: {avg_val_loss:.4f}")
                                classifier_hyperparams = {
                                        "input_dim": output_dim,
                                        "output_dim": output_dim,
                                        "lr": lr,
                                        "hidden_size": hidden_size,
                                        "batch_size": batch_size,
                                        "num_epochs": NUM_EPOCHS,
                                        "num_layers": num_layers,
                                        "layer_type": layer_type,
                                        "weight_decay": decay
                                    }
                                if avg_val_loss < best_classifier_val_loss:
                                    print("New best classifier model found.")
                                    best_classifier_val_loss = avg_val_loss
                                    best_classifier_hyperparams = classifier_hyperparams
                                    best_classifier_state = classifier.state_dict()
                                classifier_info_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter,distance_folder, "classifier_results.txt")
                                classifier_info_text = (
                                    f"Classifier Hyperparameters: {classifier_hyperparams}\n"
                                    f"  Best classifier Loss: {avg_val_loss:.4f}\n\n\n"
                                )

                                # Write or append to the file
                                try:
                                    with open(classifier_info_file, 'x') as f:
                                        f.write(classifier_info_text)
                                except FileExistsError:
                                    with open(classifier_info_file, 'a') as f:
                                        f.write("\n" + classifier_info_text)

        if best_classifier_hyperparams == None:
            print("Error: No valid classifier hyperparameters found during grid search.")
            return

        print("\nBest hyperparameters found:")
        print("Best regressor hyperparams:", best_regressor_hyperparams)
        print("Best classifier hyperparams:", best_classifier_hyperparams)
        print(f"Best overall regressor loss: {best_regressor_val_loss:.4f}")
        print(f"Best overall classifier loss: {best_classifier_val_loss:.4f}")
        # Build and load the best classifier model
        best_classifier = Classifier(input_dim=best_classifier_hyperparams["input_dim"], hidden_dim=best_classifier_hyperparams["hidden_size"], output_dim= best_classifier_hyperparams["output_dim"],
                                   num_layers=best_classifier_hyperparams["num_layers"], layer_type=best_classifier_hyperparams["layer_type"]).to(device)
        best_classifier.load_state_dict(best_classifier_state, strict=True)

        # Run regression on test set
        predicted_test, ground_truth_test = evaluate_regressor(
            best_regressor, test_loader, device
        )

        # Run classification on predicted distances
        classifications, classifications_true, classifcation_list, classifcation_true_list = evaluate_classifier(best_classifier, device, predicted_test, ground_truth_test,best_classifier_hyperparams["batch_size"])

        #classifcation_list = np.concatenate(classifcation_list)

        metrics_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset,  dataset_iter,
                                      distance_folder, "evaluation_metrics")
        os.makedirs(metrics_folder, exist_ok=True)

        # Compute binary labels for classification
        y_class_targets = classifications_true.flatten()
        y_class_probs = classifications.flatten()
        
        #print(f"y_class_targets: {y_class_targets}")
        #print(f"y_class_probs: {y_class_probs}")

        threshold_to_use = CLASSIFICATION_THRESHOLDS[0]

        # Plot metrics
        plot_precision_recall_curve(y_class_targets, y_class_probs,
                                    save_path=os.path.join(metrics_folder, "precision_recall_curve.png"))
        plot_confusion_matrix_all(y_class_targets, y_class_probs, threshold=threshold_to_use,
                                  save_path=os.path.join(metrics_folder, "confusion_matrix.png"))
        plot_roc_curve(y_class_targets, y_class_probs, save_path=os.path.join(metrics_folder, "roc_curve.png"))

        print("Saved precision-recall and confusion matrix plots.")

        mae_array = []
        rmse_array = []
        mre_array = []
        range_metrics = []

        for i in range(len(ground_truth_test)):
            # add all non nan numbers to array
            mae_array.append(np.nanmean(np.abs(ground_truth_test.numpy()[i] - predicted_test.numpy()[i])))
            rmse_array.append(np.sqrt(np.nanmean((ground_truth_test.numpy()[i] - predicted_test.numpy()[i]) ** 2)))
            mre_array.append(np.nanmean(np.abs((ground_truth_test.numpy()[i] - predicted_test.numpy()[i]) / (ground_truth_test.numpy()[i] + 1e-10))) if np.any(ground_truth_test.numpy()[i]) else 0)
            range_metrics.append(compute_error_metrics(ground_truth_test.numpy()[i], predicted_test.numpy()[i], distance))

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

        # Save the best model.
        models_folder = os.path.join("./Echolocation", "Models",distance_folder)
        os.makedirs(models_folder, exist_ok=True)
        model_file_regressor = os.path.join(models_folder,
                                  f"{chosen_dataset}_{NUM_EPOCHS}_{best_regressor_hyperparams['num_layers']}_model_regressor.pth")
        torch.save({
            "model_state_dict": best_regressor.state_dict(),
            "hyperparameters": best_regressor_hyperparams,
            "classification_threshold": best_threshold
        }, model_file_regressor)
        model_file_classifier = os.path.join(models_folder,
                                            f"{chosen_dataset}_{NUM_EPOCHS}_model_classifier.pth")
        torch.save({
            "model_state_dict": best_classifier.state_dict(),
            "hyperparameters": best_classifier_hyperparams,
            "classification_threshold": best_threshold
        }, model_file_classifier)

        print("Saving comparison plots...")
        # Create folders

        cartesian_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter,
                                        distance_folder, f"cartesian_plots")
        scan_index_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter,
                                         distance_folder, f"scan_index_plots")
        os.makedirs(cartesian_folder, exist_ok=True)
        os.makedirs(scan_index_folder, exist_ok=True)


        # Create and start workers
        start_multiprocessing_plotting(
    ground_truth_test, predicted_test, classifcation_list, original_distances_test,
    NUM_EPOCHS, best_regressor_hyperparams['num_layers'], cartesian_folder, scan_index_folder,
    CLASSIFICATION_THRESHOLDS, dataset_iter, sample_ids_test
)


        print(f"Best model saved to {model_file_regressor}")

        # Evaluate classification accuracy
        classification_accuracy = ((classifications > best_threshold) == (classifications_true <= distance)).mean()
        print(f"Classification Accuracy: {classification_accuracy:.4f}")
        print("\nBest hyperparameters found:")
        print(best_classifier_hyperparams)
        print(f"Best classification threshold: {best_threshold}")
        print(f"Best validation classification accuracy: {best_classifier_val_loss:.4f}")
        print(f"Best model saved to {model_file_regressor}")

        # Saving error metrics to seperate CSV for each range
        header = ['chirp', 'range', 'best_validation_loss', 'mean_absolute_error', 'root_mean_square_error',
                  'mean_relative_error']
        error_metrics_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter,distance_folder, "error_metrics.csv")
        for range_bin in range_metrics_average:
            row = [chosen_dataset, range_bin, best_classifier_val_loss, range_metrics_average[range_bin]['mae'],
                   range_metrics_average[range_bin]['rmse'], range_metrics_average[range_bin]['mre']]

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
        print("\nBest hyperparameters found:")
        print("Best regressor hyperparams:\n", best_regressor_hyperparams)
        print(f"Best overall regressor loss: {best_regressor_val_loss:.4f}")

        print(
            f"\nBest classifier hyperparameters:\n{best_classifier_hyperparams}\n"
            f"Best overall classifier loss: {best_classifier_val_loss:.4f}\n"
            f"Mean Absolute Error: {mean_absolute_error:.4f}\n"
            f"Root Mean Square Error: {root_mean_square_error:.4f}\n"
            f"Mean Relative Error: {mean_relative_error:.4f}\n"
            f"Classification Accuracy: {classification_accuracy:.4f}"
        )
        # Save distance threshold and other details to a text file
        distance_info_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, dataset_iter, "models_info.txt")
        distance_info_text = (
            f"Distance Threshold: {distance}\n"
            f"  Best Regressor Hyperparameters: {best_regressor_hyperparams}\n"
            f"      Best Regressor Loss: {best_regressor_val_loss:.4f}\n"
            f"  Best Classifier Hyperparameters: {best_classifier_hyperparams}\n"
            f"      Classification Accuracy: {classification_accuracy:.4f}\n"
            f"      Best Classifier Loss: {best_classifier_val_loss:.4f}\n"
            f"      Mean Absolute Error: {mean_absolute_error:.4f}\n"
            f"      Root Mean Square Error: {root_mean_square_error:.4f}\n"
            f"      Mean Relative Error: {mean_relative_error:.4f}\n\n\n"
            
        )

        # Write or append to the file
        try:
            with open(distance_info_file, 'x') as f:
                f.write(distance_info_text)
        except FileExistsError:
            with open(distance_info_file, 'a') as f:
                f.write("\n" + distance_info_text)

        print(f"Distance threshold and model details saved to {distance_info_file}")

