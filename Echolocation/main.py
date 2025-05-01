#!/usr/bin/env python3
import os
import subprocess
import sys
import time

# Import local scripts
from FeatureExtraction import FeatureExtractionScript
from FeatureExtraction import NormalizeFeatures
from MachineLearning.Training import mainTraining
from Data import DownloadDataKaggle
from MachineLearning.ModelPredict import load_model as load_model


def main():
    train_predict_input = input("Would you like to train or predict? (t/p) ").strip().lower()
    if train_predict_input == "t" or str(train_predict_input) == "0":
        dataset_root_path = os.path.join("./Echolocation/Data", "dataset")
        chosen_dataset = None
        # checking if the dataset directory exists (whether we have any data)
        download_new_dataset = input("Would you like to download a new dataset? (y/n) ").strip().lower()
        if download_new_dataset == "y":
            DownloadDataKaggle.DownloadDataset()
        else:
            print("Skipping dataset download.")
        # choosing a dataset to train on
        print("Choose a dataset to train on:")
        for i, dataset in enumerate(os.listdir(dataset_root_path)):
            dataset_path = os.path.join(dataset_root_path, dataset)
            if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
                continue
            else:
                print(f"{dataset} [{i}]")
        choose_dataset_input = input("Enter the index of the dataset you want to train: ")
        try:
            chosen_dataset = os.listdir(dataset_root_path)[int(choose_dataset_input)]
        except IndexError as e:
            print(f"Invalid index: {e}. Please choose a valid dataset index.")
            return
        dataset_root_path = os.path.join(dataset_root_path, chosen_dataset)
        print(f"Selected dataset: {dataset_root_path}")

        # optional to extract features
        extract_features_input = input("Would you like to extract features? (y/n) ").strip().lower()
        if extract_features_input == "y" or extract_features_input == 0:
            print("Extracting features...")
            FeatureExtractionScript.extract_features(dataset_root_path, chosen_dataset)
            print("Normalizing features...")
            NormalizeFeatures.normalize_features(chosen_dataset)
        else:
            print("Skipping feature extraction.")


        # train model based on features
        extracted_features_path = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset,
                                               "features_all_normalized.csv")

        if not os.path.exists(extracted_features_path):
            print(f"Feature file {extracted_features_path} does not exist.")
            print(f"Extracting features for {extracted_features_path}...")
            FeatureExtractionScript.extract_features(dataset_root_path, chosen_dataset)
            print("Normalizing features...")
            NormalizeFeatures.normalize_features(chosen_dataset)

        # train model
        print("Training model...")
        mainTraining.model_training(dataset_root_path, chosen_dataset)
        print("Model training complete.")

    elif train_predict_input == "p" or str(train_predict_input) == "1":
        print("predict script")
        model, hyperparameters = load_model(200, 2)
        print("Model: \n", model)
        print("hyperparameters: \n", hyperparameters)
    else:
        print("Invalid input. Please enter 'train' or 'predict'.")
        return


if __name__ == '__main__':
    main()