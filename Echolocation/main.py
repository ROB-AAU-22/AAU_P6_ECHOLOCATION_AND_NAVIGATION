#!/usr/bin/env python3
import os
import subprocess
import sys

# Import local scripts
from FeatureExtraction import FeatureExtractionScript
from FeatureExtraction import NormalizeFeatures
from MachineLearning import ModelTrain

def main():
    train_predict_input = input("Would you like to train or predict? (train/predict) ").strip().lower()
    if train_predict_input == "train":
        dataset_root_path = os.path.join("./Echolocation/Data", "dataset")
        chosen_dataset = None
        # checking if the dataset directory exists (whether we have any data)
        if not os.path.exists(dataset_root_path):
            print(f"Dataset directory {dataset_root_path} does not exist.")
            # if we dont have any data, pull it from somewhere (?)
            return
        else:
            # choosing a dataset to train on 
            print("Choose a dataset to train on:")
            for i, dataset in enumerate(os.listdir(dataset_root_path)):
                dataset_path = os.path.join(dataset_root_path, dataset)
                if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
                    continue
                else:
                    print(f"{dataset} [{i}]")
            choose_dataset_input = input("Enter the index of the dataset you want to train: ")
            chosen_dataset = os.listdir(dataset_root_path)[int(choose_dataset_input)]
            dataset_root_path = os.path.join(dataset_root_path, chosen_dataset)
            print(f"Selected dataset: {dataset_root_path}")
        
        # optional to extract features
        extract_features_input = input("Would you like to extract features? (y/n) ").strip().lower()
        if extract_features_input == "y":
            print("Extracting features...")
            FeatureExtractionScript.extract_features(dataset_root_path, chosen_dataset)
            print("Normalizing features...")
            NormalizeFeatures.normalize_features(chosen_dataset)
        else:
            print("Skipping feature extraction.")
        
        # train model based on features
        extracted_features_path = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset + "_features_all_normalized.csv")
        
        if not os.path.exists(extracted_features_path):
            print(f"Feature file {extracted_features_path} does not exist.")
            print(f"Extracting features for {extracted_features_path}...")
            FeatureExtractionScript.extract_features(dataset_root_path, chosen_dataset)
            print("Normalizing features...")
            NormalizeFeatures.normalize_features(chosen_dataset)

        # otherwise train model
        print("Training model...")
        ModelTrain.model_training(dataset_root_path, chosen_dataset)
        print("Model training complete.")
        
    elif train_predict_input == "predict":
        print("predict script")
    else:
        print("Invalid input. Please enter 'train' or 'predict'.")
        return

if __name__ == '__main__':
    main()