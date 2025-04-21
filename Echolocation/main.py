#!/usr/bin/env python3
import os
import subprocess
import sys

# Import local scripts
from FeatureExtraction import Feature_extraction_CSV_0_2
from FeatureExtraction import Normalized_features
from MachineLearning import ML_pytourch


def run_feature_extraction(dataset_root):
    # Set dataset path
    Feature_extraction_CSV_0_2.dataset_root = dataset_root
    # Run feature extraction
    Feature_extraction_CSV_0_2.main()


def run_normalization():
    # Run normalization
    Normalized_features.main()


def run_ml_pipeline():
    # Run ML pipeline
    ML_pytourch.main()


def main(dataset_root):
    # Ensure dataset_root is absolute
    dataset_root = os.path.abspath(dataset_root)

    print(f"Running pipeline on dataset: {dataset_root}")

    # Step 1: Feature Extraction
    print("\n--- Extracting Features ---")
    run_feature_extraction(dataset_root)

    # Step 2: Feature Normalization
    print("\n--- Normalizing Features ---")
    run_normalization()

    # Step 3: Machine Learning Pipeline
    print("\n--- Running ML Pipeline ---")
    run_ml_pipeline()

    print("\nPipeline completed successfully.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_dataset>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    main(dataset_path)
