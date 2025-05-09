import os
import pandas as pd

def normalize_features(chosen_dataset):
    # define path to feature csv
    input_feature_csv = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, "features_all.csv")
    output_normalized_feature_csv = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, "features_all_normalized.csv")
    output_mean_std_csv = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset, "mean_std.csv")

    df = pd.read_csv(input_feature_csv)
    
    df_normalized = df.copy()

    # Dictionary to store mean and std for each column
    mean_std_data = {"Feature": [], "Mean": [], "Std": []}
    
    for col in df.columns:
        # Skip non-numeric columns such as "filename"
        if df[col].dtype.kind not in 'biufc':
            continue
        
        # Compute z-score normalization for the column
        mean = df[col].mean()
        std = df[col].std()
        mean_std_data["Feature"].append(col)
        mean_std_data["Mean"].append(mean)
        mean_std_data["Std"].append(std)

        if std != 0:
            df_normalized[col] = (df[col] - mean) / std
        else:
            # if std is zero, all values are constant and we can simply fill with 0
            df_normalized[col] = 0.0
        
    # save the normalized DataFrame to a new CSV file
    df_normalized.to_csv(output_normalized_feature_csv, index=False)
    print(f"Normalized features saved to {output_normalized_feature_csv}")

    # Save the mean and std to a separate CSV file
    mean_std_df = pd.DataFrame(mean_std_data)
    mean_std_df.to_csv(output_mean_std_csv, index=False)
    print(f"Mean and standard deviation saved to {output_mean_std_csv}")

def normalize_single_feature_vector(feature_vector, mean_std_csv):
    """
    Normalize a single feature vector using the mean and standard deviation from a CSV file.
    
    Parameters:
        feature_vector (list): The feature vector to normalize.
        mean_std_csv (str): Path to the CSV file containing mean and std for each feature.
        
    Returns:
        list: The normalized feature vector.
    """
    # Load the mean and std values from the CSV file
    mean_std_df = pd.read_csv(mean_std_csv)
    print(f"Loaded mean and std from {mean_std_df}")
    
    # Create a dictionary for quick lookup of mean and std values
    mean_std_dict = dict(zip(mean_std_df["Feature"], zip(mean_std_df["Mean"], mean_std_df["Std"])))
    mean_std_list = list(mean_std_dict.values())
    print(f"Mean and std dictionary: {mean_std_list}")
    
    # Normalize the feature vector
    normalized_vector = []
    for i, value in enumerate(feature_vector):
        mean = mean_std_list[i][0]
        std = mean_std_list[i][1]
        if std != 0:
            normalized_value = (value - mean) / std
        else:
            normalized_value = 0.0
        normalized_vector.append(normalized_value)
    
    return normalized_vector