import os
import pandas as pd

def normalize_features(chosen_dataset):
    # define path to feature csv
    input_feature_csv = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset + "_features_all.csv")
    output_normalized_feature_csv = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", chosen_dataset + "_features_all_normalized.csv")
    
    df = pd.read_csv(input_feature_csv)
    
    df_normalized = df.copy()
    
    for col in df.columns:
        # Skip non-numeric columns such as "filename"
        if df[col].dtype.kind not in 'biufc':
            continue
        
        # Compute z-score normalization for the column
        mean = df[col].mean()
        std = df[col].std()
        if std != 0:
            df_normalized[col] = (df[col] - mean) / std
        else:
            # if std is zero, all values are constant and we can simply fill with 0
            df_normalized[col] = 0.0
        
    # save the normalized DataFrame to a new CSV file
    df_normalized.to_csv(output_normalized_feature_csv, index=False)
    print(f"Normalized features saved to {output_normalized_feature_csv}")