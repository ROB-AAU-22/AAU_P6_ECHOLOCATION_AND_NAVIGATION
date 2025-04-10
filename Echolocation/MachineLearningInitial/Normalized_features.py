import os
import pandas as pd

def main():
    # Define paths
    input_csv = os.path.join("Extracted features", "features_all.csv")
    output_csv = os.path.join("Extracted features", "features_all_normalized.csv")
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Copy DataFrame to avoid changing original values for non-numeric columns
    df_normalized = df.copy()
    
    # Loop over each column in the DataFrame
    for col in df.columns:
        # Skip non-numeric columns such as "filename"
        if df[col].dtype.kind not in 'biufc':  # b, i, u, f, c: boolean, integer, unsigned integer, float, complex
            continue
        
        # Compute z-score normalization for the column
        mean = df[col].mean()
        std = df[col].std()
        if std != 0:
            df_normalized[col] = (df[col] - mean) / std
        else:
            # if std is zero, all values are constant and we can simply fill with 0
            df_normalized[col] = 0.0

    # Save the normalized DataFrame to a new CSV file
    df_normalized.to_csv(output_csv, index=False)
    
    print(f"Normalized features saved to {output_csv}")

if __name__ == '__main__':
    main()