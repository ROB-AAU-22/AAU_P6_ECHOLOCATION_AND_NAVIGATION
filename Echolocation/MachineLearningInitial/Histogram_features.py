#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

def main():
    # Path to the CSV file with features
    csv_file = os.path.join("Extracted features", "features_all_normalized.csv")
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Use pandas .describe() to get a statistical summary of numeric columns
    description = df.describe()
    print("DataFrame description:")
    print(description)
    
    # Save the descriptive statistics to a text file
    description_file = os.path.join("Extracted features", "features_description.txt")
    with open(description_file, "w") as f:
        f.write(description.to_string())
    print(f"DataFrame description saved to {description_file}")
    
    # Create a subdirectory for plots if it doesn't exist
    plots_dir = os.path.join("Extracted features", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Iterate over DataFrame columns (skip "filename") and plot numeric features.
    for col in df.columns:
        if col == "filename":
            continue
        if not is_numeric_dtype(df[col]):
            print(f"Skipping column '{col}' because it is not numeric.")
            continue

        # Create a new figure for this feature
        plt.figure()
        df[col].dropna().hist(bins=20)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
        # Save the plot to the plots folder
        plot_file = os.path.join(plots_dir, f"{col}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved plot for '{col}' to {plot_file}")
    
    print("\nPlotting complete and descriptive statistics saved.")

if __name__ == '__main__':
    main()
