#!/usr/bin/env python3
import os
import json
import pandas as pd

def main():
    # Define the path to the JSON file with all features
    json_path = os.path.join("Extracted features", "features_all.json")
    
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Convert the dictionary to a list of records.
    # Each record will have a 'filename' field and the associated features.
    records = []
    for filename, features in data.items():
        record = {"filename": filename}
        record.update(features)
        records.append(record)
    
    # Flatten the records using pandas.json_normalize.
    # The 'sep' parameter defines how nested keys will be joined.
    df = pd.json_normalize(records, sep="_")
    
    # Display the first few rows of the dataframe
    print("DataFrame preview:")
    print(df.head())
    
    
    # Use pandas .describe() to get a statistical summary of numeric columns
    description = df.describe()
    print("DataFrame description:")
    print(description)
    
    # Save the descriptive statistics to a text file
    description_file = os.path.join("Extracted features", "features_description.txt")
    with open(description_file, "w") as f:
        f.write(description.to_string())
    print(f"DataFrame description saved to {description_file}")

    # Optionally, save the dataframe to a CSV file in the same output folder.
    output_csv = os.path.join("Extracted features", "features_all_df.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nDataFrame saved to {output_csv}")

if __name__ == '__main__':
    main()