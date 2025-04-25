import os
import kaggle as kg

os.environ['KAGGLE_USERNAME'] = '<Your_Kaggle_Username>'
os.environ['KAGGLE_KEY'] = '<Your_Kaggle_API_Key>'

DATASET_PATH_ZIP = "Echolocation/Data/dataset_zip"
DATASET_PATH = "Echolocation/Data/dataset"

def DownloadDataset():
    kg.api.authenticate()
    datasets = kg.api.dataset_list(user="villiamb")
    print("Available Datasets:")
    for i, dataset in enumerate(datasets, start=1):
        print(f"[{i}] : {(dataset.ref).split("/")[1]}")

    try:
        choice = int(input("Enter the number corresponding to the dataset to download and unzip: ")) - 1
        if choice < 0 or choice >= len(datasets):
            raise ValueError("Invalid choice. Please choose a valid index.")
    except ValueError as e:
        print(f"Error: {e}")
        return
    selected_dataset = datasets[choice].ref
    selected_dataset = selected_dataset.split("/")[1]

    dataset_zip_path = os.path.join(DATASET_PATH_ZIP, f"{selected_dataset}.zip")
    if os.path.exists(dataset_zip_path):
        print(f"{selected_dataset}.zip already downloaded.")
    else:
        os.makedirs(DATASET_PATH_ZIP, exist_ok=True)  # Ensure zip path exists
        os.makedirs(DATASET_PATH, exist_ok=True)  # Ensure data path exists
        kg.api.dataset_download_files(dataset=selected_dataset, path=DATASET_PATH, quiet=True, unzip=True)
        print(f"{selected_dataset}.zip downloaded and contents extracted to {DATASET_PATH}.")


if __name__ == "__main__":
    DownloadDataset()