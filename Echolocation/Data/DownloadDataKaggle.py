import os
import kaggle as kg

#os.environ['KAGGLE_USERNAME'] = '<villiamb>'
#os.environ['KAGGLE_KEY'] = '<f3d78baa2406e4f3196bb9ee583fed7f>'

DATASET_PATH_ZIP = "Echolocation/Data/dataset_zip"
DATASET_PATH = "Echolocation/Data/dataset/{}"

def DownloadDataset():
    kg.api.authenticate()

    datasets = kg.api.dataset_list(user="villiamb")
    print("Available Datasets:")
    for i, dataset in enumerate(datasets, start=1):
        print(f"[{i}] : {(dataset.ref).split('/')[1]}")

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
    #os.makedirs(DATASET_PATH_ZIP, exist_ok=True)  # Ensure zip path exists
    dataset_path = DATASET_PATH.format(selected_dataset)
    os.makedirs(dataset_path, exist_ok=True)  # Ensure data path exists
    kg.api.dataset_download_files(dataset=selected_dataset, path=dataset_path, quiet=True, unzip=True)
    print(f"{selected_dataset}.zip downloaded and contents extracted to {dataset_path}.")


if __name__ == "__main__":
    DownloadDataset()