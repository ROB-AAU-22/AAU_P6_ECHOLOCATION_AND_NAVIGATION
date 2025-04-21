import torch
from torchvision.datasets.utils import download_file_from_google_drive
import zipfile

#pip install gdown
DRIVE_FOLDER_LINK = "1oRUs8O4nU3b2e6uLXILHdhJwXXxJm91R"

def download_data(save_path, file_name):
    download_file_from_google_drive(DRIVE_FOLDER_LINK, save_path,file_name)
    return

if __name__ == "__main__":
    download_data("Echolocation/Data/dataset_zip", "dataset.zip")
    # extract
    with zipfile.ZipFile("Echolocation/Data/dataset_zip/dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("Echolocation/Data/dataset")



