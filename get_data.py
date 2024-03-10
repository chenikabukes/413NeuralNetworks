###
# REMEMBER TO LOG INTO YOUR KAGGLE WITH YOUR USERNAME AND API KEY TO GET THE DATA
###
import os
import subprocess

# Set the path to the dataset directory
dataset_dir = "/h/u6/c4/05/zha11021/CSC413/413NeuralNetworks/dataset"

# Optional Downloads with default options set to True
file_download_flags = {
    "rmdb_data.csv": True,
    "sample_submission.csv": True,
    "test_sequences.csv": True,
    "train_data.csv": True,
    "train_data_QUICK_START.csv": True,
    "Ribonanza_bpp_files.zip": False,
    "eterna_openknot_metadata.zip": False,
    "rhofold_pdbs.zip": False,
    "sequence_libraries.zip": False,
    "supplementary_silico_predictions.zip": False,
}

def download_and_unzip(file_name, dataset_dir):
    """
    Download and unzip a file from the Kaggle competition to the specified directory.
    """
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        subprocess.run(["kaggle", "competitions", "download", "-c", "stanford-ribonanza-rna-folding", "-f", file_name, "-p", dataset_dir, "--quiet"])
        zip_path = os.path.join(dataset_dir, file_name+".zip")
        print(f"Unzipping {zip_path}...")
        subprocess.run(["unzip", "-o", zip_path, "-d", dataset_dir])
        os.remove(zip_path)
    else:
        print(f"The file {file_name} already exists, skipping download.")

if __name__ == "__main__":
    # Ensure the dataset directory exists
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Loop through the files and download as necessary
    for file_name, should_download in file_download_flags.items():
        if should_download:
            download_and_unzip(file_name, dataset_dir)
