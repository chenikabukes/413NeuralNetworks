###
# REMEMBER TO LOG INTO YOUR KAGGLE WITH YOUR USERNAME AND API KEY TO GET THE DATA

###
import os
import subprocess
import pandas as pd
import numpy as np

# Set the path to the dataset directory
# dataset_dir = "/h/u6/c4/05/zha11021/CSC413/413NeuralNetworks/data"
# dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + 'data')

# Optional Downloads with default options set to True
file_download_flags = {
    "rmdb_data.csv": True,
    "sample_submission.csv": True,
    "test_sequences.csv": True,
    "train_data.csv": True,
    "train_data_QUICK_START.csv": True,  # cleaned data, maybe buggy
    "Ribonanza_bpp_files": False,
    "eterna_openknot_metadata": False,
    "rhofold_pdbs": False,
    "sequence_libraries": False,
    "supplementary_silico_predictions": False,
}


def download_and_convert(file_name, dataset_dir):
    # Determine if the file needs conversion to parquet
    needs_conversion = file_name.endswith(".csv")
    base_name = file_name[:-4]
    # base_name = os.path.splitext(file_name)
    final_path = os.path.join(dataset_dir, base_name + (".parquet" if needs_conversion else ""))

    if os.path.exists(final_path):
        print(f"The file {os.path.basename(final_path)} already exists, skipping download.")
        return

    # # Download the file from Kaggle
    print(f"Downloading {file_name}...")
    subprocess.run(["kaggle", "competitions", "download", "-c", "stanford-ribonanza-rna-folding", "-f", file_name, "-p", dataset_dir, "--quiet"], check=True)

    # Unzip if a ZIP file was downloaded
    zip_path = os.path.join(dataset_dir, file_name + ".zip")
    if os.path.exists(zip_path):
        print(f"Unzipping {zip_path}...")
        subprocess.run(["unzip", "-o", zip_path, "-d", dataset_dir], check=True)
        os.remove(zip_path)

    # Convert to parquet if needed
    if needs_conversion:
        csv_path = os.path.join(dataset_dir, file_name)
        print(f"Converting {file_name} to Parquet format...")
        df = pd.read_csv(csv_path)
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        df.to_parquet(final_path)
        os.remove(csv_path)

if __name__ == "__main__":
    # Ensure the dataset directory exists
    dataset_dir = "../../data"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Loop through the files and download as necessary
    for file_name, should_download in file_download_flags.items():
        if should_download:
            download_and_convert(file_name, dataset_dir)
