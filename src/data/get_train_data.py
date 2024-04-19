import requests
import os
def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Download completed: {filename}")
    else:
        print("Failed to download the file.")

if __name__ == "__main__":
    URL = "https://huggingface.co/datasets/Xiao215/pixiv-image-with-caption/resolve/main/train_data.parquet?download=true"
    FILENAME = "train_data.parquet"
    FILENAME = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        FILENAME,
    )
    download_file(URL, FILENAME)
