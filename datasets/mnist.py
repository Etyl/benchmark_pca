import os
import urllib.request
import gzip

from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path
from benchmark_utils.data import is_data_valid, mark_data_status

with safe_import_context() as import_ctx:
    import numpy as np

# Base URL and filenames for MNIST data
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz"
}

def download_mnist_to_disk(base_path):
    raw_path = os.path.join(base_path, "mnist", "raw")
    os.makedirs(raw_path, exist_ok=True)
    for filename in FILES.values():
        out_path = os.path.join(raw_path, filename)
        if not os.path.exists(out_path):
            print(f"Downloading {filename} to {out_path}...")
            urllib.request.urlretrieve(BASE_URL + filename, out_path)
    return raw_path

def extract_images_from_file(filepath):
    with gzip.open(filepath, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # Magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols).astype('float32') / 255.0

def get_mnist_on_disk(base_path):
        raw_path = download_mnist_to_disk(base_path)
        train_images = extract_images_from_file(os.path.join(raw_path, FILES["train_images"]))
        test_images = extract_images_from_file(os.path.join(raw_path, FILES["test_images"]))
        all_images = np.vstack([train_images, test_images])
        return all_images

class Dataset(BaseDataset):

    name = "mnist"
    parameters = {}
    requirements = ["numpy", "gzip"]

    def get_data(self):
        base_path = get_data_path()
        data_dir = os.path.join(base_path, "mnist")
        data_path = os.path.join(data_dir, "data.npy")
        status_path = os.path.join(data_dir, "status.json")

        if os.path.exists(data_path) and is_data_valid(status_path):
            print("Using existing preprocessed MNIST data.")
            X = np.load(data_path).astype(np.float32)

        else: 
            print("Processing MNIST data from raw files...")

            # Mark as incomplete at the beginning
            mark_data_status(status_path, "incomplete")

            X = get_mnist_on_disk(base_path)

            os.makedirs(data_dir, exist_ok=True)
            np.save(data_path, X)

            # Mark as complete at the end
            mark_data_status(status_path, "complete")

        return dict(X=X)
