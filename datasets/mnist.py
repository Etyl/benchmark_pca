import os
import urllib.request
import gzip
import numpy as np

from benchmark_utils.data import DiskDataset

# Base URL and filenames for MNIST data
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {"train_images": "train-images-idx3-ubyte.gz", "test_images": "t10k-images-idx3-ubyte.gz"}


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
    with gzip.open(filepath, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")  # Magic number
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols).astype("float32") / 255.0


def get_mnist_on_disk(base_path):
    raw_path = download_mnist_to_disk(base_path)
    train_images = extract_images_from_file(os.path.join(raw_path, FILES["train_images"]))
    test_images = extract_images_from_file(os.path.join(raw_path, FILES["test_images"]))
    all_images = np.vstack([train_images, test_images])
    return all_images


class Dataset(DiskDataset):
    name = "mnist"
    parameters = {}
    requirements = ["numpy", "gzip"]

    def download(self, raw_data_dir):
        for filename in FILES.values():
            out_path = os.path.join(raw_data_dir, filename)
            urllib.request.urlretrieve(BASE_URL + filename, out_path)

    def extract_images_from_file(self, filepath):
        with gzip.open(filepath, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")  # Magic number
            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows * cols).astype("float32") / 255.0

    def preprocess_and_save(self, raw_data_dir, data_dir):
        train_images = self.extract_images_from_file(
            os.path.join(raw_data_dir, FILES["train_images"])
        )
        test_images = self.extract_images_from_file(
            os.path.join(raw_data_dir, FILES["test_images"])
        )
        all_images = np.vstack([train_images, test_images])
        np.save(os.path.join(data_dir, "data.npy"), all_images)

    def load(self, data_dir):
        X = np.load(os.path.join(data_dir, "data.npy"))
        return dict(X=X)
