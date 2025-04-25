from benchopt import BaseDataset, safe_import_context
from benchopt.benchmark import get_running_benchmark

with safe_import_context() as import_ctx:
    import numpy as np
    import io
    import urllib.request
    import gzip
    import numpy as np


# URLs and filenames for MNIST data
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz"
}

def download_mnist_in_memory():
    """Download MNIST dataset files into memory (no disk writing)."""
    mnist_files = {}
    for name, filename in FILES.items():
        url = BASE_URL + filename
        print(f"Downloading {filename}...")
        with urllib.request.urlopen(url) as response:
            mnist_files[name] = io.BytesIO(response.read())
    return mnist_files

def extract_images_from_memory(mnist_files, filename):
    """Extract images from the MNIST file (from in-memory bytes)."""
    with gzip.GzipFile(fileobj=mnist_files[filename]) as f:
        _ = int.from_bytes(f.read(4), 'big')  # Magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols).astype('float32') / 255.0

def get_mnist():
    """Download and prepare the MNIST dataset in memory."""
    # Download MNIST files into memory
    mnist_files = download_mnist_in_memory()

    # Extract images for both train and test sets
    train_images = extract_images_from_memory(mnist_files, "train_images")
    test_images = extract_images_from_memory(mnist_files, "test_images")

    # Combine train and test images
    all_images = np.vstack([train_images, test_images])

    return all_images

class Dataset(BaseDataset):

    name = "mnist"

    parameters = {
    }
    
    requirements = ["numpy", "gzip"]

    def get_data(self):
        benchmark = get_running_benchmark()
        cached = benchmark.cache(get_mnist)
        return dict(X=cached())
    
