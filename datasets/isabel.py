import os
import gzip
import requests
import numpy as np

from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.data import is_data_valid, mark_data_status


# Constants
NUM_FRAMES = 48
SHAPE_PER_FRAME = (500, 500, 100)
TOTAL_ELEMENTS = np.prod(SHAPE_PER_FRAME)
URL_TEMPLATE = "https://www.earthsystemgrid.org/api/v1/dataset/isabeldata/file/TCf{frame:02d}.bin.gz"


def download_isabel_to_disk(base_path):
    raw_path = os.path.join(base_path, "isabel", "raw")
    os.makedirs(raw_path, exist_ok=True)

    for frame in range(1, NUM_FRAMES + 1):
        filename = f"TCf{frame:02d}.bin.gz"
        out_path = os.path.join(raw_path, filename)
        if not os.path.exists(out_path):
            url = URL_TEMPLATE.format(frame=frame)
            print(f"Downloading frame {frame} from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)
    return raw_path


def preprocess_isabel_from_disk(base_path):
    raw_path = download_isabel_to_disk(base_path)
    data = np.empty((NUM_FRAMES, *SHAPE_PER_FRAME), dtype=np.float32)

    for frame in range(1, NUM_FRAMES + 1):
        filename = f"TCf{frame:02d}.bin.gz"
        filepath = os.path.join(raw_path, filename)

        with gzip.open(filepath, 'rb') as f:
            raw_data = f.read()

        frame_data = np.frombuffer(raw_data, dtype='>f4')
        if frame_data.size != TOTAL_ELEMENTS:
            raise ValueError(f"Unexpected data size for frame {frame}")
        data[frame - 1] = frame_data.reshape(SHAPE_PER_FRAME)

    return data.reshape(NUM_FRAMES, -1)


class Dataset(BaseDataset):
    name = "isabel"
    parameters = {}
    requirements = ["numpy", "requests", "gzip"]

    def get_data(self):
        base_path = get_data_path()
        data_dir = os.path.join(base_path, "isabel")
        data_path = os.path.join(data_dir, "data.npy")
        status_path = os.path.join(data_dir, "status.json")

        if os.path.exists(data_path) and is_data_valid(status_path):
            print("Using cached Isabel dataset.")
            X = np.load(data_path)
        
        else:
            print("Processing Isabel dataset...")
            mark_data_status(status_path, "incomplete")

            X = preprocess_isabel_from_disk(base_path)

            os.makedirs(data_dir, exist_ok=True)
            np.save(data_path, X)
            mark_data_status(status_path, "complete")
        
        return dict(X=X)

    