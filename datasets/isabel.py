import os
import gzip
import requests
import numpy as np

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.data import DiskDataset


# Constants
NUM_FRAMES = 48
SHAPE_PER_FRAME = (500, 500, 100)
TOTAL_ELEMENTS = np.prod(SHAPE_PER_FRAME)
URL_TEMPLATE = (
    "https://www.earthsystemgrid.org/api/v1/dataset/isabeldata/file/TCf{frame:02d}.bin.gz"
)


class Dataset(DiskDataset):
    name = "isabel"
    parameters = {}
    requirements = ["numpy", "requests", "gzip"]

    def download(self, raw_data_dir):
        for frame in range(1, NUM_FRAMES + 1):
            filename = f"TCf{frame:02d}.bin.gz"
            out_path = os.path.join(raw_data_dir, filename)
            url = URL_TEMPLATE.format(frame=frame)
            print(f"Downloading frame {frame} from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)

    def preprocess_and_save(self, raw_data_dir, data_dir):
        data = np.empty((NUM_FRAMES, *SHAPE_PER_FRAME), dtype=np.float32)

        for frame in range(1, NUM_FRAMES + 1):
            filename = f"TCf{frame:02d}.bin.gz"
            filepath = os.path.join(raw_data_dir, filename)

            with gzip.open(filepath, "rb") as f:
                raw_data = f.read()

            frame_data = np.frombuffer(raw_data, dtype=">f4")
            if frame_data.size != TOTAL_ELEMENTS:
                raise ValueError(f"Unexpected data size for frame {frame}")
            data[frame - 1] = frame_data.reshape(SHAPE_PER_FRAME)

        data.reshape(NUM_FRAMES, -1)
        np.save(os.path.join(data_dir, "data.npy"), data)

    def load(self, data_dir):
        X = np.load(os.path.join(data_dir, "data.npy"))
        X = X.reshape(NUM_FRAMES, -1)
        return X
