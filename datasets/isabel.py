
from benchopt import BaseDataset, safe_import_context
from env_vars import DATA_CACHE_DIR
from joblib import Memory

with safe_import_context() as import_ctx:
    import numpy as np
    import requests
    import gzip
    from io import BytesIO

memory = Memory(DATA_CACHE_DIR)

@memory.cache
def get_isabel_data():
    # Shape details
    num_frames = 48
    shape_per_frame = (500, 500, 100)
    total_elements_per_frame = np.prod(shape_per_frame)
    
    # Preallocate the array
    data = np.empty((num_frames, *shape_per_frame), dtype=np.float32)
    
    # URL template
    url_template = "https://www.earthsystemgrid.org/api/v1/dataset/isabeldata/file/TCf{frame:02d}.bin.gz"
    
    for frame in range(1, num_frames + 1):
        url = url_template.format(frame=frame)
        print(f"Downloading frame {frame} from {url}...")

        # Download and decompress
        response = requests.get(url)
        response.raise_for_status()
        with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
            raw_data = f.read()
        
        # Load data as big-endian float32 ('>f4')
        frame_data = np.frombuffer(raw_data, dtype='>f4')  # > = big-endian, f4 = float32
        if frame_data.size != total_elements_per_frame:
            raise ValueError(f"Unexpected data size for frame {frame}: {frame_data.size} elements")
        
        # Store in the main array
        data[frame - 1] = frame_data.reshape(shape_per_frame)
    
    # Reshape to (48, -1)
    return data.reshape(num_frames, -1)

class Dataset(BaseDataset):

    name = "isabel"

    parameters = {
    }
    
    requirements = ["numpy", "gzip"]

    def get_data(self):
        return dict(X=get_isabel_data())
    