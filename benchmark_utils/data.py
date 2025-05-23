import os
import json
import shutil

from benchopt.base import BaseDataset
from benchopt.config import get_data_path

# TODO: Add facility for custom cache validation ? 

def read_data_status(status_path):
    """
    Check what is the data status marked in the status.json file.
    """
    if not os.path.exists(status_path):
        return "none"
    try:
        with open(status_path, "r") as f:
            status = json.load(f)
            return status.get("status")
    except (json.JSONDecodeError, IOError):
        return "none"

def mark_data_status(status_path, status):
    """
    Write a status.json file with the given status ('none', 'raw', 'preprocessed').
    """
    with open(status_path, "w") as f:
        json.dump({"status": status}, f)

class DiskDataset(BaseDataset):
    
    def clean_folder(self, data_dir):
        shutil.rmtree(data_dir)
        os.makedirs(data_dir)
    
    def get_data(self):
        base_path = get_data_path()
        data_dir = os.path.join(base_path, self.name)
        raw_data_dir = os.path.join(data_dir, "raw")
        status_path = os.path.join(data_dir, "status.json")

        status = read_data_status(status_path)

        if status == "none":
            print(f"Read data status : 'none'")
            print(f"Cleaning (rm + mkdir) of current data dir {data_dir}")
            self.clean_folder()
            print(f"Downloading dataset {self.name}")
            self.download(raw_data_dir)
            mark_data_status(status_path, "raw")
            status = "raw"
        
        if status == "raw":
            print(f"Reading data status : 'raw'")
            print(f"Preprocessing and saving {self.name} data")
            self.preprocess(raw_data_dir, data_dir)
            mark_data_status(status_path, "preprocessed")
            status = "preprocessed"

        if status != "preprocessed":
            raise ValueError(f"data status should be 'preprocessed', current status is {status}")
        
        print(f"Using existing preprocessed {self.name} data.")
        X = self.load(data_dir)
        return dict(X=X)
    
    def download(self, raw_data_dir):
        raise NotImplementedError("Subclass of DiskDataset should implement download method")

    def preprocess(self, raw_data_dir, data_dir):
        raise NotImplementedError("Subclass of DiskDataset should implement preprocess method")
    
    def load(self, data_dir):
        raise NotImplementedError("Subclass of DiskDataset should implement load method")