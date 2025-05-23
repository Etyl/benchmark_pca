import os
import json

def is_data_valid(status_path):
    """
    Check if the data is marked as 'complete' in the status.json file.
    """
    if not os.path.exists(status_path):
        return False
    try:
        with open(status_path, "r") as f:
            status = json.load(f)
            return status.get("status") == "complete"
    except (json.JSONDecodeError, IOError):
        return False

def mark_data_status(status_path, status):
    """
    Write a status.json file with the given status ('incomplete' or 'complete').
    """
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w") as f:
        json.dump({"status": status}, f)
