from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml


def get_w8a(data_home):
    w8a = fetch_openml(name="w8a", data_home=data_home)
    w8a = w8a.data.toarray()
    return w8a

class Dataset(BaseDataset):

    name = "w8a"

    parameters = {
    }
    
    requirements = ["scikit-learn"]


    def get_data(self):
        return dict(X=get_w8a(get_data_path()))
    
