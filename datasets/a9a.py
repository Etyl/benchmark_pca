from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml

def get_a9a(data_home):
    a9a = fetch_openml(name="a9a", data_home=data_home)
    a9a = a9a.data.toarray()
    return a9a

class Dataset(BaseDataset):

    name = "a9a"

    parameters = {
    }

    requirements = ["scikit-learn"]


    def get_data(self):
        data_home = get_data_path()
        return dict(X=get_a9a(data_home))
    
