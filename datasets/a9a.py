from benchopt import BaseDataset, safe_import_context
from env_vars import DATA_CACHE_DIR
from joblib import Memory

with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml

##TODO: See with Thomas if cache validation still works with dynamically wrapped function
memory = Memory(location=DATA_CACHE_DIR)
@memory.cache
def get_a9a():
    a9a = fetch_openml(name="a9a")
    a9a = a9a.data.toarray()
    return a9a

class Dataset(BaseDataset):

    name = "a9a"

    parameters = {
    }

    requirements = ["scikit-learn"]


    def get_data(self):
        return dict(X=get_a9a())
    
