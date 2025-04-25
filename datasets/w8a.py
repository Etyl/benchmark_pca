from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml
    from env_vars import DATA_CACHE_DIR
    from joblib import Memory


##TODO: See with Thomas if cache validation still works with dynamically wrapped function
memory = Memory(location=DATA_CACHE_DIR)
@memory.cache
def get_w8a():
    w8a = fetch_openml(name="w8a")
    w8a = w8a.data.toarray()
    return w8a

class Dataset(BaseDataset):

    name = "w8a"

    parameters = {
    }
    
    requirements = ["scikit-learn"]


    def get_data(self):
        return dict(X=get_w8a())
    
