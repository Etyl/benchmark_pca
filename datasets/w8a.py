from sklearn.datasets import fetch_openml

from benchopt import BaseDataset
from benchopt.config import get_data_path


class Dataset(BaseDataset):
    name = "w8a"

    parameters = {}

    requirements = ["scikit-learn"]

    def get_data(self):
        data_home = get_data_path()
        X = fetch_openml(name="w8a", data_home=data_home)
        X = X.data.toarray()
        return dict(X=X)
