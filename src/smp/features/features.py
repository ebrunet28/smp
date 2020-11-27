import pandas as pd
from smp import data_dir
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from scipy.sparse.csr import csr_matrix
from typing import Union
import numpy as np


class Loader:
    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "train.csv", index_col="Id"
        )
        self.test: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "test.csv", index_col="Id"
        )


class Base(ABC):
    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def transform(self, X):
        return X

    @property
    @abstractmethod
    def description(self):
        return "Describe your class"

    def to_step(self):
        return self.description, self


class Dataset(Base):
    def __init__(self, features):
        self.features = [f for f in features]

    def fit(self, X, y=None):
        for feature in self.features:
            feature.fit(X[feature.col_name])

    def transform(self, X):
        return pd.concat(
            [
                self.to_dataframe(f.transform(X[f.col_name]), f.col_name, X.index)
                for f in self.features
            ],
            axis=1,
        )

    @staticmethod
    def to_dataframe(pandas: Union[pd.DataFrame, pd.Series], col_name: str, index):
        if isinstance(pandas, pd.DataFrame):
            return pandas
        elif isinstance(pandas, pd.Series):
            return pd.DataFrame({col_name: pandas})
        elif isinstance(pandas, np.ndarray):
            return pd.DataFrame({col_name: pandas.flatten()}, index=index)
        elif isinstance(pandas, csr_matrix):
            df = pd.DataFrame.sparse.from_spmatrix(pandas, index=index)
            return df.add_prefix("{}_".format(col_name))
        else:
            raise ValueError("unsupported Type")

    @property
    def description(self):
        return "Building Dataset"


class Feature(Base):
    def __init__(self, var_name):
        self.col_name = var_name
        self.var_name = var_name.lower().replace(" ", "_")
        self._pipe: Pipeline = None

    @property
    def description(self):
        return self.col_name

    def fit(self, X, y=None):
        self._pipe.fit(X, y)

    def transform(self, X):
        return self._pipe.transform(X)

    def fit_transform(self, X, y=None):
        self._pipe.fit(X, y)
        return self._pipe.transform(X)
