import pandas as pd
from smp import data_dir
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse.csr import csr_matrix
from typing import Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Loader:
    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "train.csv", index_col="Id"
        )
        self.test: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "test.csv", index_col="Id"
        )


class Base(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)
    #     return self.transform(X)

    @abstractmethod
    def transform(self, X):
        return X

    @property
    @abstractmethod
    def description(self):
        return "Describe your class"

    def to_step(self):
        return self.description, self


class Dataset(FeatureUnion):
    def __init__(self, transformer_list, *, n_jobs=None,
                 transformer_weights=None, verbose=False):
        super().__init__([f.to_step() for f in transformer_list], n_jobs=n_jobs,
                 transformer_weights=transformer_weights, verbose=verbose)


    # def fit(self, X, y=None):
    #     for feature in self.features:
    #         feature.fit(X[feature.col_name])

    # def transform(self, X):
    #     return pd.concat(
    #         [
    #             self.to_dataframe(f.transform(X[f.col_name]), f.col_name, X.index)
    #             for f in self.features
    #         ],
    #         axis=1,
    #     )
    #
    # @staticmethod
    # def to_dataframe(pandas: Union[pd.DataFrame, pd.Series], col_name: str, index):
    #     if isinstance(pandas, pd.DataFrame):
    #         return pandas
    #     elif isinstance(pandas, pd.Series):
    #         return pd.DataFrame({col_name: pandas})
    #     elif isinstance(pandas, np.ndarray):
    #         if pandas.shape[-1] == 1:
    #             return pd.DataFrame({col_name: pandas.flatten()}, index=index)
    #         else:
    #             return pd.DataFrame(
    #                 {f"{col_name}_{i}": col for i, col in enumerate(pandas.T)},
    #                 index=index,
    #             )
    #     elif isinstance(pandas, csr_matrix):
    #         df = pd.DataFrame.sparse.from_spmatrix(pandas, index=index)
    #         return df.add_prefix("{}_".format(col_name))
    #     else:
    #         raise ValueError("unsupported Type")

    @property
    def description(self):
        return "Building Dataset"

    def to_step(self):
        return self.description, self


class Feature(Base):
    def __init__(self, var_name):
        self.col_name = var_name
        self.var_name = var_name.lower().replace(" ", "_")
        self._pipe: Pipeline = None

    @property
    def description(self):
        return self.col_name

    def fit(self, X, y=None):
        self._pipe.fit(X[self.col_name], y)

    def transform(self, X):
        return self._pipe.transform(X[self.col_name])

    def fit_transform(self, X, y=None):
        self._pipe.fit(X[self.col_name], y)
        return self._pipe.transform(X[self.col_name])
