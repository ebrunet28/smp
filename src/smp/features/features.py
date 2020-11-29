import pandas as pd
from smp import data_dir
from abc import abstractmethod
from sklearn.pipeline import Pipeline, FeatureUnion
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
    def __init__(self, transformer_list, *, n_jobs=-1,
                 transformer_weights=None, verbose=False):
        super().__init__([f.to_step() for f in transformer_list], n_jobs=n_jobs,
                         transformer_weights=transformer_weights, verbose=verbose)

    @property
    def description(self):
        return "Building Dataset"

    def to_step(self):
        return self.description, self


class Feature(Base):
    def __init__(self, var_name):
        self.var_name = var_name
        self._pipe: Pipeline = None

    @property
    def description(self):
        return self.var_name

    def fit(self, X, y=None):
        self._pipe.fit(X[self.var_name], y)
        return self

    def transform(self, X):
        return self._pipe.transform(X[self.var_name])

    def fit_transform(self, X, y=None, **fit_params):
        self._pipe.fit(X[self.var_name], y)
        return self._pipe.transform(X[self.var_name])


class ToVector(Base): # TODO/32: remove
    def transform(self, X: pd.Series):
        return X.values.reshape(-1, 1)

    @property
    def description(self):
        return "ToVector"
