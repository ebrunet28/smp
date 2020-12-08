import pandas as pd
from smp import data_dir
from abc import abstractmethod
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


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


class ToDense(Base):
    """
    Some algorithms does not accept csr_matrix. We need to convert the dataset to dense
    """

    def transform(self, X):
        try:
            return X.toarray()
        except:
            return X

    @property
    def description(self):
        return "To Dense"


class Feature(Base):
    def __init__(self, var_name):
        self.var_name = var_name
        self._pipe: Pipeline = None

    @property
    def description(self):
        if isinstance(self.var_name, list):
            return ' '.join(map(str, self.var_name))
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


class ToLog(Base):
    def transform(self, X):
        return np.log(1 + X)

    @property
    def description(self):
        return "Convert to log(1+x)"


class CapStd(Base):
    def transform(self, X):
        if isinstance(X, pd.Series):
            mu = X.mean()
            std = X.std()
            X = np.clip(X, mu - 3 * std, mu + 3 * std)
        elif isinstance(X, pd.DataFrame):
            for col in X.columns:
                mu = X[col].mean()
                std = X[col].std()
                X[col] = np.clip(X[col], mu - 3 * std, mu + 3 * std)
        else:
            for i in range(0, X.shape[1]):
                mu = X[:, i].mean()
                std = X[:, i].std()
                X[:, i] = np.clip(X[:, i], mu - 3 * std, mu + 3 * std)

        return X


class CapIQR(Base):
    def transform(self, X):
        if isinstance(X, pd.Series):
            percentiles = X.quantile([0.01, 0.99]).values
            X = np.clip(X, percentiles[0], percentiles[1])
        elif isinstance(X, pd.DataFrame):
            for col in X.columns:
                percentiles = X[col].quantile([0.01, 0.99]).values
                X[col] = np.clip(X[col], percentiles[0], percentiles[1])
        else:
            for i in range(0, X.shape[1]):
                percentiles = np.quantile(X[:, i], [0.01, 0.99])
                X[:, i] = np.clip(X[:, i], percentiles[0], percentiles[1])
        return X

    @property
    def description(self):
        return "Cap  values"
