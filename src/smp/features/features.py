import pandas as pd
from smp import data_dir
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline


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
        return pd.DataFrame(
            {f.col_name: f.transform(X[f.col_name]) for f in self.features}
        )

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

    # TODO: 2 remove
    def __call__(self, loader, train_records, test_records):
        self.convert(loader.train, train_records)
        self.convert(loader.test, test_records)

    # TODO: 2 remove
    def convert(self, data, records):
        raise NotImplemented


class Preprocessor:
    def __init__(self, loader):
        self.loader = loader
        self.train_records = {obs_id: {} for obs_id in loader.train.index}
        self.test_records = {obs_id: {} for obs_id in loader.test.index}

    def preprocess(self, stack):
        for var in stack:
            var(self.loader, self.train_records, self.test_records)

        train = pd.DataFrame.from_dict(self.train_records, orient="index")
        test = pd.DataFrame.from_dict(self.test_records, orient="index")

        return train, test
