import pandas as pd
from smp import data_dir


class Loader:
    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "train.csv", index_col="Id"
        )
        self.test: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "test.csv", index_col="Id"
        )


class Feature:
    def __init__(self, var_name):
        self.col_name = var_name
        self.var_name = var_name.lower().replace(" ", "_")

    def __call__(self, loader, train_records, test_records):
        self.convert(loader.train, train_records)
        self.convert(loader.test, test_records)

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
