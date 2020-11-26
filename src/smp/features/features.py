from smp import data_dir
import pandas as pd


class Feature:
    def __init__(self):
        pass

    def __call__(self, loader, train_records, test_records):
        pass


class Loader:
    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "train.csv", index_col="Id"
        )
        self.test: pd.DataFrame = pd.read_csv(
            data_dir / "raw" / "test.csv", index_col="Id"
        )


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
