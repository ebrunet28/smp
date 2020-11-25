import pandas as pd


class Loader:
    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.read_csv("data/train.csv", index_col="Id")
        self.test: pd.DataFrame = pd.read_csv("data/test.csv", index_col="Id")


if __name__ == '__main__':
    loader = Loader()
