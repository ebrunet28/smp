import pandas as pd


class Loader:
    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.read_csv("data/train.csv")
        self.test: pd.DataFrame = pd.read_csv("data/test.csv")


if __name__ == '__main__':
    loader = Loader()
