import pandas as pd


class Loader:
    def __init__(self) -> None:
        self.train: pd.DataFrame = pd.read_csv("data/train.csv", index_col="Id")
        self.test: pd.DataFrame = pd.read_csv("data/test.csv", index_col="Id")


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


if __name__ == "__main__":

    from smp.preprocess.rgb import RGB
    from smp.preprocess.float import Float

    loader = Loader()
    preprocessor = Preprocessor(loader)
    train_data, test_data = preprocessor.preprocess(
        [
            RGB("Profile Text Color"),
            RGB("Profile Page Color"),
            RGB("Profile Theme Color"),
            Float("UTC Offset"),
            Float("Num of Followers"),
            Float("Num of People Following"),
            Float("Num of Status Updates"),
            Float("Num of Direct Messages"),
            Float("Avg Daily Profile Visit Duration in seconds"),
            Float("Avg Daily Profile Clicks"),
            # Float("Num of Profile Likes"),
        ]
    )

    print(train_data.head(10))
    print(train_data.shape)
