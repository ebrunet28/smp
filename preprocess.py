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


class Float(Feature):
    def convert(self, data, records):
        for obs_id, val in data[self.col_name].iteritems():
            records[obs_id].update({self.var_name: float(val)})


class RGB(Feature):
    def convert(self, data, records):
        for obs_id, hex_code in data[self.col_name].iteritems():
            r, g, b = self.hex_to_rgb(str(hex_code))
            records[obs_id].update(
                {
                    self.var_name + "_r": r,
                    self.var_name + "_g": g,
                    self.var_name + "_b": b,
                }
            )

    @staticmethod
    def hex_to_rgb(hex_col):
        if hex_col == "nan":
            return -1, -1, -1
        hex_col = hex_col.lstrip("#")
        hlen = len(hex_col)
        return tuple(
            int(hex_col[i : i + hlen // 3], 16) / 255 for i in range(0, hlen, hlen // 3)
        )


class ProfileTextColor(RGB):
    def __init__(self):
        super().__init__("Profile Text Color")


class ProfilePageColor(RGB):
    def __init__(self):
        super().__init__("Profile Page Color")


class ProfileThemeColor(RGB):
    def __init__(self):
        super().__init__("Profile Theme Color")


class UtcOffset(Float):
    def __init__(self):
        super().__init__("UTC Offset")


class NumOfFollowers(Float):
    def __init__(self):
        super().__init__("Num of Followers")


class NumOfPeopleFollowing(Float):
    def __init__(self):
        super().__init__("Num of People Following")


class NumOfStatusUpdates(Float):
    def __init__(self):
        super().__init__("Num of Status Updates")


class NumOfDirectMessages(Float):
    def __init__(self):
        super().__init__("Num of Direct Messages")


class AvgDailyProfileVisitDuration(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Visit Duration in seconds")


class AvgDailyProfileClicks(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Clicks")


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
