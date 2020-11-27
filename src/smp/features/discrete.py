from smp.features.features import Feature, Base
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ToInt(Base):
    def transform(self, X: pd.Series):
        return X.astype(int).values.reshape(-1, 1)

    @property
    def description(self):
        return "Convert to int"


class Discrete(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)
        self._pipe = Pipeline([ToInt().to_step(), ("Std Scaler", StandardScaler())], verbose=True)


class UtcOffset(Discrete):
    def __init__(self):
        super().__init__("UTC Offset")

    # TODO: resolve UTC offset with Location


class NumOfFollowers(Discrete):
    def __init__(self):
        super().__init__("Num of Followers")


class NumOfPeopleFollowing(Discrete):
    def __init__(self):
        super().__init__("Num of People Following")


class NumOfStatusUpdates(Discrete):
    def __init__(self):
        super().__init__("Num of Status Updates")


class NumOfDirectMessages(Discrete):
    def __init__(self):
        super().__init__("Num of Direct Messages")
