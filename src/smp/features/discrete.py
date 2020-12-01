from smp.features.features import Feature, Base, ToVector, ToLog
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class ToInt(Base):
    def transform(self, X):
        return X.astype(int)

    @property
    def description(self):
        return "Convert to int"


class Discrete(Feature):
    def __init__(self, var_name, scaler=StandardScaler):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [ToVector().to_step(), ToInt().to_step(), ToLog().to_step(),
             ("Std Scaler", scaler())], verbose=True
        )


class UtcOffset(Discrete):
    def __init__(self, scaler=StandardScaler):
        super().__init__("UTC Offset")
        self._pipe = Pipeline(
            [ToVector().to_step(), ToInt().to_step(),
             ("Simple Imputer", SimpleImputer(strategy="mean")),
             ("Std Scaler", scaler())], verbose=True
        )
    # TODO: resolve UTC offset with Location


class NumOfFollowers(Discrete):
    def __init__(self, scaler=StandardScaler):
        super().__init__("Num of Followers", scaler)


class NumOfPeopleFollowing(Discrete):
    def __init__(self, scaler=StandardScaler):
        super().__init__("Num of People Following", scaler)


class NumOfStatusUpdates(Discrete):
    def __init__(self, scaler=StandardScaler):
        super().__init__("Num of Status Updates", scaler)


class NumOfDirectMessages(Discrete):
    def __init__(self, scaler=StandardScaler):
        super().__init__("Num of Direct Messages", scaler)
