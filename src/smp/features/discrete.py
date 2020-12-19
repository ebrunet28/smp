import numpy as np

from smp.features.features import Feature, Base, ToVector, ToLog
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


class ToInt(Base):
    def transform(self, X):
        return X.astype(int)

    @property
    def description(self):
        return "Convert to int"


class ToAbs(Base):
    def transform(self, X):
        return np.abs(X)

    @property
    def description(self):
        return "Convert to absolute"


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
             ("Simple Imputer", SimpleImputer(strategy="most_frequent")),
             ("OneHotEncoder", OneHotEncoder())]
        )


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


class CustomFeature(Discrete):
    def __init__(self, scaler=StandardScaler):
        super().__init__(["Num of People Following", "Num of Status Updates"], scaler)

    @property
    def description(self):
        return '+'.join(self.var_name)

    def fit(self, X, y=None):
        X = (np.log(X[self.var_name[0]]+1)*np.log(X[self.var_name[1]]+1))**(1/2)
        self._pipe.fit(X, y)
        return self

    def transform(self, X):
        X = (np.log(X[self.var_name[0]] + 1) * np.log(X[self.var_name[1]] + 1)) ** (
                    1 / 2)
        return self._pipe.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        X = (np.log(X[self.var_name[0]] + 1) * np.log(X[self.var_name[1]] + 1)) ** (
                    1 / 2)
        self._pipe.fit(X, y)
        return self._pipe.transform(X)