from smp.features.features import Feature, Base
import pandas as pd
from sklearn.pipeline import Pipeline


class ToFloat(Base):
    def transform(self, X: pd.Series):
        return X.astype(float)

    @property
    def description(self):
        return "Convert to float"


class FillNaWithMean(Base):
    def fit(self, X, y=None):
        self._mean = X.mean()

    def transform(self, X: pd.Series):

        return X.fillna(self._mean)

    @property
    def description(self):
        return "Fill blanks with mean of feature"


class Float(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [FillNaWithMean().to_step(), ToFloat().to_step()], verbose=True
        )


class AvgDailyProfileVisitDuration(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Visit Duration in seconds")


class AvgDailyProfileClicks(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Clicks")
