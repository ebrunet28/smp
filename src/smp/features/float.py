from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from smp.features.features import Feature, ToVector


class Float(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="mean"),),
                ("Std Scaler", StandardScaler()),
            ],
            verbose=True,
        )


class AvgDailyProfileVisitDuration(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Visit Duration in seconds")


class AvgDailyProfileClicks(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Clicks")
