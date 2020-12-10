from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from smp.features.features import Feature, ToVector, ToLog


class Float(Feature):
    def __init__(self, var_name, scaler=StandardScaler):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ToLog().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="mean"),),
                ("Scaler", scaler()),
            ],
            verbose=True,
        )


class AvgDailyProfileVisitDuration(Float):
    def __init__(self, scaler=RobustScaler):
        super().__init__("Avg Daily Profile Visit Duration in seconds", scaler)


class AvgDailyProfileClicks(Float):
    def __init__(self, scaler=StandardScaler):
        super().__init__("Avg Daily Profile Clicks", scaler)
