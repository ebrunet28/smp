from smp.features.features import Feature, Base, ToLog
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TimeDelta(Base):
    def __init__(self, var_name):
        self.var_name = var_name

    def transform(self, X: pd.Series):
        dates = pd.to_datetime(X).dt.date
        most_recent = dates.max()
        return dates.apply(lambda x: (most_recent - x).days).to_frame()

    @property
    def description(self):
        return "Convert profile timestamp to int"


class ElapsedTime(Feature):
    def __init__(self, var_name, scaler=StandardScaler):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                TimeDelta(var_name).to_step(),
                ToLog().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="mean"),),
                ("Std Scaler", scaler()),
            ],
            verbose=True
        )


class ProfileCreationTimestamp(ElapsedTime):
    def __init__(self, scaler=StandardScaler):
        super().__init__("Profile Creation Timestamp", scaler)


if __name__ == "__main__":
    from smp.features.features import Loader, Dataset
    loader = Loader()
    pipe = Pipeline(
        [
            Dataset(
                [
                    ProfileCreationTimestamp(),
                ]
            ).to_step()
        ],
        verbose=True,
    )

    train_data = pipe.fit_transform(loader.train)
    test_data = pipe.transform(loader.test)
    assert loader.train.shape[0] == train_data.shape[0]
    assert loader.train.shape[0] == train_data.dropna().shape[0]
    assert loader.test.shape[0] == test_data.shape[0]
    assert loader.test.shape[0] == test_data.dropna().shape[0]
