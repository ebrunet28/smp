from sklearn.impute import MissingIndicator
from sklearn.pipeline import Pipeline
from smp.features.features import Feature, ToVector


class Boolean(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)


class PersonalURL(Boolean):
    def __init__(self):
        super().__init__("Personal URL")
        self._pipe = Pipeline(
            [ToVector().to_step(), ("MissingIndicator", MissingIndicator()),],
            verbose=True,
        )


class IsProfileViewSizeCustomized(Boolean):
    def __init__(self):
        super().__init__("Is Profile View Size Customized?")
        self._pipe = Pipeline([ToVector().to_step(),], verbose=True,)


class Location(Boolean):
    def __init__(self):
        super().__init__("Location")
        self._pipe = Pipeline(
            [ToVector().to_step(), ("MissingIndicator", MissingIndicator()),],
            verbose=True,
        )
