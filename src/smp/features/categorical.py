import numpy as np
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from smp.features.features import Feature, Base, ToVector


class LowerCase(Base):
    def transform(self, X):
        return np.char.lower(X.astype(str)).astype(object)

    @property
    def description(self):
        return "LowerCase"


class MyLabelBinarizer(
    TransformerMixin
):  # https://stackoverflow.com/questions/46162855/fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize
    def __init__(self, *args, **kwargs):
        self._encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self._encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self._encoder.transform(x)


class Categorical(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)


class ProfileCoverImageStatus(Categorical):
    def __init__(self):
        super().__init__("Profile Cover Image Status")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="most_frequent"),),
                ("MyLabelBinarizer", MyLabelBinarizer(),),
            ],
            verbose=True,
        )


class ProfileVerificationStatus(Categorical):
    def __init__(self):
        super().__init__("Profile Verification Status")
        self._pipe = Pipeline(
            [ToVector().to_step(), ("OneHotEncoder", OneHotEncoder(drop="first"),),],
            verbose=True,
        )


class LocationPublicVisibility(Categorical):
    def __init__(self):
        super().__init__("Location Public Visibility")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                LowerCase().to_step(),
                (
                    "SimpleImputer",
                    SimpleImputer(missing_values="??", strategy="most_frequent"),
                ),
                ("MyLabelBinarizer", MyLabelBinarizer(),),
            ],
            verbose=True,
        )


class UserLanguage(Categorical):
    def __init__(self):
        super().__init__("User Language")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                (
                    "OneHotEncoder",
                    OneHotEncoder(
                        categories=[  # IMPORTANT: do not drop when forcing categories
                            ["en", "es"]
                        ],
                        handle_unknown="ignore",
                    ),
                ),
            ],
            verbose=True,
        )


class UserTimeZone(Categorical):
    def __init__(self):
        super().__init__("User Time Zone")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="most_frequent"),),
                (
                    "OneHotEncoder",
                    OneHotEncoder(
                        categories=[  # IMPORTANT: do not drop when forcing categories
                            [
                                "Eastern Time (US & Canada)",
                                "Pacific Time (US & Canada)",
                                "Central Time (US & Canada)",
                            ]
                        ],
                        handle_unknown="ignore",
                    ),
                ),
            ],
            verbose=True,
        )


class ProfileCategory(Categorical):
    def __init__(self):
        super().__init__("Profile Category")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                (
                    "SimpleImputer",
                    SimpleImputer(missing_values=" ", strategy="most_frequent"),
                ),
                ("OneHotEncoder", OneHotEncoder(drop="first"),),
            ],
            verbose=True,
        )


if __name__ == "__main__":

    from smp.features.build_features import main

    main()
