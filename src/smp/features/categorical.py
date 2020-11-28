import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from smp.features.features import Feature, Base


class ToVector(Base):  # TODO delete after using FeatureUnion instead of Dataset
    def transform(self, X):
        return X.values.reshape(-1, 1)

    @property
    def description(self):
        return "Skipping Imputer, transforming to vector"


class LowerCase(Base):
    def transform(self, X):
        return np.char.lower(X.astype(str)).astype(object)

    @property
    def description(self):
        return "Lower case"


class Categorical(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)


class PersonalURL(Categorical):
    def __init__(self):
        super().__init__("Personal URL")
        self._pipe = Pipeline(
            [ToVector().to_step(), ("MissingIndicator", MissingIndicator()),],
            verbose=True,
        )


class ProfileCoverImageStatus(Categorical):
    def __init__(self):
        super().__init__("Profile Cover Image Status")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("imputing", SimpleImputer(strategy="most_frequent"),),
                ("encoding", OneHotEncoder(drop="first"),),  # TODO: make Binary after FeatureUnion
            ],
            verbose=True,
        )


class ProfileVerificationStatus(Categorical):
    def __init__(self):
        super().__init__("Profile Verification Status")
        self._pipe = Pipeline(
            [ToVector().to_step(), ("encoder", OneHotEncoder(drop="first"),),],
            verbose=True,
        )


class IsProfileViewSizeCustomized(Categorical):
    def __init__(self):
        super().__init__("Is Profile View Size Customized?")
        self._pipe = Pipeline([ToVector().to_step(),], verbose=True,)


class LocationPublicVisibility(Categorical):
    def __init__(self):
        super().__init__("Location Public Visibility")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                LowerCase().to_step(),
                (
                    "imputer",
                    SimpleImputer(missing_values="??", strategy="most_frequent"),
                ),
                ("encoder", OneHotEncoder(drop="first"),),  # TODO: make Binary after FeatureUnion
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
                    "encoder",
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
                ("imputer", SimpleImputer(strategy="most_frequent"),),
                (
                    "encoder",
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
                    "imputer",
                    SimpleImputer(missing_values=" ", strategy="most_frequent"),
                ),
                ("encoder", OneHotEncoder(drop="first"),),
            ],
            verbose=True,
        )


if __name__ == "__main__":

    from smp.features.build_features import main

    main()
