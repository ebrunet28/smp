import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from smp.features.features import Feature, Base


class ToVector(Base):
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


class OneHot(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)


class PersonalURL(OneHot):
    def __init__(self):
        super().__init__("Personal URL")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("MissingIndicator", MissingIndicator()),
                ("OneHotEncoder", OneHotEncoder()),
            ],
            verbose=True,
        )


class ProfileCoverImageStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Cover Image Status")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="most_frequent"),),
                ("OneHotEncoder", OneHotEncoder()),
            ],
            verbose=True,
        )


class ProfileVerificationStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Verification Status")
        self._pipe = Pipeline(
            [ToVector().to_step(), ("OneHotEncoder", OneHotEncoder()),], verbose=True,
        )


class IsProfileViewSizeCustomized(OneHot):
    def __init__(self):
        super().__init__("Is Profile View Size Customized?")
        self._pipe = Pipeline(
            [ToVector().to_step(), ("OneHotEncoder", OneHotEncoder()),], verbose=True,
        )


class LocationPublicVisibility(OneHot):
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
                ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore")),
            ],
            verbose=True,
        )


class UserLanguage(OneHot):
    def __init__(self):
        super().__init__("User Language")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore")),
            ],
            verbose=True,
        )


class UserTimeZone(OneHot):
    def __init__(self):
        super().__init__("User Time Zone")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="most_frequent"),),
                ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore")),
            ],
            verbose=True,
        )


class ProfileCategory(OneHot):
    def __init__(self):
        super().__init__("Profile Category")
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                (
                    "SimpleImputer",
                    SimpleImputer(missing_values=" ", strategy="most_frequent"),
                ),
                ("OneHotEncoder", OneHotEncoder()),
            ],
            verbose=True,
        )


if __name__ == "__main__":

    from smp.features.build_features import main

    main()
