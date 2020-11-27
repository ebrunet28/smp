import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder
from smp.features.features import Feature


class OneHot(Feature):
    def __call__(self, loader, train_records, test_records):
        train, test = self._impute(
            loader.train[self.col_name].values.reshape(-1, 1),
            loader.test[self.col_name].values.reshape(-1, 1),
        )
        train, test = self._preprocess(train, test)
        self.convert(
            pd.DataFrame.sparse.from_spmatrix(train, index=loader.train.index),
            train_records,
        )
        self.convert(
            pd.DataFrame.sparse.from_spmatrix(test, index=loader.test.index),
            test_records,
        )

    def _impute(
        self, train, test,
    ):

        self._imputer = SimpleImputer(strategy="most_frequent")
        self._imputer.fit(train)

        train = self._imputer.transform(train)
        test = self._imputer.transform(test)

        return train, test

    def _preprocess(
        self, train, test,
    ):

        self._encoder = OneHotEncoder(handle_unknown="ignore")
        self._encoder.fit(train)

        train = self._encoder.transform(train)
        test = self._encoder.transform(test)

        return train, test

    def convert(self, data, records):
        for obs_id, row in data.iterrows():
            records[obs_id].update({self.var_name: row.to_dict()})


class PersonalURL(OneHot):
    def __init__(self):
        super().__init__("Personal URL")

    def _impute(
        self, train, test,
    ):

        self._imputer = MissingIndicator()
        self._imputer.fit(train)

        train = self._imputer.transform(train)
        test = self._imputer.transform(test)

        return train, test


class ProfileCoverImageStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Cover Image Status")


class ProfileVerificationStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Verification Status")


class IsProfileViewSizeCustomized(OneHot):
    def __init__(self):
        super().__init__("Is Profile View Size Customized?")

    def _impute(
        self, train, test,
    ):

        return train, test


class LocationPublicVisibility(OneHot):
    def __init__(self):
        super().__init__("Location Public Visibility")

    def _impute(
        self, train, test,
    ):

        train = np.char.lower(train.astype(str)).astype(object)
        test = np.char.lower(test.astype(str)).astype(object)

        self._imputer = SimpleImputer(missing_values="??", strategy="most_frequent",)
        self._imputer.fit(train)

        train = self._imputer.transform(train)
        test = self._imputer.transform(test)

        return train, test


class UserLanguage(OneHot):
    def __init__(self):
        super().__init__("User Language")


class UserTimeZone(OneHot):
    def __init__(self):
        super().__init__("User Time Zone")


class ProfileCategory(OneHot):
    def __init__(self):
        super().__init__("Profile Category")


if __name__ == "__main__":

    from smp.features.features import Loader, Preprocessor

    loader = Loader()
    preprocessor = Preprocessor(loader)
    train_data, test_data = preprocessor.preprocess(
        [
            PersonalURL(),
            ProfileCoverImageStatus(),
            ProfileVerificationStatus(),
            IsProfileViewSizeCustomized(),
            LocationPublicVisibility(),
            UserLanguage(),
            UserTimeZone(),
            ProfileCategory(),
        ]
    )
