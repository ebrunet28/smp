import numpy as np

from smp.features.features import Feature


class OneHot(Feature):
    def convert(self, data, records):
        raise NotImplementedError


class PersonalURL(OneHot):
    def __init__(self):
        super().__init__("Personal URL")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value is np.nan:
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


class ProfileCoverImageStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Cover Image Status")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value == "Set":
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


class ProfileVerificationStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Verification Status")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value == "Verified":
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


class IsProfileViewSizeCustomized(OneHot):
    def __init__(self):
        super().__init__("Is Profile View Size Customized?")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value:
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


class LocationPublicVisibility(OneHot):
    def __init__(self):
        super().__init__("Location Public Visibility")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value.lower() == "enabled":
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


class UserLanguage(OneHot):
    def __init__(self):
        super().__init__("User Language")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value == "en":
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


class UserTimeZone(OneHot):
    def __init__(self):
        super().__init__("User Time Zone")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value == "Eastern Time (US & Canada)":
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


class ProfileCategory(OneHot):
    def __init__(self):
        super().__init__("Profile Category")

    def convert(self, data, records):
        for obs_id, value in data[self.col_name].iteritems():
            if value == "unknown":
                encoded = 0
            else:
                encoded = 1
            records[obs_id].update({self.var_name: encoded})


if __name__ == "__main__":

    from smp.preprocess.base import Loader, Preprocessor

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
