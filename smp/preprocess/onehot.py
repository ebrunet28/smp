from smp.preprocess.base import Feature


class OneHot(Feature):
    def convert(self, data, records):
        raise NotImplementedError


class PersonalURL(OneHot):
    def __init__(self):
        super().__init__("Personal URL")


class ProfileCoverImageStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Cover Image Status")


class ProfileVerificationStatus(OneHot):
    def __init__(self):
        super().__init__("Profile Verification Status")


class IsProfileViewSizeCustomized(OneHot):
    def __init__(self):
        super().__init__("Is Profile View Size Customized?")


class LocationPublicVisibility(OneHot):
    def __init__(self):
        super().__init__("Location Public Visibility")


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
