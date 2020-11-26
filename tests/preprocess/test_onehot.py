import unittest

from smp.preprocess.base import Loader, Preprocessor
from smp.preprocess.onehot import (
    PersonalURL,
    ProfileCoverImageStatus,
    ProfileVerificationStatus,
    IsProfileViewSizeCustomized,
    LocationPublicVisibility,
    UserLanguage,
    UserTimeZone,
    ProfileCategory,
)


class TestOneHot(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @staticmethod
    def test_preprocess():
        loader = Loader()
        preprocessor = Preprocessor(loader)
        preprocessor.preprocess(
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
