from smp.features.features import Loader, Preprocessor, Dataset
from smp.features.rgb import ProfilePageColor, ProfileTextColor, ProfileThemeColor
from smp.features.discrete import (
    UtcOffset,
    NumOfDirectMessages,
    NumOfFollowers,
    NumOfPeopleFollowing,
    NumOfStatusUpdates,
)
from smp.features.float import AvgDailyProfileClicks, AvgDailyProfileVisitDuration
from smp.features.onehot import (
    PersonalURL,
    ProfileCoverImageStatus,
    ProfileVerificationStatus,
    IsProfileViewSizeCustomized,
    LocationPublicVisibility,
    UserLanguage,
    UserTimeZone,
    ProfileCategory,
)

from sklearn.pipeline import Pipeline


def main():
    loader = Loader()
    pipe = Pipeline(
        [
            Dataset(
                [
                    ProfileTextColor(),
                    ProfilePageColor(),
                    ProfileThemeColor(),
                    PersonalURL(),
                    ProfileCoverImageStatus(),
                    ProfileVerificationStatus(),
                    # ProfileTextColor(),
                    # ProfilePageColor(),
                    # ProfileThemeColor(),
                    IsProfileViewSizeCustomized(),
                    # UtcOffset,  # TODO:
                    LocationPublicVisibility(),
                    UserLanguage(),
                    UserTimeZone(),
                    NumOfFollowers(),
                    NumOfPeopleFollowing(),
                    NumOfStatusUpdates(),
                    NumOfDirectMessages(),
                    ProfileCategory(),
                    AvgDailyProfileVisitDuration(),
                    AvgDailyProfileClicks(),
                ]
            ).to_step()
        ],
        verbose=True,
    )

    train_data = pipe.fit_transform(loader.train)
    test_data = pipe.transform(loader.test)

    print(train_data.head(10))
    print(train_data.shape)

if __name__ == "__main__":
    main()

