from smp.features.features import Loader, Dataset
from smp.features.rgb import ProfilePageColor, ProfileTextColor, ProfileThemeColor
from smp.features.discrete import (
    UtcOffset,
    NumOfDirectMessages,
    NumOfFollowers,
    NumOfPeopleFollowing,
    NumOfStatusUpdates,
)
from smp.features.float import AvgDailyProfileClicks, AvgDailyProfileVisitDuration
from smp.features.categorical import (
    PersonalURL,
    ProfileCoverImageStatus,
    ProfileVerificationStatus,
    IsProfileViewSizeCustomized,
    LocationPublicVisibility,
    UserLanguage,
    UserTimeZone,
    ProfileCategory,
)
from smp.features.elapsed_time import ProfileCreationTimestamp

from sklearn.pipeline import Pipeline


def main():
    loader = Loader()
    pipe = Pipeline(
        [
            Dataset(
                [
                    PersonalURL(),
                    ProfileCoverImageStatus(),
                    ProfileVerificationStatus(),
                    ProfileTextColor(),
                    ProfilePageColor(),
                    ProfileThemeColor(),
                    IsProfileViewSizeCustomized(),
                    # UtcOffset,  # TODO:
                    LocationPublicVisibility(),
                    UserLanguage(),
                    ProfileCreationTimestamp(),
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

