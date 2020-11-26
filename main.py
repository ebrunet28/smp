import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression

from smp.preprocess.base import Loader, Preprocessor
from smp.preprocess.float import (
    NumOfFollowers,
    NumOfPeopleFollowing,
    NumOfStatusUpdates,
    NumOfDirectMessages,
)
from smp.preprocess.rgb import (
    ProfileTextColor,
    ProfilePageColor,
    ProfileThemeColor,
)
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


def predict():

    loader = Loader()
    preprocessor = Preprocessor(loader)
    train_data, test_data = preprocessor.preprocess(
        [
            ProfileTextColor(),
            ProfilePageColor(),
            ProfileThemeColor(),
            NumOfFollowers(),
            NumOfPeopleFollowing(),
            NumOfStatusUpdates(),
            NumOfDirectMessages(),
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

    X = train_data
    y = loader.train["Num of Profile Likes"]

    regressor = LinearRegression()
    regressor.fit(X=X, y=y)

    X_test = test_data
    predictions = regressor.predict(X_test)

    df = pd.DataFrame({"Id": X_test.index, "Predicted": predictions.round()}, dtype=int)

    return df


if __name__ == "__main__":

    _ = predict()
    _.to_csv("submissions/submission_{}".format(dt.datetime.now()), index=False)
