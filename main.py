import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression

from smp.preprocess.base import Loader, Preprocessor
from smp.preprocess.rgb import (
    ProfileTextColor,
    ProfilePageColor,
    ProfileThemeColor,
)
from smp.preprocess.float import (
    UtcOffset,
    NumOfFollowers,
    NumOfPeopleFollowing,
    NumOfStatusUpdates,
    NumOfDirectMessages,
    AvgDailyProfileVisitDuration,
    AvgDailyProfileClicks,
)


def predict():

    loader = Loader()
    preprocessor = Preprocessor(loader)
    train_data, test_data = preprocessor.preprocess(
        [
            ProfileTextColor(),
            ProfilePageColor(),
            ProfileThemeColor(),
            UtcOffset(),
            NumOfFollowers(),
            NumOfPeopleFollowing(),
            NumOfStatusUpdates(),
            NumOfDirectMessages(),
            AvgDailyProfileVisitDuration(),
            AvgDailyProfileClicks(),
        ]
    )

    selected = [
        "num_of_followers",
        "num_of_people_following",
        "num_of_status_updates",
        "num_of_direct_messages",
    ]

    X = train_data[selected]
    y = loader.train["Num of Profile Likes"]

    regressor = LinearRegression()
    regressor.fit(X=X, y=y)

    X_test = test_data[selected]
    predictions = regressor.predict(X_test)

    df = pd.DataFrame({"Id": X_test.index, "Predicted": predictions.round()}, dtype=int)

    return df


if __name__ == "__main__":

    _ = predict()
    _.to_csv("submissions/submission_{}".format(dt.datetime.now()), index=False)
