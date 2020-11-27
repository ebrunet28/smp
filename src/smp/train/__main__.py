import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression
from smp import submissions_dir
from smp.features.features import Loader
from smp.features.discrete import (
    NumOfFollowers,
    NumOfPeopleFollowing,
    NumOfStatusUpdates,
    NumOfDirectMessages,
)
from smp.features.rgb import ProfileThemeColor, ProfileTextColor, ProfilePageColor
from smp.features.float import AvgDailyProfileVisitDuration, AvgDailyProfileClicks
from sklearn.pipeline import Pipeline
from smp.features.features import Dataset
from sklearn.model_selection import train_test_split


def predict():

    loader = Loader()
    pipe = Pipeline(
        [
            Dataset(
                [
                    ProfileTextColor(),
                    ProfilePageColor(),
                    ProfileThemeColor(),
                    # UtcOffset,  # TODO:
                    NumOfFollowers(),
                    NumOfPeopleFollowing(),
                    NumOfStatusUpdates(),
                    NumOfDirectMessages(),
                    AvgDailyProfileVisitDuration(),
                    AvgDailyProfileClicks(),
                ]
            ).to_step(),
            ("Linear Regressor", LinearRegression()),
        ],
        verbose=True,
    )

    X_train = loader.train.iloc[:, :-1]
    y_train = loader.train.iloc[:, -1]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, random_state=0
    )
    pipe.fit(X_train, y_train)
    valid_score = pipe.score(X_valid, y_valid)

    print(valid_score)

    predictions = pipe.predict(loader.test)

    df = pd.DataFrame(
        {"Id": loader.test.index, "Predicted": predictions.round()}, dtype=int
    )

    return df


if __name__ == "__main__":

    _ = predict()
    _.to_csv(
        submissions_dir
        / f"submission_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        index=False,
    )
