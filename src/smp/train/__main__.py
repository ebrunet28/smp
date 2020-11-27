import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from smp import submissions_dir
from smp.features.discrete import (
    NumOfFollowers,
    NumOfPeopleFollowing,
    NumOfStatusUpdates,
    NumOfDirectMessages,
)
from smp.features.features import Dataset
from smp.features.features import Loader
from smp.features.float import AvgDailyProfileVisitDuration, AvgDailyProfileClicks
from smp.features.rgb import ProfileThemeColor, ProfileTextColor, ProfilePageColor


def linear_regressor():

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
            ("Linear Regressor", LinearRegression(),),
        ],
        verbose=True,
    )

    X_train = loader.train.iloc[:, :-1]
    y_train = loader.train.iloc[:, -1]

    scores = cross_val_score(pipe, X_train, y_train, cv=5)

    print(f"\nCross-validation scores: {scores}\n")

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(loader.test)

    df = pd.DataFrame(
        {"Id": loader.test.index, "Predicted": predictions.round()}, dtype=int
    )

    return df


if __name__ == "__main__":

    _ = linear_regressor()
    _.to_csv(
        submissions_dir
        / f"submission_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        index=False,
    )
