import datetime as dt
import pprint

import pandas as pd
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
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
    parameters = {
        "n_neighbors": range(4, 500, 25),
        "weights": ("uniform", "distance"),
        "leaf_size": range(5, 100, 10),
        "p": range(1, 6),
    }
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
            (
                "Grid Search",
                GridSearchCV(KNeighborsRegressor(), param_grid=parameters, n_jobs=-1),
            ),
        ],
        verbose=True,
    )

    X_train = loader.train.iloc[:, :-1]
    y_train = loader.train.iloc[:, -1]
    pipe.fit(X_train, y_train)
    scores = pipe.steps[-1][-1].cv_results_
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(f"Mean scores: {scores['mean_test_score']}")
    pp.pprint(f"Best params{pipe.steps[-1][-1].best_params_}")

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
