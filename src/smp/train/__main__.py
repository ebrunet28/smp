import datetime as dt
import pprint

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from smp import submissions_dir
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
from smp.features.elapsed_time import ProfileCreationTimestamp
from smp.features.image import ProfileImage


def grid_search(model, parameters):

    class PredictMinZero(model):
        _model = model

        def predict(self, *args, **kwargs):
            return np.maximum(self._model.predict(self, *args, **kwargs), 0)

    loader = Loader()

    pipe = Pipeline(
        [
            Dataset(
                [
                    PersonalURL(),
                    ProfileCoverImageStatus(),
                    ProfileVerificationStatus(),
                    IsProfileViewSizeCustomized(),
                    ProfileTextColor(),
                    ProfilePageColor(),
                    ProfileThemeColor(),
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
                    ProfileImage(offset=10, n_components=10),
                ]
            ).to_step(),
            (
                "Grid Search",
                GridSearchCV(
                    PredictMinZero(),
                    param_grid=parameters,
                    n_jobs=None,
                    scoring="neg_mean_squared_log_error",
                ),
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
    pp.pprint(f"Best score: {max(scores['mean_test_score'])}")

    predictions = pipe.predict(loader.test)

    df = pd.DataFrame(
        {"Id": loader.test.index, "Predicted": predictions.round()}, dtype=int
    )

    return df


if __name__ == "__main__":
    parameters = {
        "fit_intercept": (True, False),
        "normalize": (True, False),
    }

    _ = grid_search(LinearRegression, parameters)
    _.to_csv(
        submissions_dir
        / f"submission_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        index=False,
    )
