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
from smp.features.image import ProfileImage
from smp.features.features import Base


class ToDense(Base):
    """
    Some algorithms does not accept csr_matrix. We need to convert the dataset to dense
    """
    def transform(self, X):
        return X.toarray()

    @property
    def description(self):
        return "To Dense"


def grid_search(model, parameters):

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
            #ToDense().to_step(),
            (
                "Grid Search",
                GridSearchCV(
                    model(),
                    param_grid=parameters,
                    n_jobs=-1,
                    scoring="neg_mean_squared_error",
                ),
            ),
        ],
        verbose=True,
    )

    X_train = loader.train.iloc[:, :-1]
    y_train = np.log(loader.train.iloc[:, -1] + 1)
    pipe.fit(X_train, y_train)
    scores = pipe.steps[-1][-1].cv_results_
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(f"Mean scores: {(-scores['mean_test_score'])**(1/2)}")
    pp.pprint(f"Best params{pipe.steps[-1][-1].best_params_}")
    pp.pprint(f"Best score: {min((-scores['mean_test_score'])**(1/2))}")

    predictions = pipe.predict(loader.test)

    df = pd.DataFrame(
        {"Id": loader.test.index, "Predicted": (np.exp(predictions)).round().astype(int)},
        dtype=int,
    )

    return df


if __name__ == "__main__":
    parameters = {
        "fit_intercept": (True, False),
        "normalize": (True, False),
    }
    # parameters ={
    #     "n_neighbors": [4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400],
    #     "weights": ("uniform", "distance"),
    #     "leaf_size": range(1, 10, 2),
    #     "p": range(1, 3),
    # }

    _ = grid_search(LinearRegression, parameters)
    _.to_csv(
        submissions_dir
        / f"submission_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        index=False,
    )
