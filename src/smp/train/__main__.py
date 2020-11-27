import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression
from smp import submissions_dir
from smp.features.features import Loader, Preprocessor
from smp.features.discrete import (
    NumOfFollowers,
    NumOfPeopleFollowing,
    NumOfStatusUpdates,
    NumOfDirectMessages,
)
from smp.features.rgb import (
    ProfileTextColor,
    ProfilePageColor,
    ProfileThemeColor,
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
    _.to_csv(
        submissions_dir
        / f"submission_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
        index=False,
    )
