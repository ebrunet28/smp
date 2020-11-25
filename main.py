import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression

from preprocess import Loader, Preprocessor
from preprocess import RGB, Float


def predict():

    loader = Loader()
    preprocessor = Preprocessor(loader)
    train_data, test_data = preprocessor.preprocess(
        [
            RGB("Profile Text Color"),
            RGB("Profile Page Color"),
            RGB("Profile Theme Color"),
            Float("UTC Offset"),
            Float("Num of Followers"),
            Float("Num of People Following"),
            Float("Num of Status Updates"),
            Float("Num of Direct Messages"),
            Float("Avg Daily Profile Visit Duration in seconds"),
            Float("Avg Daily Profile Clicks"),
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

    df = pd.DataFrame(
        {"Id": X_test.index, "NumProfileLikes": predictions.round()}, dtype=int
    )

    return df


if __name__ == "__main__":

    _ = predict()
    _.to_csv("submissions/submission_{}".format(dt.datetime.now()), index=False)
