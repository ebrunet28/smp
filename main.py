import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression

from loader import Loader

loader = Loader()

features = [
    # "User Name",
    # "Personal URL",
    # "Profile Cover Image Status",
    # "Profile Verification Status",
    # "Profile Text Color",
    # "Profile Page Color",
    # "Profile Theme Color",
    # "Is Profile View Size Customized?",
    # "UTC Offset",
    # "Location",
    # "Location Public Visibility",
    # "User Language",
    # "Profile Creation Timestamp",
    # "User Time Zone",
    "Num of Followers",
    "Num of People Following",
    "Num of Status Updates",
    "Num of Direct Messages",
    # "Profile Category",
    # "Avg Daily Profile Visit Duration in seconds",
    # "Avg Daily Profile Clicks",
    # "Profile Image",
]

X = loader.train[features]
y = loader.train["Num of Profile Likes"]

regressor = LinearRegression()
regressor.fit(X=X, y=y)

X_test = loader.test[features]
predictions = regressor.predict(X_test)

df = pd.DataFrame(
    {"Id": X_test.index, "NumProfileLikes": predictions.round()}, dtype=int
)
df.to_csv("submissions/submission_{}".format(dt.datetime.now()), index=False)
