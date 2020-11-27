from smp.train.__main__ import grid_search
from sklearn.linear_model import LinearRegression


def test_grid_search(download_data):
    parameters = {
        "fit_intercept": (True, False),
        "normalize": (True, False),
    }
    grid_search(LinearRegression, parameters)
