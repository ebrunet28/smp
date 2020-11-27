from smp.features.features import Feature, Base
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np


class HexToRgb(Base):
    def __init__(self, var_name):
        self.var_name = var_name

    def transform(self, X: pd.Series):
        rgb = (self.hex_to_rgb(str(x)) for x in X)
        r, g, b = tuple(zip(*rgb))
        return pd.DataFrame({
            self.var_name + " r": r,
            self.var_name + " g": g,
            self.var_name + " b": b,
        }, index=X.index)

    @property
    def description(self):
        return "Convert hexidecimal to rgb"

    @staticmethod
    def hex_to_rgb(hex_col: str):
        hex_col = hex_col.lstrip("#")
        if (hex_col == "nan") or (len(hex_col) > 6):
            return np.nan, np.nan, np.nan
        if len(hex_col) < 6:
            hex_col = str(6 - len(hex_col)) + hex_col
        hlen = 6
        return tuple(
            int(hex_col[i: i + hlen // 3], 16) / 255 for i in range(0, hlen, hlen // 3)
        )


class RGB(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                HexToRgb(var_name).to_step(),
                ("SimpleImputer", SimpleImputer(strategy="mean"),),
            ],
            verbose=True
        )


class ProfileTextColor(RGB):
    def __init__(self):
        super().__init__("Profile Text Color")


class ProfilePageColor(RGB):
    def __init__(self):
        super().__init__("Profile Page Color")


class ProfileThemeColor(RGB):
    def __init__(self):
        super().__init__("Profile Theme Color")


if __name__ == "__main__":
    from smp.features.features import Loader, Dataset

    loader = Loader()
    pipe = Pipeline(
        [
            Dataset(
                [
                    ProfileTextColor(),
                    ProfilePageColor(),
                    ProfileThemeColor(),
                ]
            ).to_step()
        ],
        verbose=True,
    )

    train_data = pipe.fit_transform(loader.train)
    test_data = pipe.transform(loader.test)

    assert loader.train.shape[0] == train_data.shape[0]
    assert loader.train.shape[0] == train_data.dropna().shape[0]
    assert loader.test.shape[0] == test_data.shape[0]
    assert loader.test.shape[0] == test_data.dropna().shape[0]
