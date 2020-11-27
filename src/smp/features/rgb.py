from smp.features.features import Feature, Base
import pandas as pd
from sklearn.pipeline import Pipeline


class HexToRgb(Base):
    def __init__(self, var_name):
        self.var_name = var_name

    def transform(self, X: pd.Series):
        rgb = (self.hex_to_rgb(str(x)) for x in X)
        r, g, b = tuple(zip(*rgb))
        return pd.DataFrame({
            self.var_name + "_r": r,
            self.var_name + "_g": g,
            self.var_name + "_b": b,
        }, index=X.index)

    @property
    def description(self):
        return "Convert hexidecimal to rgb"

    @staticmethod
    def hex_to_rgb(hex_col):
        if hex_col == "nan":
            return -1, -1, -1
        hex_col = hex_col.lstrip("#")
        hlen = len(hex_col)
        return tuple(
            int(hex_col[i : i + hlen // 3], 16) / 255 for i in range(0, hlen, hlen // 3)
        )


class RGB(Feature):
    def __init__(self, var_name):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [HexToRgb(var_name).to_step()], verbose=True
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
