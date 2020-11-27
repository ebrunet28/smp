from smp.features.features import Feature
import numpy as np


class RGB(Feature):

    def __init__(self, var_name):
        super().__init__(var_name)

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

    def convert(self, data, records):
        rgb = np.empty((data.shape[0], 3))
        for i, (obs_id, hex_code) in enumerate(data[self.col_name].iteritems()):
            rgb[i, :] = self.hex_to_rgb(str(hex_code))

        # records[obs_id].update(
        #     {
        #         self.var_name + "_r": r,
        #         self.var_name + "_g": g,
        #         self.var_name + "_b": b,
        #     }
        # )
        # import pandas as pd
        # data[self.col_name].apply(lambda x: len(str(x).lstrip("#|.|+"))).sort_values().describe()
        # data[self.col_name].apply(lambda x: len(''.join(e for e in str(x) if e.isnumeric()))).sort_values().describe()


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
    from smp.features.features import Loader, Preprocessor
    loader = Loader()
    preprocessor = Preprocessor(loader)
    train_data, test_data = preprocessor.preprocess(
        [
            ProfileTextColor(),
            ProfilePageColor(),
            ProfileThemeColor(),
        ]
    )

    # a = np.array(
    #  [[ 0.93230948,  np.nan    ,  0.47773439,  0.76998063],
    #   [ 0.94460779,  0.87882456,  0.79615838,  0.56282885],
    #   [ 0.94272934,  0.48615268,  0.06196785,  np.nan],
    #   [ 0.64940216,  0.74414127,  np.nan    ,  np.nan]]
    # )
    #  #Obtain mean of columns as you need, nanmean is convenient.
    #  col_mean = np.nanmean(a, axis=0)
    #  print(col_mean)
    #
    #  #Find indices that you need to replace
    #  inds = np.where(np.isnan(a))
    #
    #  #Place column means in the indices. Align the arrays using take
    #  a[inds] = np.take(col_mean, inds[1])