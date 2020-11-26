from smp.features.features import Feature


class RGB(Feature):
    def convert(self, data, records):
        for obs_id, hex_code in data[self.col_name].iteritems():
            r, g, b = self.hex_to_rgb(str(hex_code))
            records[obs_id].update(
                {
                    self.var_name + "_r": r,
                    self.var_name + "_g": g,
                    self.var_name + "_b": b,
                }
            )

    @staticmethod
    def hex_to_rgb(hex_col):
        if hex_col == "nan":
            return -1, -1, -1
        hex_col = hex_col.lstrip("#")
        hlen = len(hex_col)
        return tuple(
            int(hex_col[i : i + hlen // 3], 16) / 255 for i in range(0, hlen, hlen // 3)
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
