from .features import Feature


class RGB(Feature):
    def __init__(self, var_name):
        super().__init__()
        self.col_name = var_name
        self.var_name = var_name.lower().replace(" ", "_")

    def __call__(self, loader, train_records, test_records):
        self.convert(loader.train, train_records)
        self.convert(loader.test, test_records)

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

    def hex_to_rgb(self, hex_col):
        if hex_col == "nan":
            return -1, -1, -1
        hex_col = hex_col.lstrip("#")
        hlen = len(hex_col)
        return tuple(
            int(hex_col[i : i + hlen // 3], 16) / 255 for i in range(0, hlen, hlen // 3)
        )