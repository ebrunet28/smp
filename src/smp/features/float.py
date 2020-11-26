from .features import Feature

class Float(Feature):
    def __init__(self, var_name):
        super().__init__()
        self.col_name = var_name
        self.var_name = var_name.lower().replace(" ", "_")

    def __call__(self, loader, train_records, test_records):
        self.convert(loader.train, train_records)
        self.convert(loader.test, test_records)

    def convert(self, data, records):
        for obs_id, val in data[self.col_name].iteritems():
            records[obs_id].update({self.var_name: float(val)})