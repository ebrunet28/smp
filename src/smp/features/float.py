from smp.features.features import Feature


class Float(Feature):
    def convert(self, data, records):
        for obs_id, val in data[self.col_name].iteritems():
            records[obs_id].update({self.var_name: float(val)})


class UtcOffset(Float):
    def __init__(self):
        super().__init__("UTC Offset")


class NumOfFollowers(Float):
    def __init__(self):
        super().__init__("Num of Followers")


class NumOfPeopleFollowing(Float):
    def __init__(self):
        super().__init__("Num of People Following")


class NumOfStatusUpdates(Float):
    def __init__(self):
        super().__init__("Num of Status Updates")


class NumOfDirectMessages(Float):
    def __init__(self):
        super().__init__("Num of Direct Messages")


class AvgDailyProfileVisitDuration(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Visit Duration in seconds")


class AvgDailyProfileClicks(Float):
    def __init__(self):
        super().__init__("Avg Daily Profile Clicks")
