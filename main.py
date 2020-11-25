if __name__ == "__main__":

    import datetime as dt

    import pandas as pd

    from loader import Loader

    loader = Loader()

    predictions = pd.read_csv("data/sample_submission.csv")

    predictions.to_csv("submissions/submission_{}".format(dt.datetime.now()))
