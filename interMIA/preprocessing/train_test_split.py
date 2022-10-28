import pandas as pd


class TrainValTestSplit():
    """
    Splits a given csv file into train test split with approximately
    equal distribution of labels
    """

    def __init__(self, in_csv_all, in_csv_paths, train, val, test):
        self.in_df_all = pd.read_csv(in_csv_all)
        self.in_df_paths = pd.read_csv(in_csv_paths)
        self.train = train
        self.val = val
        self.test = test

    def split(self, ):
        pass
