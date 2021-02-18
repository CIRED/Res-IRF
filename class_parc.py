import pandas as pd


class DataFrameParc:
    """
    N-Dimension MultiIndex pandas Series

    """

    def __init__(self, df):

        self.serie = df
        self.number_levels = len(self.serie.index.names)
        self.index_names = self.serie.index.names


if __name__ == '__main__':

    pass