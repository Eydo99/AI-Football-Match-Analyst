import pandas as pd
from src.features.base import Transformer


class BallSpeedImputer(Transformer):
    def __init__(self, column='ball_speed', group=None, fill_value=None):
        self.column = column
        self.group = group
        self.fill_value = fill_value  # precomputed constant, takes priority over live median

    def transform(self, df):
        result = df.copy()
        result[self.column + '_missing'] = result[self.column].isna().astype(int)

        if self.fill_value is not None:
            result[self.column] = result[self.column].fillna(self.fill_value)
        elif self.group is None:
            result[self.column] = result[self.column].fillna(result[self.column].median())
        else:
            result[self.column] = result.groupby(self.group)[self.column].transform(
                lambda x: x.fillna(x.median())
            )
        return result