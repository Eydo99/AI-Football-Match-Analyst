import pandas as pd
from src.features.base import Transformer

class BallSpeedImputer(Transformer):
    def __init__(self, column='ball_speed',group=None):
        self.column = column
        self.group = group
    def transform(self, df):
        result = df.copy()

        result[self.column+'_missing'] = result[self.column].isna().astype(int)
        if self.group is None:
             result[self.column] = result[self.column].fillna(result[self.column].median())
        else:
            result[self.column] = result.groupby(self.group)[self.column].transform(lambda x: x.fillna(x.median()))
        return result