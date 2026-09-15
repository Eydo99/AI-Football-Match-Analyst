import pandas as pd
from src.features.base import Transformer

class BallSpeedImputer(Transformer):
    def transform(self, df):
        result = df.copy()
        result['ball_speed_missing'] = result['ball_speed'].isna().astype(int)
        result['ball_speed'] = result['ball_speed'].fillna(result['ball_speed'].median())
        return result