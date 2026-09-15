import pandas as pd
from src.features.base import Transformer

class OpponentDataDropper(Transformer):
    REQUIRED_COLUMNS = ['dist_nearest_defender', 'team_centroid_distance', 'open_angle_goal']

    def transform(self, df):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"OpponentDataDropper requires columns {missing}, not found in DataFrame")
        return df.dropna(subset=['dist_nearest_defender']).copy()