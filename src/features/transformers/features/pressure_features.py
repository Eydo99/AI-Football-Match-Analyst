from typing import override
import pandas as pd

from src.features.base import Transformer

DEFENDER_RADIUS_M = 3


class DistNearestDefender(Transformer):
    def __init__(self, x_col=None, y_col=None, opponents_col=None):
        # TODO: defaults ('ball_x_start', 'ball_y_start', 'opponent_locations')
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: df.copy(), KeyError checks
        # TODO: per-row: compute distance from (x,y) to every tuple in
        # opponents_col, take min. Decide: empty list -> what value?
        # TODO: -> 'dist_nearest_defender'
        # TODO: return
        pass


class DefendersIn3m(Transformer):
    def __init__(self, x_col=None, y_col=None, opponents_col=None, radius=None):
        # TODO: same defaults, plus radius default DEFENDER_RADIUS_M
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: df.copy(), KeyError checks
        # TODO: per-row: count tuples in opponents_col within radius of (x,y)
        # empty list -> 0, not NaN
        # TODO: -> 'defenders_in_3m'
        # TODO: return
        pass