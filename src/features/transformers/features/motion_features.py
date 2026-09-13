from typing import override
import pandas as pd

from src.features.base import Transformer


class BallSpeed(Transformer):
    def __init__(self, time_col=None, x_col=None, y_col=None, group_col=None):
        # TODO: defaults ('event_time_seconds', 'ball_x_start', 'ball_y_start',
        # and a grouping column -- CHECK your table for a surviving match
        # identifier column first, this decides the shape of transform()
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: df.copy(), KeyError checks
        # TODO: sort by group_col (if any) + time_col to ensure order
        # TODO: groupby(group_col) if grouping exists, else flat -- compute
        # position delta and time delta vs. previous row (.diff())
        # TODO: speed = distance / time_delta -- decide what happens at
        # the first row of each group (no previous row to diff against)
        # TODO: -> 'ball_speed'
        # TODO: return
        pass