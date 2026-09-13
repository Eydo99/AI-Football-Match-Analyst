from typing import override
import pandas as pd
from src.features.base import Transformer


TRANSFORMED_COL='shot_outcome'
class GoalOutcomeTransformer(Transformer):
    def __init__(self, column=None):
        if column is None:
            column = TRANSFORMED_COL
        self.column = column

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        if self.column not in transformed_df.columns:
            raise KeyError((f"Column '{self.column}' not found in DataFrame"))

        transformed_df["ends_in_goal"] = (transformed_df[self.column]=='Goal').astype(int)

        return transformed_df


class TrajectoryLength(Transformer):
    def __init__(self, start_x=None, start_y=None, end_col_map=None):
        # TODO: defaults for start_x/start_y ('ball_x_start'/'ball_y_start')
        # TODO: end_col_map -- some structure mapping event 'type' to
        # which (x_end, y_end) column pair applies, e.g. a dict keyed by type
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: df.copy(), KeyError checks
        # TODO: for each row, pick the right end-location columns based on
        # the event's type (or coalesce whichever end-location pair is non-null)
        # TODO: compute Euclidean distance -> 'trajectory_length'
        # TODO: return
        pass


class TrajectoryAngle(Transformer):
    def __init__(self, start_x=None, start_y=None, end_col_map=None):
        # TODO: same shape as TrajectoryLength
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: same row-wise end-location selection as TrajectoryLength
        # TODO: atan2(y_end - y_start, x_end - x_start) -> 'trajectory_angle'
        # TODO: return
        pass