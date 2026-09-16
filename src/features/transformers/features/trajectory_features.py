from typing import override
import pandas as pd
import numpy as np
from src.features.base import Transformer


TRANSFORMED_COL = 'shot_outcome'
class GoalOutcomeTransformer(Transformer):
    def __init__(self, column=None):
        if column is None:
            column = TRANSFORMED_COL
        self.column = column

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        if self.column not in transformed_df.columns:
            raise KeyError(f"column '{self.column}' not found in DataFrame")

        transformed_df["ends_in_goal"] = (transformed_df[self.column] == 'Goal').astype(int)

        return transformed_df


class TrajectoryLength(Transformer):
    def __init__(self, start_x=None, start_y=None, end_x=None,end_y=None):
        self.start_x = start_x or "ball_x_start"
        self.start_y = start_y or "ball_y_start"
        self.end_x = end_x or "x_end"
        self.end_y = end_y or "y_end"

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()

        for col in (self.start_x, self.end_x, self.start_y, self.end_y):
            if col not in transformed_df.columns:
                raise KeyError(f"column '{col}' not found in DataFrame")

        transformed_df["trajectory_length"] = np.sqrt(
            (transformed_df[self.end_x] - transformed_df[self.start_x]) ** 2
            + (transformed_df[self.end_y] - transformed_df[self.start_y]) ** 2
        )
        return transformed_df


class TrajectoryAngle(Transformer):
    def __init__(self, start_x=None, start_y=None, end_x=None,end_y=None):
        self.start_x = start_x or "ball_x_start"
        self.start_y = start_y or "ball_y_start"
        self.end_x = end_x or "x_end"
        self.end_y = end_y or "y_end"

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        for col in (self.start_x, self.end_x, self.start_y, self.end_y):
            if col not in transformed_df.columns:
                raise KeyError(f"column '{col}' not found in DataFrame")

        transformed_df["trajectory_angle"] = np.arctan2(
            transformed_df[self.end_y] - transformed_df[self.start_y],
            transformed_df[self.end_x] - transformed_df[self.start_x],
        )
        return transformed_df