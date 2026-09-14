from typing import override
import pandas as pd
import numpy as np
from src.features.base import Transformer


TRANSFORMED_COL = 'shot_outcome'


def _select_end_location(df, end_col_map):
    """Adds _x_end/_y_end columns to df based on each row's 'type',
    per end_col_map. Returns the modified df."""
    df=df.copy()
    df["_x_end"] = np.nan
    df["_y_end"] = np.nan

    for event_type, (x_col, y_col) in end_col_map.items():
        mask = df["type"] == event_type
        if not mask.any():
            continue
        if x_col not in df.columns:
            raise KeyError(f"column '{x_col}' not found in DataFrame")
        if y_col not in df.columns:
            raise KeyError(f"column '{y_col}' not found in DataFrame")

        df.loc[mask, "_x_end"] = df.loc[mask, x_col]
        df.loc[mask, "_y_end"] = df.loc[mask, y_col]

    return df



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
    def __init__(self, start_x=None, start_y=None, end_col_map=None):
        self.start_x = start_x or "ball_x_start"
        self.start_y = start_y or "ball_y_start"
        self.end_col_map = end_col_map or {
            "Pass": ("pass_x_end", "pass_y_end"),
            "Shot": ("shot_x_end", "shot_y_end"),
            "Carry": ("carry_x_end", "carry_y_end"),
            "Goal Keeper": ("goalkeeper_x_end", "goalkeeper_y_end"),
        }

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        if self.start_x not in transformed_df.columns:
            raise KeyError(f"column '{self.start_x}' not found in DataFrame")
        if self.start_y not in transformed_df.columns:
            raise KeyError(f"column '{self.start_y}' not found in DataFrame")
        if "type" not in transformed_df.columns:
            raise KeyError("column 'type' not found in DataFrame")

        transformed_df = _select_end_location(transformed_df, self.end_col_map)

        transformed_df["trajectory_length"] = np.sqrt(
            (transformed_df["_x_end"] - transformed_df[self.start_x]) ** 2
            + (transformed_df["_y_end"] - transformed_df[self.start_y]) ** 2
        )
        transformed_df = transformed_df.drop(columns=["_x_end", "_y_end"])
        return transformed_df


class TrajectoryAngle(Transformer):
    def __init__(self, start_x=None, start_y=None, end_col_map=None):
        self.start_x = start_x or "ball_x_start"
        self.start_y = start_y or "ball_y_start"
        self.end_col_map = end_col_map or {
            "Pass": ("pass_x_end", "pass_y_end"),
            "Shot": ("shot_x_end", "shot_y_end"),
            "Carry": ("carry_x_end", "carry_y_end"),
            "Goal Keeper": ("goalkeeper_x_end", "goalkeeper_y_end"),
        }

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        if self.start_x not in transformed_df.columns:
            raise KeyError(f"column '{self.start_x}' not found in DataFrame")
        if self.start_y not in transformed_df.columns:
            raise KeyError(f"column '{self.start_y}' not found in DataFrame")
        if "type" not in transformed_df.columns:
            raise KeyError("column 'type' not found in DataFrame")

        transformed_df = _select_end_location(transformed_df, self.end_col_map)

        transformed_df["trajectory_angle"] = np.arctan2(
            transformed_df["_y_end"] - transformed_df[self.start_y],
            transformed_df["_x_end"] - transformed_df[self.start_x],
        )

        transformed_df = transformed_df.drop(columns=["_x_end", "_y_end"])
        return transformed_df