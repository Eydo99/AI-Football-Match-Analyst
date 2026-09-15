import pandas as pd
from typing_extensions import override

from src.features.base import Transformer


class TrajectoryImputer(Transformer):

    def __init__(self, length_col='trajectory_length', angle_col='trajectory_angle'):
        self.length_col = length_col
        self.angle_col = angle_col

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in (self.length_col, self.angle_col):
            if col not in df.columns:
                raise KeyError(f"TrajectoryImputer expected column '{col}' but it was not found in the DataFrame")

        result = df.copy()
        result['trajectory_missing'] = result[self.length_col].isna().astype(int)
        result[self.length_col] = result[self.length_col].fillna(0)
        result[self.angle_col] = result[self.angle_col].fillna(0)
        return result