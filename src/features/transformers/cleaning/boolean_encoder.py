import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

BOOLEAN_COLS = ['pass_goal_assist', 'pass_shot_assist', 'pass_aerial_won', 'pass_cross', 'pass_cut_back','pass_deflected'
, 'pass_switch', 'pass_through_ball', 'shot_aerial_won', 'shot_first_time', 'shot_one_on_one','foul_committed_advantage'
, 'foul_committed_offensive', 'foul_committed_penalty', 'foul_won_advantage', 'foul_won_defensive','foul_won_penalty',
'under_pressure']

class BooleanEncoder(Transformer):

    def __init__(self, columns=None):
        if columns is None:
            columns = BOOLEAN_COLS
        self.columns = columns
    @override
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        boolean_encoded_df=df.copy()
        for col in self.columns:
            if col not in boolean_encoded_df.columns:
                raise KeyError(f"BooleanEncoder expected column '{col}' but it was not found in the DataFrame")
            boolean_encoded_df[col] = boolean_encoded_df[col].fillna(False).astype(int)

        return boolean_encoded_df
