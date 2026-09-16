import numpy as np
import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

END_COL_CANDIDATES = [
    ('pass_x_end', 'pass_y_end'),
    ('shot_x_end', 'shot_y_end'),
    ('carry_x_end', 'carry_y_end'),
    ('goalkeeper_x_end', 'goalkeeper_y_end'),
]


class EndLocationCollapser(Transformer):
    def __init__(self, end_col_candidates=None, drop_source=True):
        self.end_col_candidates = end_col_candidates or END_COL_CANDIDATES
        self.drop_source = drop_source

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()

        missing = [
            col for pair in self.end_col_candidates for col in pair
            if col not in transformed_df.columns
        ]
        if missing:
            raise KeyError(f"columns {missing} not found in DataFrame")

        transformed_df['x_end'] = np.nan
        transformed_df['y_end'] = np.nan

        for x_col, y_col in self.end_col_candidates:
            mask = transformed_df['x_end'].isna() & transformed_df[x_col].notna()
            transformed_df.loc[mask, 'x_end'] = transformed_df.loc[mask, x_col]
            transformed_df.loc[mask, 'y_end'] = transformed_df.loc[mask, y_col]

        if self.drop_source:
            source_cols = [col for pair in self.end_col_candidates for col in pair]
            transformed_df = transformed_df.drop(columns=source_cols, errors='ignore')

        return transformed_df