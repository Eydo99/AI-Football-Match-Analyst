import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

RESCALING_COLS = [
    ('ball_x_start', 'x'), ('ball_y_start', 'y'),
    ('pass_x_end', 'x'), ('pass_y_end', 'y'),
    ('shot_x_end', 'x'), ('shot_y_end', 'y'),
    ('goalkeeper_x_end', 'x'), ('goalkeeper_y_end', 'y'),
    ('carry_x_end', 'x'), ('carry_y_end', 'y'),
]

RESCALING_FACTORS={'x':(105/120),'y':68/80 }
class CoordinateRescaler(Transformer):
    def __init__(self, columns=None, factors=None):
        if columns is None:
            columns = RESCALING_COLS
        if factors is None:
            factors = RESCALING_FACTORS
        self.columns = columns
        self.factors = factors

    @override
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        rescaled_df=df.copy()

        for col,axis in self.columns:
            if col not in rescaled_df.columns:
                raise KeyError(f"Coordinate Rescaler expected column '{col}' but it was not found in the DataFrame")
            if axis == 'x':
                rescaled_df[col]=rescaled_df[col]*self.factors['x']
            elif axis == 'y':
                rescaled_df[col]=rescaled_df[col]*self.factors['y']
        return rescaled_df




