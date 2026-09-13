import pandas as pd
from typing_extensions import override

from src.features.utils import safe_extract
from src.features.base import Transformer

DISTANCE_COLS =[
    ('location', 'ball_x_start', 'ball_y_start'),
    ('pass_end_location', 'pass_x_end', 'pass_y_end'),
    ('shot_end_location', 'shot_x_end', 'shot_y_end'),
    ('goalkeeper_end_location', 'goalkeeper_x_end', 'goalkeeper_y_end'),
    ('carry_end_location', 'carry_x_end', 'carry_y_end'),
]
class LocationSplitter(Transformer):
    def __init__(self, columns=None):
        if columns is None:
            columns=DISTANCE_COLS
        self.columns = columns

    @override
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        location_split_df = df.copy()
        for col,x,y in self.columns:
            location_split_df[x] = location_split_df[col].apply(lambda X: safe_extract(X,0))
            location_split_df[y] = location_split_df[col].apply(lambda Y: safe_extract(Y,1))

        source_cols=[col[0] for col in self.columns ]
        location_split_df=location_split_df.drop(columns=source_cols,errors='ignore')

        return location_split_df


