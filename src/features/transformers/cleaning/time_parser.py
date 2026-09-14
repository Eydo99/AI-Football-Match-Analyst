import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

TIME_COL='timestamp'

class TimeParser(Transformer):
    def __init__(self, column=None):
        if column is None:
            column = TIME_COL
        self.column = column

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        time_parsed_df = df.copy()
        if self.column not in time_parsed_df.columns:
            raise KeyError(f"Column '{self.column}' not found in dataframe")
        time_parsed_df['event_time_seconds'] = pd.to_timedelta(time_parsed_df[self.column]).dt.total_seconds()
        time_parsed_df=time_parsed_df.drop(self.column, axis=1)
        return time_parsed_df
