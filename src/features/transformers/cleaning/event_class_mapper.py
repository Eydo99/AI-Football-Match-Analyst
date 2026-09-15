import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

DEFAULT_CLASS_MAP = {
    'Shot': 'Shot',
    'Pass': 'Pass',
    'Carry': 'Carry',
    'Foul Committed': 'Foul',
    'Foul Won': 'Foul',
    'Pressure': 'Other',
    'Goal Keeper': 'Other',
}


class EventClassMapper(Transformer):
    """Adds an `event_class` column mapped from `type`, per DEFAULT_CLASS_MAP."""

    def __init__(self, class_map=None, source_col='type', target_col='event_class'):
        if class_map is None:
            class_map = DEFAULT_CLASS_MAP
        self.class_map = class_map
        self.source_col = source_col
        self.target_col = target_col

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.source_col not in df.columns:
            raise KeyError(f"EventClassMapper expected column '{self.source_col}' but it was not found in the DataFrame")
        mapped_df = df.copy()
        mapped_df[self.target_col] = mapped_df[self.source_col].map(self.class_map)
        return mapped_df