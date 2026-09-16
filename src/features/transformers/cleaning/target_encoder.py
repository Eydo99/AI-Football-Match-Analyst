# src/features/transformers/cleaning/event_class_encoder.py
import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

EVENT_CLASS_MAPPING = {
    'Carry': 0,
    'Foul': 1,
    'Other': 2,
    'Pass': 3,
    'Shot': 4,
}


class EventClassEncoder(Transformer):
    """Encodes the event_class target to a fixed integer mapping."""

    def __init__(self, column='event_class', mapping=None):
        self.column = column
        self.mapping = mapping or EVENT_CLASS_MAPPING

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = df.copy()

        if self.column not in encoded_df.columns:
            raise KeyError(f"Column '{self.column}' not found in DataFrame")

        unexpected = set(encoded_df[self.column].unique()) - set(self.mapping.keys())
        if unexpected:
            raise ValueError(
                f"Unexpected {self.column} values not in EVENT_CLASS_MAPPING: {unexpected}"
            )

        encoded_df[self.column] = encoded_df[self.column].map(self.mapping).astype('int64')
        return encoded_df