from typing import override

import numpy as np
import pandas as pd

from src.features.base import Transformer
from src.features.utils import safe_extract

EXTRACTED_COL='shot_freeze_frame'

class FreezeFrameExtractor(Transformer):
    def __init__(self,column=None):
        if column is None:
            column = EXTRACTED_COL
        self.column = column

    @override
    def transform(self,df: pd.DataFrame) -> pd.DataFrame:
        extracted_df = df.copy()

        if self.column not in extracted_df.columns:
            raise KeyError(f"Column '{self.column}' not found in DataFrame")

        extracted_df['opponent_locations'] = extracted_df[self.column].apply(self._extract_opponent_locations)
        extracted_df = extracted_df.drop(self.column, axis=1)
        return extracted_df


    def _extract_opponent_locations(self,freeze_frame):

        if freeze_frame is None or not isinstance(freeze_frame, (list, tuple, np.ndarray)):
            return []

        opponent_locations = []
        for entry in freeze_frame:
            if not entry.get('teammate', True):
                loc = entry.get('location')
                x = safe_extract(loc, 0) * (105 / 120)
                y = safe_extract(loc, 1) * (68 / 80)
                opponent_locations.append((x, y))

        return opponent_locations
