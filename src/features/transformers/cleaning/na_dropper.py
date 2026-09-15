import pandas as pd
from typing_extensions import override

from src.features.base import Transformer


class NaDropper(Transformer):


    def __init__(self, subset: list):
        self.subset = subset

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.subset if c not in df.columns]
        if missing:
            raise KeyError(f"NaDropper expected columns {missing}, not found in DataFrame")
        return df.dropna(subset=self.subset).reset_index(drop=True)