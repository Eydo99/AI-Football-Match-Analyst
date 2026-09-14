from typing import override

from src.features.base import Transformer
import pandas as pd


class Pipeline(Transformer):
    def __init__(self, transformers):
        self.transformers = transformers

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.transformers is None or len(self.transformers) == 0:
            return df
        for transformer in self.transformers:
            df = transformer.transform(df)

        return df