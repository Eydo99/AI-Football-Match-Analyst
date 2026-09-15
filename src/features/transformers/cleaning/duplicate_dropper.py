import pandas as pd
from typing_extensions import override

from src.features.base import Transformer


class DuplicateDropper(Transformer):
    """Drops duplicate rows and resets the index. Generic — reused across pipelines."""

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)