"""
A runners whose shape doesn't fit the simple one-input/one-output template
(FeatureBuildRunner: scans a directory instead of reading one file, and
writes two outputs) overrides load_input() and/or run() directly — see
FeatureBuildRunner for that case. save_dataframe() is exposed as a reusable
building block for exactly that situation.
"""
import os
from abc import ABC, abstractmethod

import pandas as pd


class BaseRunner(ABC):

    def __init__(self, input_path: str, output_path: str, missing_input_hint: str = ""):
        self.input_path = input_path
        self.output_path = output_path
        self.missing_input_hint = missing_input_hint

    @abstractmethod
    def build_pipeline(self):
        """Return the Pipeline (list of Transformers) this runners executes."""
        raise NotImplementedError

    def load_input(self) -> pd.DataFrame:
        """Default: read a single parquet at self.input_path. Override for anything else."""
        if not os.path.exists(self.input_path):
            hint = f" — {self.missing_input_hint}" if self.missing_input_hint else ""
            raise FileNotFoundError(f"{os.path.basename(self.input_path)} not found at {self.input_path}{hint}")
        return pd.read_parquet(self.input_path)

    def save_dataframe(self, df: pd.DataFrame, path: str, label: str = None) -> None:
        """Shared save+print helper — also used directly by runners with more than one output."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Saved {label or os.path.basename(path)} to {path}")

    def run(self) -> pd.DataFrame:
        """Template method: load -> transform -> save -> return."""
        raw_df = self.load_input()
        pipeline = self.build_pipeline()
        result_df = pipeline.transform(raw_df)
        self.save_dataframe(result_df, self.output_path, label=self.__class__.__name__)
        return result_df

    def load_output(self) -> pd.DataFrame:
        """Read this runners's already-saved output back from disk, skipping run()."""
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(
                f"{os.path.basename(self.output_path)} not found at {self.output_path} — "
                f"run {self.__class__.__name__}().run() first."
            )
        return pd.read_parquet(self.output_path)