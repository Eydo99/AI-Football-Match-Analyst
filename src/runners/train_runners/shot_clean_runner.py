import os

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.shot_filter import ShotFilter
from src.features.transformers.cleaning.opponent_data_dropper import OpponentDataDropper
from src.features.transformers.cleaning.ball_speed_imputer import BallSpeedImputer
from src.runners.base_runner import BaseRunner,PROJECT_ROOT



class ShotCleaningRunner(BaseRunner):
    """Filters master_df down to a cleaned shots-only dataframe for the xG models."""

    def __init__(self, input_path: str = None, output_path: str = None):
        input_path = input_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'master_df.parquet')
        output_path = output_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'shot_df_clean.parquet')
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            missing_input_hint="run FeatureBuildRunner().run() first to build it.",
        )

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ShotFilter(),
            OpponentDataDropper(),
            BallSpeedImputer(),
        ])

    def load_X_y(self, drop_cols: list, target_col: str) -> tuple:
        """Load features (X) and target (y) from the cleaned shot dataframe."""
        df = self.load_output()

        if target_col not in df.columns:
            raise KeyError(f"target_col '{target_col}' not found in shot_df_clean columns.")

        missing = [c for c in drop_cols if c not in df.columns]
        if missing:
            raise KeyError(f"drop_cols not found in dataframe: {missing}")

        y = df[target_col]
        cols_to_drop = set(drop_cols) | {target_col}
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return X, y


if __name__ == "__main__":
    ShotCleaningRunner().run()