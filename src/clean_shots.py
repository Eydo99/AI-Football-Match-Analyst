import os
import pandas as pd

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.shot_filter import ShotFilter
from src.features.transformers.cleaning.opponent_data_dropper import OpponentDataDropper
from src.features.transformers.cleaning.ball_speed_imputer import BallSpeedImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean_shots(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    
    if input_path is None:
        input_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'master_df.parquet')
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'shot_df_clean.parquet')

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"master_df.parquet not found at {input_path} — run runner.py first to build it."
        )

    master_df = pd.read_parquet(input_path)

    shot_cleaning_pipeline = Pipeline([
        ShotFilter(),
        OpponentDataDropper(),
        BallSpeedImputer(),
    ])

    clean_shot_df = shot_cleaning_pipeline.transform(master_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_shot_df.to_parquet(output_path, index=False)
    
    print(f"Saved clean shot df to {output_path}")

    return clean_shot_df


if __name__ == "__main__":
    clean_shots()