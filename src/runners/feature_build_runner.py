import os

import pandas as pd

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.column_pruner import ColumnPruner
from src.features.transformers.cleaning.boolean_encoder import BooleanEncoder
from src.features.transformers.features.full_tier_features import TeamCentroidDistance, OpenAngleGoal
from src.features.transformers.features.geometry_features import DistToGoal, ShotAngle, InPenaltyBox
from src.features.transformers.features.motion_features import BallSpeed
from src.features.transformers.features.pressure_features import DistNearestDefender, DefendersIn3m
from src.features.transformers.features.trajectory_features import GoalOutcomeTransformer, TrajectoryAngle, \
    TrajectoryLength
from src.features.transformers.cleaning.location_splitter import LocationSplitter
from src.features.transformers.cleaning.coordinate_rescaler import CoordinateRescaler
from src.features.transformers.cleaning.time_parser import TimeParser
from src.features.transformers.cleaning.freeze_frame_extractor import FreezeFrameExtractor
from src.runners.base_runner import BaseRunner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_EVENT_TYPES = ['Pass', 'Pressure', 'Foul Won', 'Shot', 'Goal Keeper', 'Foul Committed', 'Carry']


class FeatureBuildRunner(BaseRunner):

    def __init__(self, events_dir: str = None, output_dir: str = None):
        self.events_dir = events_dir or os.path.join(PROJECT_ROOT, 'data', 'raw', 'events')
        output_dir = output_dir or os.path.join(PROJECT_ROOT, 'data', 'processed')
        self.master_df_path = os.path.join(output_dir, 'master_df.parquet')
        self.statsbomb_xg_path = os.path.join(output_dir, 'statsbomb_xg.parquet')

        # input_path/output_path aren't single files here (see load_input()
        # and run() overrides below) — passed through anyway to satisfy
        # BaseRunner's constructor contract.
        super().__init__(input_path=self.events_dir, output_path=self.master_df_path)

    def load_input(self) -> pd.DataFrame:
        """Load every raw per-match parquet, keep only the required event types."""
        event_files = [
            os.path.join(self.events_dir, f)
            for f in os.listdir(self.events_dir) if f.endswith('.parquet')
        ]

        filtered_dfs = []
        for file in event_files:
            try:
                df = pd.read_parquet(file)
                subset = df[df['type'].isin(REQUIRED_EVENT_TYPES)]
                if not subset.empty:
                    filtered_dfs.append(subset)
            except Exception:
                print(f"Error reading {file}")

        return pd.concat(filtered_dfs, ignore_index=True)

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ColumnPruner(flag="PRE"), LocationSplitter(), CoordinateRescaler(), BooleanEncoder(), TimeParser(),
            FreezeFrameExtractor(), DistToGoal(), ShotAngle(), InPenaltyBox(), GoalOutcomeTransformer(),
            TrajectoryAngle(), TrajectoryLength(), DistNearestDefender(), DefendersIn3m(), BallSpeed(),
            TeamCentroidDistance(), OpenAngleGoal(), ColumnPruner(flag="POST"),
        ])

    def run(self) -> pd.DataFrame:
        raw_events_df = self.load_input()
        statsbomb_xg_validation = raw_events_df[raw_events_df['type'] == 'Shot'][
            ['id', 'match_id', 'shot_statsbomb_xg']
        ]

        pipeline = self.build_pipeline()
        master_df = pipeline.transform(raw_events_df)

        print(master_df.columns)
        print(master_df.dtypes)
        print(master_df.shape)

        self.save_dataframe(master_df, self.master_df_path, label="master df")
        self.save_dataframe(statsbomb_xg_validation, self.statsbomb_xg_path, label="statsbomb_xg")

        return master_df


if __name__ == "__main__":
    FeatureBuildRunner().run()