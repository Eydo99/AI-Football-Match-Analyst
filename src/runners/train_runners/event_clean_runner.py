import os

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.event_class_mapper import EventClassMapper
from src.features.transformers.cleaning.duplicate_dropper import DuplicateDropper
from src.features.transformers.cleaning.column_pruner import ColumnPruner
from src.features.transformers.cleaning.ball_speed_imputer import BallSpeedImputer
from src.features.transformers.cleaning.trajectory_imputer import TrajectoryImputer
from src.features.transformers.cleaning.na_dropper import NaDropper
from src.runners.base_runner import BaseRunner,PROJECT_ROOT
from src.features.transformers.cleaning.target_encoder import EventClassEncoder

# Cols only known for shot rows (need the freeze frame, which only shots have).
SHOT_ONLY_COLS = ['team_centroid_distance', 'dist_nearest_defender', 'open_angle_goal']

# Cols dropped once event_class exists — raw type label and the xG-only target.
POST_CLASS_DROP_COLS = ['team', 'type', 'ends_in_goal']

# Rows missing these core geometric features are <0.01% of the data — drop, don't impute.
CORE_GEOMETRIC_COLS = ['ball_x_start', 'ball_y_start', 'dist_to_goal', 'shot_angle']


class EventCleaningRunner(BaseRunner):
    """Builds the 5-class event_class dataframe for the event classification models."""

    def __init__(self, input_path: str = None, output_path: str = None):
        input_path = input_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'master_df.parquet')
        output_path = output_path or os.path.join(PROJECT_ROOT, 'data', 'processed', 'events_df.parquet')
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            missing_input_hint="run FeatureBuildRunner().run() first to build it.",
        )

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            EventClassMapper(),
            EventClassEncoder(),
            DuplicateDropper(),
            ColumnPruner(columns=SHOT_ONLY_COLS),
            BallSpeedImputer(column='ball_speed', group='event_class'),
            TrajectoryImputer(length_col='trajectory_length', angle_col='trajectory_angle'),
            NaDropper(subset=CORE_GEOMETRIC_COLS),
            ColumnPruner(columns=POST_CLASS_DROP_COLS),
            DuplicateDropper(),
        ])


if __name__ == "__main__":
    EventCleaningRunner().run()