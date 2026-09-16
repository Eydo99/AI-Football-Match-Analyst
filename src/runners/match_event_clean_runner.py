import os

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.duplicate_dropper import DuplicateDropper
from src.features.transformers.cleaning.column_pruner import ColumnPruner
from src.features.transformers.cleaning.ball_speed_imputer import BallSpeedImputer
from src.features.transformers.cleaning.trajectory_imputer import TrajectoryImputer
from src.features.transformers.cleaning.na_dropper import NaDropper
from src.runners.base_runner import BaseRunner

from src.runners.event_clean_runner import SHOT_ONLY_COLS, POST_CLASS_DROP_COLS, CORE_GEOMETRIC_COLS, \
    EventCleaningRunner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVENT_CLEAN_COLUMN_DROPPED = SHOT_ONLY_COLS + ['player', 'minute', 'second']
BALL_SPEED_FALLBACK_MEDIAN = EventCleaningRunner().load_output()['ball_speed'].median()


class MatchEventCleanRunner(BaseRunner):
    """Cleaning stage for match-inference: mirrors EventCleaningRunner but with
    no EventClassMapper/EventClassEncoder (event_class doesn't exist pre-prediction),
    and BallSpeedImputer uses a fixed global fallback instead of grouping by
    event_class, since that column isn't available at this stage either."""

    def __init__(self, match_id, input_dir: str = None, output_dir: str = None):
        self.match_id = match_id
        input_dir = input_dir or os.path.join(PROJECT_ROOT, 'data', 'processed', 'inference')
        output_dir = output_dir or os.path.join(PROJECT_ROOT, 'data', 'processed', 'inference')

        input_path = os.path.join(input_dir, f'match_{match_id}_features.parquet')
        output_path = os.path.join(output_dir, f'match_{match_id}_clean.parquet')

        super().__init__(
            input_path=input_path,
            output_path=output_path,
            missing_input_hint=f"run MatchInferenceRunner({match_id}).run() first to build it.",
        )

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            DuplicateDropper(),
            ColumnPruner(columns=EVENT_CLEAN_COLUMN_DROPPED),
            BallSpeedImputer(column='ball_speed', fill_value=BALL_SPEED_FALLBACK_MEDIAN),
            TrajectoryImputer(length_col='trajectory_length', angle_col='trajectory_angle'),
            NaDropper(subset=CORE_GEOMETRIC_COLS),
            ColumnPruner(columns=POST_CLASS_DROP_COLS),
            DuplicateDropper(),
        ])