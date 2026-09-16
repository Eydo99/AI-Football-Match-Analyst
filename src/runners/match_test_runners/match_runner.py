import os

import pandas as pd
from statsbombpy import sb

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.column_pruner import ColumnPruner, DROP_COLS_PRE_FEATURE_ENG, \
    DROP_COLS_POST_FEATURE_ENG
from src.features.transformers.cleaning.boolean_encoder import BOOLEAN_COLS
from src.features.transformers.cleaning.location_splitter import LocationSplitter
from src.features.transformers.cleaning.end_location_collapser import EndLocationCollapser
from src.features.transformers.cleaning.coordinate_rescaler import CoordinateRescaler
from src.features.transformers.cleaning.time_parser import TimeParser
from src.features.transformers.cleaning.freeze_frame_extractor import FreezeFrameExtractor
from src.features.transformers.features.geometry_features import DistToGoal, ShotAngle, InPenaltyBox
from src.features.transformers.features.trajectory_features import TrajectoryLength, TrajectoryAngle
from src.features.transformers.features.pressure_features import DistNearestDefender, DefendersIn3m
from src.features.transformers.features.motion_features import BallSpeed
from src.features.transformers.features.full_tier_features import TeamCentroidDistance, OpenAngleGoal
from src.runners.base_runner import BaseRunner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# No target columns exist for a new, unlabeled match — type and shot_outcome
# are the raw inputs EventClassMapper/GoalOutcomeTransformer would have used
# to build event_class/ends_in_goal during training, but neither of those
# steps runs here, so both source columns are dropped up front instead.
# BOOLEAN_COLS are dropped for the same reason discussed earlier: their
# non-null pattern (pass_*/shot_*/foul_*) leaks event family just as surely
# as `type` itself would.
INFERENCE_PRE_DROP = DROP_COLS_PRE_FEATURE_ENG + BOOLEAN_COLS + ['type', 'shot_outcome']


class MatchInferenceRunner(BaseRunner):
    def __init__(self, match_id, output_dir: str = None):
        self.match_id = match_id
        output_dir = output_dir or os.path.join(PROJECT_ROOT, 'data', 'processed', 'inference')
        output_path = os.path.join(output_dir, f'match_{match_id}_features.parquet')

        super().__init__(
            input_path=str(match_id),
            output_path=output_path,
            missing_input_hint=f"sb.events() returned no data for match_id={match_id}",
        )

    def load_input(self) -> pd.DataFrame:
        """Fetch this match's events directly from the StatsBomb API,
        filtered to the same event types the training pipeline keeps."""
        df = sb.events(match_id=self.match_id)

        if df is None or df.empty:
            raise ValueError(f"sb.events() returned no rows for match_id={self.match_id}")

        df['event_id'] = df['id']

        if df.empty:
            raise ValueError(f"match_id={self.match_id} produced no rows after event-type filtering")
        return df

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ColumnPruner(columns=INFERENCE_PRE_DROP),
            LocationSplitter(),
            EndLocationCollapser(),
            CoordinateRescaler(columns=[
                ('ball_x_start', 'x'), ('ball_y_start', 'y'),
                ('x_end', 'x'), ('y_end', 'y'),
            ]),
            TimeParser(),
            FreezeFrameExtractor(),
            DistToGoal(),
            ShotAngle(),
            InPenaltyBox(),
            TrajectoryLength(),
            TrajectoryAngle(),
            DistNearestDefender(),
            DefendersIn3m(),
            BallSpeed(),
            TeamCentroidDistance(),
            OpenAngleGoal(),
            ColumnPruner(columns=DROP_COLS_POST_FEATURE_ENG),
        ])


if __name__ == "__main__":
    import sys
    match_id = sys.argv[1] if len(sys.argv) > 1 else None
    if match_id is None:
        raise SystemExit("usage: python match_inference_runner.py <match_id>")
    MatchInferenceRunner(match_id).run()