import os

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.column_pruner import ColumnPruner
from src.features.transformers.cleaning.shot_filter import ShotFilter
from src.features.transformers.cleaning.opponent_data_dropper import OpponentDataDropper
from src.features.transformers.cleaning.ball_speed_imputer import BallSpeedImputer
from src.runners.base_runner import BaseRunner
from src.runners.event_clean_runner import EventCleaningRunner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

events_df = EventCleaningRunner().load_output()
SHOT_BALL_SPEED_FALLBACK_MEDIAN =  events_df[events_df['event_class'] == 4]['ball_speed'].median()



class MatchShotCleanRunner(BaseRunner):
    """Cleaning stage for match-inference shots: mirrors ShotCleaningRunner but
    reads from the predicted-type match_events_df (no ends_in_goal — that's
    the xG target, doesn't exist pre-prediction), and BallSpeedImputer uses a
    fixed shot-specific fallback instead of a live per-match median."""

    def __init__(self, match_id, input_dir: str = None, output_dir: str = None):
        self.match_id = match_id
        input_dir = input_dir or os.path.join(PROJECT_ROOT, 'data', 'processed', 'inference')
        output_dir = output_dir or os.path.join(PROJECT_ROOT, 'data', 'processed', 'inference')

        input_path = os.path.join(input_dir, f'match_{match_id}_events_with_type.parquet')
        output_path = os.path.join(output_dir, f'match_{match_id}_shots_clean.parquet')

        super().__init__(
            input_path=input_path,
            output_path=output_path,
            missing_input_hint=f"run the type-prediction pipeline for match_id={match_id} first "
                                f"to build match_{match_id}_events_with_type.parquet.",
        )

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ShotFilter(),
            OpponentDataDropper(),
            BallSpeedImputer(fill_value=SHOT_BALL_SPEED_FALLBACK_MEDIAN),
            ColumnPruner(columns=['minute','second','player'])
        ])