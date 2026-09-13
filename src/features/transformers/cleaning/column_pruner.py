import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

DROP_COLS = [
        # Event types entirely outside req (Duel, Dribble, Block, Clearance,
        # Ball Recovery, Ball Receipt, Interception, Substitution, Tactics,
        # 50/50, Bad Behaviour, Half Start/End, misc admin)
        'bad_behaviour_card',
        'ball_receipt_outcome',
        'ball_recovery_recovery_failure',
        'ball_recovery_offensive',
        'block_deflection',
        'block_offensive',
        'block_save_block',
        'clearance_aerial_won',
        'clearance_body_part',
        'clearance_head',
        'clearance_left_foot',
        'clearance_right_foot',
        'clearance_other',
        'dribble_nutmeg',
        'dribble_outcome',
        'dribble_overrun',
        'dribble_no_touch',
        'duel_outcome',
        'duel_type',
        'interception_outcome',
        'substitution_outcome',
        'substitution_outcome_id',
        'substitution_replacement',
        'substitution_replacement_id',
        'tactics',
        '50_50',
        'half_start_late_video_start',
        'half_end_early_video_end',
        'injury_stoppage_in_chain',
        'player_off_permanent',
        'off_camera',
        'out',
        'counterpress',
        'miscontrol_aerial_won',

        # Pass — excess trivia not tied to any Stage2 feature
        'pass_backheel',
        'pass_inswinging',
        'pass_outswinging',
        'pass_straight',
        'pass_no_touch',
        'pass_miscommunication',

        # Shot — excess trivia not tied to any Stage2 feature
        'shot_saved_to_post',
        'shot_saved_off_target',
        'shot_deflected',
        'shot_open_goal',
        'shot_follows_dribble',
        'shot_redirect',

        # Goal Keeper — excess trivia not tied to any Stage2 feature
        'goalkeeper_shot_saved_to_post',
        'goalkeeper_shot_saved_off_target',
        'goalkeeper_punched_out',
        'goalkeeper_success_in_play',
        'goalkeeper_lost_in_play',
        'goalkeeper_lost_out',
        'goalkeeper_success_out',
        'goalkeeper_saved_to_post',
    ]

class ColumnPruner(Transformer):


    def __init__(self, columns=None):
        if columns is None:
            columns =DROP_COLS
        self.columns = columns
    @override
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        dropped_cols_df = df.copy()
        dropped_cols_df=dropped_cols_df.drop(columns=self.columns,errors='ignore')
        return dropped_cols_df