import pandas as pd
from typing_extensions import override

from src.features.base import Transformer

DROP_COLS_PRE_FEATURE_ENG = [
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
        'duration'
    ]

DROP_COLS_POST_FEATURE_ENG = [
    # Intermediate inputs consumed during feature engineering
    'opponent_locations',       # consumed by TeamCentroidDistance / OpenAngleGoal
    'index',                    # consumed by BallSpeed (event ordering / tie-breaking)
    'period',                   # consumed by BallSpeed (grouping)
    'event_time_seconds',       # consumed by BallSpeed (elapsed time)
    'pass_x_end', 'pass_y_end',
    'shot_x_end', 'shot_y_end',
    'goalkeeper_x_end', 'goalkeeper_y_end',
    'carry_x_end', 'carry_y_end',
    'shot_outcome',              # consumed by GoalOutcomeTransformer

    # Redundant key
    'team_id',                   # keeping 'team' instead

    # Not used by any model or feature (contextual/attribute columns)
    'play_pattern',
    'under_pressure',
    'position',
    'player',
    'player_id',
    'id',
    'related_events',
    'possession',
    'possession_team',
    'possession_team_id',
    'minute',
    'second',

    # foul_* attribute columns
    'foul_committed_advantage',
    'foul_committed_card',
    'foul_committed_offensive',
    'foul_committed_type',
    'foul_committed_penalty',
    'foul_won_advantage',
    'foul_won_defensive',
    'foul_won_penalty',

    # goalkeeper_* attribute columns
    'goalkeeper_body_part',
    'goalkeeper_outcome',
    'goalkeeper_position',
    'goalkeeper_technique',
    'goalkeeper_type',

    # pass_* attribute columns
    'pass_aerial_won',
    'pass_angle',
    'pass_assisted_shot_id',
    'pass_body_part',
    'pass_cross',
    'pass_cut_back',
    'pass_deflected',
    'pass_goal_assist',
    'pass_height',
    'pass_length',
    'pass_outcome',
    'pass_recipient',
    'pass_recipient_id',
    'pass_shot_assist',
    'pass_switch',
    'pass_technique',
    'pass_through_ball',
    'pass_type',

    # shot_* attribute columns
    'shot_aerial_won',
    'shot_body_part',
    'shot_key_pass_id',
    'shot_one_on_one',
    'shot_first_time',
    'shot_technique',
    'shot_type',
    'shot_statsbomb_xg'
]

class ColumnPruner(Transformer):

    def __init__(self, columns=None,flag="PRE"):
        if columns is None:
            if flag == "PRE":
                columns = DROP_COLS_PRE_FEATURE_ENG
            elif flag == "POST":
                columns = DROP_COLS_POST_FEATURE_ENG
        self.columns = columns
    @override
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        dropped_cols_df = df.copy()
        dropped_cols_df=dropped_cols_df.drop(columns=self.columns,errors='ignore')
        return dropped_cols_df