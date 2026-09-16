"""
Browse StatsBomb competitions/seasons/matches that were NOT part of
training (see loader.py's comp_dict: La Liga=11, World Cup=43,
Champions League=16), and build engineered features for a single
selected match on demand - no pre-processed parquet needed.

IMPORTANT - read before trusting this for real predictions:
build_shot_features() below is a *reconstruction* of your Stage 2
pipeline, assembled from the transformer classes and column names
visible in your test suite and build guide. It is NOT copied from your
actual feature_build_runner.py / runner.py, which I haven't seen. Two
things could be off:
  1. Import paths - I've guessed where BallSpeedImputer,
     OpponentDataDropper, and ShotFilter live; adjust if wrong.
  2. Raw StatsBomb column names for carry/goalkeeper end locations,
     and the exact transformer order your real pipeline uses.

Before trusting this on a genuinely new match, validate it on a match
you already have processed: run build_shot_features() on that match's
raw events and diff the output against its rows in
shot_df_clean.parquet. If they match column-for-column, you're safe to
use this on new matches. If not, share your actual runner code and
I'll fix this to match exactly instead of guessing.
"""
import warnings

import pandas as pd
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.location_splitter import LocationSplitter
from src.features.transformers.cleaning.coordinate_rescaler import CoordinateRescaler
from src.features.transformers.cleaning.time_parser import TimeParser
from src.features.transformers.cleaning.freeze_frame_extractor import FreezeFrameExtractor
from src.features.transformers.features.geometry_features import DistToGoal, ShotAngle, InPenaltyBox
from src.features.transformers.features.trajectory_features import TrajectoryLength, TrajectoryAngle
from src.features.transformers.features.pressure_features import DistNearestDefender, DefendersIn3m
from src.features.transformers.features.full_tier_features import TeamCentroidDistance, OpenAngleGoal
from src.features.transformers.features.motion_features import BallSpeed

# TODO: confirm these actually live here - adjust the import path to
# wherever your real BallSpeedImputer/OpponentDataDropper/ShotFilter
# classes are defined in the repo.
from src.features.transformers.cleaning.ball_speed_imputer import BallSpeedImputer
from src.features.transformers.cleaning.opponent_data_dropper import OpponentDataDropper
from src.features.transformers.cleaning.shot_filter import ShotFilter

warnings.simplefilter('ignore', category=NoAuthWarning)

# Exactly which (competition_id, season_id) pairs were used for training,
# taken from loader.py's comp_dict. A competition can have some seasons
# trained on and others not - so we exclude by pair, not by whole
# competition_id, otherwise every untrained season of La Liga / World Cup /
# Champions League would be hidden from the live browser for no reason.
TRAINED_SEASONS_BY_COMPETITION = {
    11: {90, 42, 4, 1, 2, 27, 26, 25, 24, 23, 22, 21, 41, 40, 39, 38},  # La Liga
    43: {106, 3},  # World Cup
    16: {4, 1, 2, 27, 26, 25, 24, 23, 22, 21, 41, 39, 37, 44, 76, 71, 276, 277},  # Champions League
}

LOCATION_SPLIT_CONFIG = [
    ('location', 'ball_x_start', 'ball_y_start'),
    ('pass_end_location', 'pass_x_end', 'pass_y_end'),
    ('shot_end_location', 'shot_x_end', 'shot_y_end'),
    ('carry_end_location', 'carry_x_end', 'carry_y_end'),
    ('goalkeeper_end_location', 'goalkeeper_x_end', 'goalkeeper_y_end'),
]

RESCALE_CONFIG = [
    (col, axis)
    for _, x_col, y_col in LOCATION_SPLIT_CONFIG
    for col, axis in [(x_col, 'x'), (y_col, 'y')]
]


# ---------------------------------------------------------------------
# Competition / season / match browsing (live, excludes trained comps)
# ---------------------------------------------------------------------

def get_available_competitions() -> pd.DataFrame:
    """All StatsBomb (competition, season) rows EXCLUDING the exact
    (competition_id, season_id) pairs already used for training - not
    whole competitions, since e.g. only some La Liga seasons were
    trained on. Columns: competition_id, competition_name, season_id,
    season_name (standard sb.competitions() output).
    """
    comps = sb.competitions()

    def _is_untrained(row) -> bool:
        trained_seasons = TRAINED_SEASONS_BY_COMPETITION.get(row['competition_id'], set())
        return row['season_id'] not in trained_seasons

    mask = comps.apply(_is_untrained, axis=1)
    return comps[mask].copy()


def get_matches_for(competition_id: int, season_id: int) -> pd.DataFrame:
    """Matches for one competition+season. Columns include match_id,
    home_team, away_team (standard sb.matches() output).
    """
    return sb.matches(competition_id=competition_id, season_id=season_id)


# ---------------------------------------------------------------------
# Live fetch + feature pipeline for a single match
# ---------------------------------------------------------------------

def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        LocationSplitter(columns=LOCATION_SPLIT_CONFIG),
        CoordinateRescaler(columns=RESCALE_CONFIG),
        TimeParser(),
        FreezeFrameExtractor(),
        DistToGoal(),
        ShotAngle(),
        InPenaltyBox(),
        TrajectoryLength(),
        TrajectoryAngle(),
        TeamCentroidDistance(),
        OpenAngleGoal(),
        DistNearestDefender(),
        DefendersIn3m(),
        BallSpeed(),
        BallSpeedImputer(),
        OpponentDataDropper(),
        ShotFilter(),
    ])


def fetch_and_process_match(match_id: int):
    """Fetch raw events for match_id and run them through the Stage 2
    feature pipeline, producing shot-level features ready for the model.

    Returns (shot_features_df, display_info_df):
      - shot_features_df: engineered features, one row per shot.
      - display_info_df: id/player/minute/second for those same shots,
        for showing in the UI only - never fed to the model.
    """
    events_df = sb.events(match_id=match_id)
    events_df['match_id'] = match_id

    pipeline = build_feature_pipeline()
    shot_features_df = pipeline.transform(events_df)

    # None of the feature transformers above derive whether a shot went
    # in - that column (ends_in_goal) only ever existed in the
    # pre-processed training parquet. For a live, never-seen-before match
    # we derive it straight from StatsBomb's shot_outcome field and merge
    # it in the same way as player/minute/second below, since app.py's
    # `shots = shot_features.merge(display_info, on='id', how='left')`
    # pulls it in for free without touching the model-facing feature set.
    shots_raw = events_df.loc[events_df['type'] == 'Shot']
    display_info_df = shots_raw[['id', 'player', 'minute', 'second']].copy()
    display_info_df['ends_in_goal'] = (shots_raw['shot_outcome'] == 'Goal').astype(int)

    return shot_features_df, display_info_df