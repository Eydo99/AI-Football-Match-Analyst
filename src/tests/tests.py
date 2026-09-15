"""
Unit tests for src/features/transformers/*.

Run with:  pytest tests/test_transformers.py -v

Each test class targets one Transformer. Tests use small, hand-built
DataFrames (not real match data) so every edge case is explicit and
the expected output is known in advance -- including the exact bugs
we already caught once during development, so they can't silently
come back.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.base import Transformer
from src.features.transformers.features.motion_features import BallSpeed
from src.features.transformers.cleaning.column_pruner import ColumnPruner
from src.features.transformers.cleaning.boolean_encoder import BooleanEncoder
from src.features.transformers.cleaning.location_splitter import LocationSplitter
from src.features.transformers.cleaning.coordinate_rescaler import CoordinateRescaler
from src.features.transformers.cleaning.time_parser import TimeParser
from src.features.utils import safe_extract
from src.features.transformers.cleaning.freeze_frame_extractor import FreezeFrameExtractor
from src.features.pipeline import Pipeline
from src.features.transformers.features.full_tier_features import TeamCentroidDistance
from src.features.transformers.features.full_tier_features import OpenAngleGoal
from src.features.transformers.features.geometry_features import DistToGoal, ShotAngle, InPenaltyBox
from src.features.transformers.features.trajectory_features import TrajectoryLength, TrajectoryAngle
from src.features.transformers.features.pressure_features import (
    DistNearestDefender, DefendersIn3m,
)
from src.features.transformers.cleaning.ball_speed_imputer import BallSpeedImputer
from src.features.transformers.cleaning.opponent_data_dropper import OpponentDataDropper
from src.features.transformers.cleaning.shot_filter import ShotFilter

# ---------------------------------------------------------------------
# safe_extract
# ---------------------------------------------------------------------

class TestSafeExtract:
    def test_extracts_value_from_list(self):
        assert safe_extract([61.2, 40.5], 0) == 61.2
        assert safe_extract([61.2, 40.5], 1) == 40.5

    def test_extracts_value_from_numpy_array(self):
        arr = np.array([61.2, 40.5])
        assert safe_extract(arr, 0) == 61.2
        assert safe_extract(arr, 1) == 40.5

    def test_returns_nan_for_none(self):
        assert pd.isna(safe_extract(None, 0))

    def test_returns_nan_for_float_nan_without_crashing(self):
        # This is the exact case that used to crash with
        # "truth value of an array is ambiguous" when the isinstance
        # check ran after pd.isna() instead of before it.
        assert pd.isna(safe_extract(np.nan, 0))

    def test_returns_nan_for_index_out_of_range(self):
        assert pd.isna(safe_extract([61.2], 1))

    def test_returns_nan_for_non_array_like_scalar(self):
        assert pd.isna(safe_extract(42, 0))


# ---------------------------------------------------------------------
# ColumnPruner
# ---------------------------------------------------------------------

class TestColumnPruner:
    def test_drops_specified_columns(self):
        df = pd.DataFrame({'keep': [1, 2], 'drop_me': [3, 4]})
        result = ColumnPruner(columns=['drop_me']).transform(df)
        assert 'drop_me' not in result.columns
        assert 'keep' in result.columns

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'keep': [1, 2], 'drop_me': [3, 4]})
        ColumnPruner(columns=['drop_me']).transform(df)
        assert 'drop_me' in df.columns  # original untouched

    def test_missing_column_is_ignored_not_raised(self):
        df = pd.DataFrame({'keep': [1, 2]})
        # column list includes a column that doesn't exist in df
        result = ColumnPruner(columns=['keep', 'does_not_exist']).transform(df)
        assert list(result.columns) == []

    def test_default_columns_used_when_none_passed(self):
        df = pd.DataFrame({'tactics': [1], 'keep': [2]})
        result = ColumnPruner().transform(df)
        assert 'tactics' not in result.columns
        assert 'keep' in result.columns


# ---------------------------------------------------------------------
# BooleanEncoder
# ---------------------------------------------------------------------

class TestBooleanEncoder:
    def test_true_becomes_1(self):
        df = pd.DataFrame({'flag': [True, np.nan, True]})
        result = BooleanEncoder(columns=['flag']).transform(df)
        assert result['flag'].tolist() == [1, 0, 1]

    def test_output_dtype_is_int(self):
        df = pd.DataFrame({'flag': [True, np.nan]})
        result = BooleanEncoder(columns=['flag']).transform(df)
        assert result['flag'].dtype == np.int64 or result['flag'].dtype == int

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'flag': [True, np.nan]})
        BooleanEncoder(columns=['flag']).transform(df)
        assert df['flag'].isna().sum() == 1  # original still has NaN

    def test_multiple_columns_all_encoded(self):
        df = pd.DataFrame({
            'flag_a': [True, np.nan],
            'flag_b': [np.nan, True],
        })
        result = BooleanEncoder(columns=['flag_a', 'flag_b']).transform(df)
        assert result['flag_a'].tolist() == [1, 0]
        assert result['flag_b'].tolist() == [0, 1]

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'flag_a': [True]})
        with pytest.raises(KeyError):
            BooleanEncoder(columns=['flag_a', 'does_not_exist']).transform(df)


# ---------------------------------------------------------------------
# LocationSplitter
# ---------------------------------------------------------------------

class TestLocationSplitter:
    CONFIG = [('location', 'ball_x_start', 'ball_y_start')]

    def test_splits_list_column_into_x_y(self):
        df = pd.DataFrame({'location': [[61.2, 40.5], [10.0, 20.0]]})
        result = LocationSplitter(columns=self.CONFIG).transform(df)
        assert result['ball_x_start'].tolist() == [61.2, 10.0]
        assert result['ball_y_start'].tolist() == [40.5, 20.0]

    def test_handles_numpy_array_values(self):
        df = pd.DataFrame({'location': [np.array([61.2, 40.5])]})
        result = LocationSplitter(columns=self.CONFIG).transform(df)
        assert result['ball_x_start'].iloc[0] == 61.2
        assert result['ball_y_start'].iloc[0] == 40.5

    def test_nan_location_produces_nan_x_and_y(self):
        df = pd.DataFrame({'location': [np.nan, [5.0, 6.0]]})
        result = LocationSplitter(columns=self.CONFIG).transform(df)
        assert pd.isna(result['ball_x_start'].iloc[0])
        assert pd.isna(result['ball_y_start'].iloc[0])
        assert result['ball_x_start'].iloc[1] == 5.0

    def test_source_column_is_dropped(self):
        df = pd.DataFrame({'location': [[1.0, 2.0]]})
        result = LocationSplitter(columns=self.CONFIG).transform(df)
        assert 'location' not in result.columns

    def test_multiple_column_configs_all_processed(self):
        df = pd.DataFrame({
            'location': [[1.0, 2.0]],
            'pass_end_location': [[3.0, 4.0]],
        })
        config = [
            ('location', 'ball_x_start', 'ball_y_start'),
            ('pass_end_location', 'pass_x_end', 'pass_y_end'),
        ]
        result = LocationSplitter(columns=config).transform(df)
        assert result['ball_x_start'].iloc[0] == 1.0
        assert result['pass_x_end'].iloc[0] == 3.0
        assert 'location' not in result.columns
        assert 'pass_end_location' not in result.columns


# ---------------------------------------------------------------------
# CoordinateRescaler
# ---------------------------------------------------------------------

class TestCoordinateRescaler:
    def test_x_column_scaled_by_105_over_120(self):
        df = pd.DataFrame({'ball_x_start': [120.0]})
        config = [('ball_x_start', 'x')]
        result = CoordinateRescaler(columns=config).transform(df)
        assert result['ball_x_start'].iloc[0] == pytest.approx(105.0)

    def test_y_column_scaled_by_68_over_80(self):
        df = pd.DataFrame({'ball_y_start': [80.0]})
        config = [('ball_y_start', 'y')]
        result = CoordinateRescaler(columns=config).transform(df)
        assert result['ball_y_start'].iloc[0] == pytest.approx(68.0)

    def test_uses_constructor_factors_not_module_default(self):
        # Regression test for the bug where transform() referenced the
        # module-level RESCALING_FACTORS constant instead of self.factors,
        # silently ignoring whatever was passed into the constructor.
        df = pd.DataFrame({'ball_x_start': [10.0]})
        config = [('ball_x_start', 'x')]
        custom_factors = {'x': 2.0, 'y': 1.0}
        result = CoordinateRescaler(columns=config, factors=custom_factors).transform(df)
        assert result['ball_x_start'].iloc[0] == pytest.approx(20.0)

    def test_nan_value_stays_nan_after_scaling(self):
        df = pd.DataFrame({'ball_x_start': [np.nan]})
        config = [('ball_x_start', 'x')]
        result = CoordinateRescaler(columns=config).transform(df)
        assert pd.isna(result['ball_x_start'].iloc[0])

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [10.0]})
        config = [('ball_x_start', 'x'), ('does_not_exist', 'y')]
        with pytest.raises(KeyError):
            CoordinateRescaler(columns=config).transform(df)


# ---------------------------------------------------------------------
# TimeParser
# ---------------------------------------------------------------------

class TestTimeParser:
    def test_parses_timestamp_to_seconds(self):
        df = pd.DataFrame({'timestamp': ['00:23:15.123']})
        result = TimeParser().transform(df)
        assert result['event_time_seconds'].iloc[0] == pytest.approx(1395.123)

    def test_zero_timestamp_parses_to_zero(self):
        df = pd.DataFrame({'timestamp': ['00:00:00.000']})
        result = TimeParser().transform(df)
        assert result['event_time_seconds'].iloc[0] == pytest.approx(0.0)

    def test_source_column_dropped(self):
        df = pd.DataFrame({'timestamp': ['00:00:01.000']})
        result = TimeParser().transform(df)
        assert 'timestamp' not in result.columns

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'not_timestamp': ['00:00:01.000']})
        with pytest.raises(KeyError):
            TimeParser().transform(df)

    def test_matches_minute_second_ground_truth(self):
        # Cross-check against the same logic used to validate this
        # manually: minute*60 + second should be <= event_time_seconds
        # and the difference should be < 1 (just the sub-second part).
        df = pd.DataFrame({'timestamp': ['00:05:10.500']})
        result = TimeParser().transform(df)
        minute, second = 5, 10
        expected_floor = minute * 60 + second
        actual = result['event_time_seconds'].iloc[0]
        assert actual >= expected_floor
        assert actual - expected_floor < 1

    # ---------------------------------------------------------------------
    # FreezeFrameExtractor
    # ---------------------------------------------------------------------

    class TestFreezeFrameExtractor:
        def test_excludes_teammate_entries(self):
            freeze_frame = [
                {'teammate': True, 'location': [50.0, 40.0]},
            ]
            df = pd.DataFrame({'shot_freeze_frame': [freeze_frame]})
            result = FreezeFrameExtractor().transform(df)
            assert result['opponent_locations'].iloc[0] == []

        def test_includes_opponent_entries_rescaled(self):
            freeze_frame = [
                {'teammate': False, 'location': [120.0, 80.0]},
            ]
            df = pd.DataFrame({'shot_freeze_frame': [freeze_frame]})
            result = FreezeFrameExtractor().transform(df)
            assert result['opponent_locations'].iloc[0] == [pytest.approx((105.0, 68.0))]

        def test_mixed_teammates_and_opponents(self):
            freeze_frame = [
                {'teammate': True, 'location': [10.0, 10.0]},
                {'teammate': False, 'location': [60.0, 40.0]},
                {'teammate': False, 'location': [30.0, 20.0]},
            ]
            df = pd.DataFrame({'shot_freeze_frame': [freeze_frame]})
            result = FreezeFrameExtractor().transform(df)
            assert len(result['opponent_locations'].iloc[0]) == 2

        def test_missing_teammate_key_defaults_to_teammate(self):
            # entry.get('teammate', True) -- an entry with no 'teammate' key
            # at all should default to True (teammate) and be excluded.
            freeze_frame = [{'location': [50.0, 40.0]}]
            df = pd.DataFrame({'shot_freeze_frame': [freeze_frame]})
            result = FreezeFrameExtractor().transform(df)
            assert result['opponent_locations'].iloc[0] == []

        def test_none_freeze_frame_produces_empty_list(self):
            df = pd.DataFrame({'shot_freeze_frame': [None]})
            result = FreezeFrameExtractor().transform(df)
            assert result['opponent_locations'].iloc[0] == []

        def test_non_list_freeze_frame_produces_empty_list(self):
            df = pd.DataFrame({'shot_freeze_frame': [np.nan]})
            result = FreezeFrameExtractor().transform(df)
            assert result['opponent_locations'].iloc[0] == []

        def test_entry_missing_location_produces_nan_pair(self):
            freeze_frame = [{'teammate': False}]
            df = pd.DataFrame({'shot_freeze_frame': [freeze_frame]})
            result = FreezeFrameExtractor().transform(df)
            x, y = result['opponent_locations'].iloc[0][0]
            assert pd.isna(x)
            assert pd.isna(y)

        def test_source_column_dropped(self):
            df = pd.DataFrame({'shot_freeze_frame': [[]]})
            result = FreezeFrameExtractor().transform(df)
            assert 'shot_freeze_frame' not in result.columns

        def test_missing_column_raises_keyerror(self):
            df = pd.DataFrame({'not_freeze_frame': [[]]})
            with pytest.raises(KeyError):
                FreezeFrameExtractor().transform(df)

        def test_default_column_used_when_none_passed(self):
            # Regression test for the constructor bug where self.column was
            # set to EXTRACTED_COL and then immediately overwritten back to
            # None by the unconditional `self.column = column` line.
            df = pd.DataFrame({'shot_freeze_frame': [[]]})
            result = FreezeFrameExtractor(column=None).transform(df)
            assert 'opponent_locations' in result.columns

# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

class _AddOneTransformer(Transformer):
    """Minimal dummy transformer for isolating Pipeline behavior from
    any real transformer's own logic."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['value'] = result['value'] + 1
        return result


class _MultiplyByTwoTransformer(Transformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['value'] = result['value'] * 2
        return result


class _RaisingTransformer(Transformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise KeyError("simulated failure mid-pipeline")


class TestPipeline:
    def test_single_transformer_applied(self):
        df = pd.DataFrame({'value': [1, 2, 3]})
        result = Pipeline([_AddOneTransformer()]).transform(df)
        assert result['value'].tolist() == [2, 3, 4]

    def test_multiple_transformers_applied_in_order(self):
        # (1 + 1) * 2 = 4, not 1 * 2 + 1 = 3 -- order must be respected
        df = pd.DataFrame({'value': [1]})
        result = Pipeline([_AddOneTransformer(), _MultiplyByTwoTransformer()]).transform(df)
        assert result['value'].iloc[0] == 4

    def test_reversed_order_gives_different_result(self):
        df = pd.DataFrame({'value': [1]})
        result = Pipeline([_MultiplyByTwoTransformer(), _AddOneTransformer()]).transform(df)
        assert result['value'].iloc[0] == 3

    def test_empty_pipeline_returns_df_unchanged(self):
        df = pd.DataFrame({'value': [1, 2, 3]})
        result = Pipeline([]).transform(df)
        assert result['value'].tolist() == [1, 2, 3]

    def test_failure_midway_propagates_and_stops(self):
        df = pd.DataFrame({'value': [1]})
        pipeline = Pipeline([
            _AddOneTransformer(),
            _RaisingTransformer(),
            _MultiplyByTwoTransformer(),
        ])
        with pytest.raises(KeyError):
            pipeline.transform(df)

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'value': [1, 2, 3]})
        Pipeline([_AddOneTransformer()]).transform(df)
        assert df['value'].tolist() == [1, 2, 3]

    def test_works_with_real_transformers_chained(self):
        # Sanity check with two real transformers, not just dummies.
        df = pd.DataFrame({
            'keep': [1],
            'flag': [True],
            'tactics': [99],
        })
        pipeline = Pipeline([
            ColumnPruner(columns=['tactics']),
            BooleanEncoder(columns=['flag']),
        ])
        result = pipeline.transform(df)
        assert 'tactics' not in result.columns
        assert result['flag'].iloc[0] == 1

# ---------------------------------------------------------------------
# TeamCentroidDistance
# ---------------------------------------------------------------------

class TestTeamCentroidDistance:
    
    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'opponent_locations': [[(60.0, 30.0)]], 'ball_x_start': [50.0]})
        with pytest.raises(KeyError):
            TeamCentroidDistance().transform(df)

    def test_computes_distance_to_mean_opponent_position(self):
        # opponents at (60,30),(65,35),(58,40) -> centroid (61.0, 35.0)
        # ball at (50,34) -> distance = sqrt((61-50)^2 + (35-34)^2) ~= 11.045
        df = pd.DataFrame({
            'opponent_locations': [[(60.0, 30.0), (65.0, 35.0), (58.0, 40.0)]],
            'ball_x_start': [50.0],
            'ball_y_start': [34.0],
        })
        result = TeamCentroidDistance().transform(df)
        assert result['team_centroid_distance'].iloc[0] == pytest.approx(11.045, abs=0.01)

    def test_empty_opponent_list_produces_nan(self):
        df = pd.DataFrame({
            'opponent_locations': [[]],
            'ball_x_start': [50.0],
            'ball_y_start': [34.0],
        })
        result = TeamCentroidDistance().transform(df)
        assert pd.isna(result['team_centroid_distance'].iloc[0])

    def test_none_opponent_locations_produces_nan(self):
        df = pd.DataFrame({
            'opponent_locations': [None],
            'ball_x_start': [50.0],
            'ball_y_start': [34.0],
        })
        result = TeamCentroidDistance().transform(df)
        assert pd.isna(result['team_centroid_distance'].iloc[0])

    def test_nan_ball_position_produces_nan(self):
        # pd.notna check must catch float NaN, not just None
        df = pd.DataFrame({
            'opponent_locations': [[(60.0, 30.0)]],
            'ball_x_start': [np.nan],
            'ball_y_start': [34.0],
        })
        result = TeamCentroidDistance().transform(df)
        assert pd.isna(result['team_centroid_distance'].iloc[0])

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({
            'opponent_locations': [[(60.0, 30.0)]],
            'ball_x_start': [50.0],
            'ball_y_start': [34.0],
        })
        TeamCentroidDistance().transform(df)
        assert 'team_centroid_distance' not in df.columns


# ---------------------------------------------------------------------
# OpenAngleGoal
# ---------------------------------------------------------------------

class TestOpenAngleGoal:

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'opponent_locations': [[(60.0, 30.0)]], 'ball_x_start': [50.0] , 'ball_y_start': [34.0]})
        with pytest.raises(KeyError):
            OpenAngleGoal().transform(df)

    def test_non_shot_row_produces_nan(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'opponent_locations': [[(100.0, 34.0)]],
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal().transform(df)
        assert pd.isna(result['open_angle_goal'].iloc[0])


    def test_defender_outside_window_is_ignored(self):
        # Defender B from the worked example: (98, 36) -> angle ~33.7 deg,
        # well outside the ~[-20.1, 20.1] deg window -- should block nothing.
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [[(98.0, 36.0)]],
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal().transform(df)
        left = np.arctan2(30.34 - 34.0, 105 - 95.0)
        right = np.arctan2(37.66 - 34.0, 105 - 95.0)
        assert result['open_angle_goal'].iloc[0] == pytest.approx(right - left)

    def test_defender_behind_ball_is_ignored(self):
        # x < ball_x_start -- physically cannot block a shot toward x=105.
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [[(90.0, 34.0)]],
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal().transform(df)
        left = np.arctan2(30.34 - 34.0, 105 - 95.0)
        right = np.arctan2(37.66 - 34.0, 105 - 95.0)
        assert result['open_angle_goal'].iloc[0] == pytest.approx(right - left)

    def test_single_defender_blocks_expected_wedge(self):
        # Defender A from the worked example: (100, 34), distance 5 from
        # ball, player_width 0.5 -> half_angle = atan(0.5/5).
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [[(100.0, 34.0)]],
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal(player_width=0.5).transform(df)
        left = np.arctan2(30.34 - 34.0, 105 - 95.0)
        right = np.arctan2(37.66 - 34.0, 105 - 95.0)
        half_angle = np.arctan2(0.5, 5.0)
        expected = (right - left) - (2 * half_angle)
        assert result['open_angle_goal'].iloc[0] == pytest.approx(expected)

    def test_overlapping_defenders_are_merged_not_double_counted(self):
        # Regression test for the bug where current_angle was pre-seeded
        # from opponent_angles[0] (unsorted order) instead of None, which
        # could duplicate or miscount an interval. Two defenders standing
        # close together (overlapping wedges) must only have their union
        # subtracted once, not the sum of both individual wedges.
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [[(100.0, 34.5), (100.0, 33.5)]],
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal(player_width=2.0).transform(df)  # wide wedges force overlap
        left = np.arctan2(30.34 - 34.0, 105 - 95.0)
        right = np.arctan2(37.66 - 34.0, 105 - 95.0)
        # naive (wrong) double-counted sum would give a smaller open angle
        # than the correct merged result -- assert we get the larger (correct) one
        d1 = np.sqrt((100.0 - 95.0) ** 2 + (34.5 - 34.0) ** 2)
        d2 = np.sqrt((100.0 - 95.0) ** 2 + (33.5 - 34.0) ** 2)
        half1 = np.arctan2(2.0, d1)
        half2 = np.arctan2(2.0, d2)
        naive_double_counted = (right - left) - (2 * half1 + 2 * half2)
        assert result['open_angle_goal'].iloc[0] > naive_double_counted

    def test_wedge_extending_past_post_is_clipped(self):
        # A defender whose wedge would extend past the right post must be
        # clipped to the post angle, not allowed to "block" empty space
        # outside the goal window.
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [[(96.0, 40.0)]],  # near-window-edge, wide wedge
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal(player_width=5.0).transform(df)
        left = np.arctan2(30.34 - 34.0, 105 - 95.0)
        right = np.arctan2(37.66 - 34.0, 105 - 95.0)
        assert result['open_angle_goal'].iloc[0] >= 0.0
        assert result['open_angle_goal'].iloc[0] <= (right - left)

    def test_open_angle_never_negative(self):
        # Many close, wide-wedge defenders could sum past the full window --
        # result must clamp at 0, never go negative.
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [
                [(96.0, 34.0), (97.0, 35.0), (98.0, 33.0), (99.0, 34.5)]
            ],
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal(player_width=5.0).transform(df)
        assert result['open_angle_goal'].iloc[0] >= 0.0

    def test_missing_ball_position_produces_nan(self):
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [[(100.0, 34.0)]],
            'ball_x_start': [np.nan],
            'ball_y_start': [34.0],
        })
        result = OpenAngleGoal().transform(df)
        assert pd.isna(result['open_angle_goal'].iloc[0])

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({
            'type': ['Shot'],
            'opponent_locations': [[(100.0, 34.0)]],
            'ball_x_start': [95.0],
            'ball_y_start': [34.0],
        })
        OpenAngleGoal().transform(df)
        assert 'open_angle_goal' not in df.columns

# ---------------------------------------------------------------------
# DistToGoal
# ---------------------------------------------------------------------

class TestDistToGoal:
    def test_ball_at_goal_center_distance_zero(self):
        df = pd.DataFrame({'ball_x_start': [105.0], 'ball_y_start': [34.0]})
        result = DistToGoal().transform(df)
        assert result['dist_to_goal'].iloc[0] == pytest.approx(0.0)

    def test_known_distance_from_center_spot(self):
        # Center spot is (52.5, 34) -- distance to goal is a known value
        df = pd.DataFrame({'ball_x_start': [52.5], 'ball_y_start': [34.0]})
        result = DistToGoal().transform(df)
        assert result['dist_to_goal'].iloc[0] == pytest.approx(52.5)

    def test_uses_custom_columns(self):
        df = pd.DataFrame({'custom_x': [105.0], 'custom_y': [34.0]})
        result = DistToGoal(x_col='custom_x', y_col='custom_y').transform(df)
        assert result['dist_to_goal'].iloc[0] == pytest.approx(0.0)

    def test_nan_input_produces_nan_output(self):
        df = pd.DataFrame({'ball_x_start': [np.nan], 'ball_y_start': [40.0]})
        result = DistToGoal().transform(df)
        assert pd.isna(result['dist_to_goal'].iloc[0])

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [50.0]})
        with pytest.raises(KeyError):
            DistToGoal().transform(df)

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'ball_x_start': [50.0], 'ball_y_start': [40.0]})
        DistToGoal().transform(df)
        assert 'dist_to_goal' not in df.columns


# ---------------------------------------------------------------------
# ShotAngle
# ---------------------------------------------------------------------

class TestShotAngle:
    def test_ball_on_goal_line_between_posts_is_pi(self):
        # Standing right on the goal line, between the posts, the angle
        # subtended should be the full straight angle: pi radians.
        df = pd.DataFrame({'ball_x_start': [105.0], 'ball_y_start': [34.0]})
        result = ShotAngle().transform(df)
        assert result['shot_angle'].iloc[0] == pytest.approx(np.pi, abs=1e-6)

    def test_angle_decreases_further_from_goal(self):
        close_df = pd.DataFrame({'ball_x_start': [95.0], 'ball_y_start': [34.0]})
        far_df = pd.DataFrame({'ball_x_start': [20.0], 'ball_y_start': [34.0]})
        close_angle = ShotAngle().transform(close_df)['shot_angle'].iloc[0]
        far_angle = ShotAngle().transform(far_df)['shot_angle'].iloc[0]
        assert close_angle > far_angle

    def test_angle_is_symmetric_about_goal_center_y(self):
        # Same x, mirrored y around the goal's center (y=34) should give
        # the same angle magnitude.
        above_df = pd.DataFrame({'ball_x_start': [70.0], 'ball_y_start': [44.0]})
        below_df = pd.DataFrame({'ball_x_start': [70.0], 'ball_y_start': [24.0]})
        above_angle = ShotAngle().transform(above_df)['shot_angle'].iloc[0]
        below_angle = ShotAngle().transform(below_df)['shot_angle'].iloc[0]
        assert above_angle == pytest.approx(below_angle)

    def test_angle_is_nonnegative(self):
        df = pd.DataFrame({'ball_x_start': [10.0], 'ball_y_start': [60.0]})
        result = ShotAngle().transform(df)
        assert result['shot_angle'].iloc[0] >= 0

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [50.0]})
        with pytest.raises(KeyError):
            ShotAngle().transform(df)

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'ball_x_start': [50.0], 'ball_y_start': [40.0]})
        ShotAngle().transform(df)
        assert 'shot_angle' not in df.columns


# ---------------------------------------------------------------------
# InPenaltyBox
# ---------------------------------------------------------------------

class TestInPenaltyBox:
    def test_point_inside_box_returns_1(self):
        df = pd.DataFrame({'ball_x_start': [95.0], 'ball_y_start': [34.0]})
        result = InPenaltyBox().transform(df)
        assert result['in_penalty_box'].iloc[0] == 1

    def test_point_outside_box_x_returns_0(self):
        df = pd.DataFrame({'ball_x_start': [80.0], 'ball_y_start': [34.0]})
        result = InPenaltyBox().transform(df)
        assert result['in_penalty_box'].iloc[0] == 0

    def test_point_outside_box_y_returns_0(self):
        df = pd.DataFrame({'ball_x_start': [95.0], 'ball_y_start': [60.0]})
        result = InPenaltyBox().transform(df)
        assert result['in_penalty_box'].iloc[0] == 0

    def test_boundary_values_are_inclusive(self):
        df = pd.DataFrame({
            'ball_x_start': [88.5, 88.5],
            'ball_y_start': [13.84, 54.16],
        })
        result = InPenaltyBox().transform(df)
        assert result['in_penalty_box'].tolist() == [1, 1]

    def test_nan_input_returns_0_not_nan(self):
        df = pd.DataFrame({'ball_x_start': [np.nan], 'ball_y_start': [34.0]})
        result = InPenaltyBox().transform(df)
        assert result['in_penalty_box'].iloc[0] == 0

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [50.0]})
        with pytest.raises(KeyError):
            InPenaltyBox().transform(df)

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'ball_x_start': [50.0], 'ball_y_start': [40.0]})
        InPenaltyBox().transform(df)
        assert 'in_penalty_box' not in df.columns


# ---------------------------------------------------------------------
# TrajectoryLength
# ---------------------------------------------------------------------

class TestTrajectoryLength:
    def test_pass_uses_pass_end_location(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'pass_x_end': [3.0], 'pass_y_end': [4.0],
            'shot_x_end': [999.0], 'shot_y_end': [999.0],
        })
        result = TrajectoryLength().transform(df)
        assert result['trajectory_length'].iloc[0] == pytest.approx(5.0)

    def test_shot_uses_shot_end_location(self):
        df = pd.DataFrame({
            'type': ['Shot'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'shot_x_end': [6.0], 'shot_y_end': [8.0],
        })
        result = TrajectoryLength().transform(df)
        assert result['trajectory_length'].iloc[0] == pytest.approx(10.0)

    def test_carry_uses_carry_end_location(self):
        df = pd.DataFrame({
            'type': ['Carry'],
            'ball_x_start': [1.0], 'ball_y_start': [1.0],
            'carry_x_end': [4.0], 'carry_y_end': [5.0],
        })
        result = TrajectoryLength().transform(df)
        assert result['trajectory_length'].iloc[0] == pytest.approx(5.0)

    def test_goalkeeper_uses_goalkeeper_end_location(self):
        df = pd.DataFrame({
            'type': ['Goal Keeper'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'goalkeeper_x_end': [3.0], 'goalkeeper_y_end': [4.0],
        })
        result = TrajectoryLength().transform(df)
        assert result['trajectory_length'].iloc[0] == pytest.approx(5.0)

    def test_event_type_outside_map_produces_nan(self):
        df = pd.DataFrame({
            'type': ['Pressure'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
        })
        result = TrajectoryLength().transform(df)
        assert pd.isna(result['trajectory_length'].iloc[0])

    def test_mixed_event_types_each_use_correct_column(self):
        df = pd.DataFrame({
            'type': ['Pass', 'Shot'],
            'ball_x_start': [0.0, 0.0], 'ball_y_start': [0.0, 0.0],
            'pass_x_end': [3.0, 999.0], 'pass_y_end': [4.0, 999.0],
            'shot_x_end': [999.0, 6.0], 'shot_y_end': [999.0, 8.0],
        })
        result = TrajectoryLength().transform(df)
        assert result['trajectory_length'].tolist() == pytest.approx([5.0, 10.0])

    def test_helper_columns_not_leaked_into_output(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'pass_x_end': [3.0], 'pass_y_end': [4.0],
        })
        result = TrajectoryLength().transform(df)
        assert '_x_end' not in result.columns
        assert '_y_end' not in result.columns

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'pass_x_end': [3.0], 'pass_y_end': [4.0],
        })
        TrajectoryLength().transform(df)
        assert 'trajectory_length' not in df.columns
        assert '_x_end' not in df.columns

    def test_missing_start_column_raises_keyerror(self):
        df = pd.DataFrame({'type': ['Pass'], 'ball_y_start': [0.0]})
        with pytest.raises(KeyError):
            TrajectoryLength().transform(df)

    def test_missing_type_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [0.0], 'ball_y_start': [0.0]})
        with pytest.raises(KeyError):
            TrajectoryLength().transform(df)

    def test_missing_end_column_for_present_type_raises_keyerror(self):
        # 'Pass' rows exist but pass_x_end isn't in the df at all
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
        })
        with pytest.raises(KeyError):
            TrajectoryLength().transform(df)


# ---------------------------------------------------------------------
# TrajectoryAngle
# ---------------------------------------------------------------------

class TestTrajectoryAngle:
    def test_pass_uses_pass_end_location(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'pass_x_end': [1.0], 'pass_y_end': [0.0],
        })
        result = TrajectoryAngle().transform(df)
        assert result['trajectory_angle'].iloc[0] == pytest.approx(0.0)

    def test_shot_uses_shot_end_location(self):
        df = pd.DataFrame({
            'type': ['Shot'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'shot_x_end': [0.0], 'shot_y_end': [1.0],
        })
        result = TrajectoryAngle().transform(df)
        assert result['trajectory_angle'].iloc[0] == pytest.approx(np.pi / 2)

    def test_event_type_outside_map_produces_nan(self):
        df = pd.DataFrame({
            'type': ['Foul Won'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
        })
        result = TrajectoryAngle().transform(df)
        assert pd.isna(result['trajectory_angle'].iloc[0])

    def test_angle_range_is_within_pi(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'pass_x_end': [-1.0], 'pass_y_end': [-1.0],
        })
        result = TrajectoryAngle().transform(df)
        angle = result['trajectory_angle'].iloc[0]
        assert -np.pi <= angle <= np.pi

    def test_helper_columns_not_leaked_into_output(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'pass_x_end': [1.0], 'pass_y_end': [0.0],
        })
        result = TrajectoryAngle().transform(df)
        assert '_x_end' not in result.columns
        assert '_y_end' not in result.columns

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({
            'type': ['Pass'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'pass_x_end': [1.0], 'pass_y_end': [0.0],
        })
        TrajectoryAngle().transform(df)
        assert 'trajectory_angle' not in df.columns
        assert '_x_end' not in df.columns

    def test_missing_type_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [0.0], 'ball_y_start': [0.0]})
        with pytest.raises(KeyError):
            TrajectoryAngle().transform(df)

    def test_missing_end_column_for_present_type_raises_keyerror(self):
        df = pd.DataFrame({
            'type': ['Shot'],
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
        })
        with pytest.raises(KeyError):
            TrajectoryAngle().transform(df)
# ---------------------------------------------------------------------
# Pressure features
# ---------------------------------------------------------------------



def _pressure_frame(opponents, x=0.0, y=0.0):
    return pd.DataFrame({
        'ball_x_start': [x],
        'ball_y_start': [y],
        'opponent_locations': [opponents],
    })


class TestPressureFeatures:
    def test_known_distances_and_inclusive_boundary(self):
        df = _pressure_frame([(3, 4), (0, 3), (0, 2), (0, 3.000001)])
        assert DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0] == 2
        assert DefendersIn3m().transform(df)['defenders_in_3m'].iloc[0] == 2

    def test_diagonal_boundary_and_custom_radius(self):
        df = _pressure_frame([(3, 4), (-3, -4), (5.000001, 0)])
        assert DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0] == 5
        assert DefendersIn3m(radius=5).transform(df)['defenders_in_3m'].iloc[0] == 2

    def test_zero_radius_and_colocated_players(self):
        df = _pressure_frame([(0, 0), (0, 0), (0.000001, 0)])
        assert DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0] == 0
        assert DefendersIn3m(radius=0).transform(df)['defenders_in_3m'].iloc[0] == 2

    @pytest.mark.parametrize('opponents', [
        [], (), np.empty((0, 2)), None, np.nan, pd.NA, 42, 'missing',
        np.array(2), [(np.nan, 0), (0, np.inf)],
    ])
    def test_no_known_opponents(self, opponents):
        df = _pressure_frame(opponents)
        assert pd.isna(DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0])
        assert DefendersIn3m().transform(df)['defenders_in_3m'].iloc[0] == 0

    @pytest.mark.parametrize('bad', [None, np.nan, pd.NA, np.inf, -np.inf, 'bad'])
    @pytest.mark.parametrize('axis', ['ball_x_start', 'ball_y_start'])
    def test_unknown_ball_position_is_not_zero_pressure(self, bad, axis):
        df = _pressure_frame([(0, 0)])
        df[axis] = [bad]
        assert pd.isna(DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0])
        assert pd.isna(DefendersIn3m().transform(df)['defenders_in_3m'].iloc[0])

    @pytest.mark.parametrize('container', [list, tuple, np.array])
    def test_supported_opponent_containers(self, container):
        df = _pressure_frame(container([(2, 0), (4, 0)]))
        assert DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0] == 2
        assert DefendersIn3m().transform(df)['defenders_in_3m'].iloc[0] == 1

    def test_bad_entries_do_not_hide_valid_opponents(self):
        df = _pressure_frame([
            None, [], [1], [1, 2, 3], {'x': 0}, 'bad', (pd.NA, 0),
            (np.nan, 0), (np.inf, 0), ('bad', 0), (0, 2), (3, 4),
        ])
        assert DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0] == 2
        assert DefendersIn3m().transform(df)['defenders_in_3m'].iloc[0] == 1

    @pytest.mark.parametrize('factory,output', [
        (DistNearestDefender, 'dist_nearest_defender'),
        (DefendersIn3m, 'defenders_in_3m'),
    ])
    def test_custom_columns(self, factory, output):
        df = pd.DataFrame({'x': [0], 'y': [0], 'opponents': [[(1, 0)]]})
        result = factory(x_col='x', y_col='y', opponents_col='opponents').transform(df)
        assert result[output].iloc[0] == 1

    @pytest.mark.parametrize('factory', [DistNearestDefender, DefendersIn3m])
    @pytest.mark.parametrize('column', ['ball_x_start', 'ball_y_start', 'opponent_locations'])
    def test_missing_required_column(self, factory, column):
        df = _pressure_frame([]).drop(columns=column)
        with pytest.raises(KeyError):
            factory().transform(df)

    @pytest.mark.parametrize('factory,output', [
        (DistNearestDefender, 'dist_nearest_defender'),
        (DefendersIn3m, 'defenders_in_3m'),
    ])
    def test_empty_dataframe_preserves_schema_and_index(self, factory, output):
        df = _pressure_frame([]).iloc[:0]
        result = factory().transform(df)
        assert result.empty
        assert list(result.columns) == list(df.columns) + [output]
        assert result[output].dtype == np.float64
        pd.testing.assert_index_equal(result.index, df.index)

    def test_preserves_input_nested_values_and_duplicate_index(self):
        import copy
        df = pd.DataFrame({
            'ball_x_start': [0, 10, 0],
            'ball_y_start': [0, 0, 0],
            'opponent_locations': [[(1, 0)], [(14, 0)], []],
            'unrelated': ['a', 'b', 'c'],
        }, index=[7, 2, 7])
        original = copy.deepcopy(df.to_dict('list'))
        snapshot = df.copy(deep=True)
        pipeline = Pipeline([DistNearestDefender(), DefendersIn3m()])
        result = pipeline.transform(df)
        assert result['dist_nearest_defender'].iloc[:2].tolist() == [1, 4]
        assert pd.isna(result['dist_nearest_defender'].iloc[2])
        assert result['defenders_in_3m'].tolist() == [1, 0, 0]
        pd.testing.assert_frame_equal(result[df.columns], snapshot)
        pd.testing.assert_frame_equal(df, snapshot)
        assert df.to_dict('list') == original
        pd.testing.assert_frame_equal(pipeline.transform(result), result)

    @pytest.mark.parametrize('radius', [
        -1, np.nan, np.inf, -np.inf, True, np.bool_(False), [], [3], 'bad', pd.NA,
    ])
    def test_invalid_radius_rejected(self, radius):
        with pytest.raises(ValueError, match='radius'):
            DefendersIn3m(radius=radius)

    def test_full_cleaning_and_existing_features_integration(self):
        # Raw coordinates: ball -> (52.5, 34), opponent -> (54.25, 34).
        df = pd.DataFrame({
            'location': [[60, 40]],
            'shot_freeze_frame': [[
                {'teammate': True, 'location': [60, 40]},
                {'teammate': False, 'location': [62, 40], 'keeper': True},
                {'teammate': False, 'location': [70, 40]},
                {'teammate': False},
            ]],
        })
        result = Pipeline([
            LocationSplitter(columns=[('location', 'ball_x_start', 'ball_y_start')]),
            CoordinateRescaler(columns=[('ball_x_start', 'x'), ('ball_y_start', 'y')]),
            FreezeFrameExtractor(), DistToGoal(),
            DistNearestDefender(), DefendersIn3m(),
        ]).transform(df)
        assert result['dist_nearest_defender'].iloc[0] == pytest.approx(1.75)
        assert result['defenders_in_3m'].iloc[0] == 1
        assert result['dist_to_goal'].iloc[0] == pytest.approx(52.5)
        assert 'location' in df.columns
        assert 'shot_freeze_frame' in df.columns

    def test_randomized_against_independent_scalar_reference(self):
        import math
        rng = np.random.default_rng(12345)
        for _ in range(100):
            ball = rng.uniform([0, 0], [105, 68])
            opponents = rng.uniform([0, 0], [105, 68], size=(11, 2))
            radius = float(rng.uniform(0, 30))
            distances = [math.dist(ball, point) for point in opponents]
            df = _pressure_frame(opponents, *ball)
            nearest = DistNearestDefender().transform(df)['dist_nearest_defender'].iloc[0]
            count = DefendersIn3m(radius=radius).transform(df)['defenders_in_3m'].iloc[0]
            assert nearest == pytest.approx(min(distances))
            assert count == sum(distance <= radius for distance in distances)
            # Translation and opponent ordering must not change either feature.
            moved = _pressure_frame(opponents[::-1] + 17, *(ball + 17))
            assert DistNearestDefender().transform(moved)['dist_nearest_defender'].iloc[0] == pytest.approx(nearest)
            assert DefendersIn3m(radius=radius).transform(moved)['defenders_in_3m'].iloc[0] == count

# ---------------------------------------------------------------------
# motion feature
# ---------------------------------------------------------------------

class TestBallSpeed:
    @staticmethod
    def frame():
        return pd.DataFrame({
            'match_id': [1, 1, 1], 'period': [1, 1, 1],
            'event_time_seconds': [0., 2., 4.],
            'ball_x_start': [0., 3., 3.], 'ball_y_start': [0., 4., 4.],
        })

    def test_known_speed_and_stationary_ball(self):
        result = BallSpeed().transform(self.frame())
        assert pd.isna(result.ball_speed.iloc[0])
        assert result.ball_speed.iloc[1:].tolist() == [2.5, 0.]

    def test_unsorted_rows_and_duplicate_named_index(self):
        df = self.frame().iloc[[2, 0, 1]].copy()
        df.index = pd.Index([7, 2, 7], name='match_id')
        original = df.copy(deep=True)
        result = BallSpeed().transform(df)
        assert result.ball_speed.iloc[0] == 0
        assert pd.isna(result.ball_speed.iloc[1])
        assert result.ball_speed.iloc[2] == 2.5
        pd.testing.assert_frame_equal(result[df.columns], original)
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(BallSpeed().transform(result), result)

    def test_match_and_period_boundaries(self):
        df = pd.concat([self.frame()] * 3, ignore_index=True)
        df['match_id'] = [1] * 6 + [2] * 3
        df['period'] = [1] * 3 + [2] * 3 + [1] * 3
        result = BallSpeed().transform(df)
        assert result.ball_speed.iloc[[0, 3, 6]].isna().all()
        assert result.ball_speed.iloc[[1, 4, 7]].tolist() == [2.5] * 3

    def test_team_transitions_do_not_create_coordinate_jumps(self):
        df = self.frame()
        df['team'] = ['a', 'b', 'b']
        df['ball_x_start'] = [0., 100., 103.]
        result = BallSpeed().transform(df)
        assert result.ball_speed.iloc[:2].isna().all()
        assert result.ball_speed.iloc[2] == 1.5

    @pytest.mark.parametrize('team', [None, pd.NA, np.nan])
    def test_unknown_team_not_compared(self, team):
        df = self.frame()
        df['team'] = ['a', team, 'a']
        assert BallSpeed().transform(df).ball_speed.isna().all()

    def test_zero_time_and_event_index_tie_break(self):
        df = self.frame()
        df['event_time_seconds'] = [0., 0., 2.]
        df['ball_x_start'] = [0., 3., 6.]
        df['ball_y_start'] = [0., 0., 0.]
        df['index'] = [1, 2, 3]
        result = BallSpeed().transform(df.iloc[[1, 2, 0]])
        assert pd.isna(result.ball_speed.iloc[0])
        assert result.ball_speed.iloc[1] == 1.5
        assert pd.isna(result.ball_speed.iloc[2])

    @pytest.mark.parametrize('value', [None, pd.NA, np.nan, np.inf, -np.inf, 'bad'])
    @pytest.mark.parametrize('column', ['ball_x_start', 'ball_y_start'])
    def test_missing_position_breaks_adjacent_intervals(self, value, column):
        df = self.frame()
        df[column] = pd.Series([0, value, 3], dtype=object)
        assert BallSpeed().transform(df).ball_speed.isna().all()

    @pytest.mark.parametrize('value', [None, pd.NA, np.nan, np.inf, 'bad'])
    def test_missing_time_is_unknown(self, value):
        df = self.frame()
        df['event_time_seconds'] = pd.Series([0, 2, value], dtype=object)
        result = BallSpeed().transform(df)
        assert pd.isna(result.ball_speed.iloc[2])
        assert result.ball_speed.iloc[1] == 2.5

    @pytest.mark.parametrize('column', ['match_id', 'period'])
    def test_missing_group_values_are_not_combined(self, column):
        df = self.frame()
        df[column] = [None, None, None]
        assert BallSpeed().transform(df).ball_speed.isna().all()

    @pytest.mark.parametrize('column', [
        'match_id', 'period', 'event_time_seconds', 'ball_x_start', 'ball_y_start',
    ])
    def test_missing_required_column(self, column):
        with pytest.raises(KeyError):
            BallSpeed().transform(self.frame().drop(columns=column))

    @pytest.mark.parametrize('groups', ['session', ['session'], ('session',)])
    def test_custom_columns_and_group(self, groups):
        df = self.frame().rename(columns={
            'match_id': 'session', 'event_time_seconds': 't',
            'ball_x_start': 'x', 'ball_y_start': 'y',
        })
        result = BallSpeed(time_col='t', x_col='x', y_col='y', group_col=groups).transform(df)
        assert result.ball_speed.iloc[1] == 2.5

    @pytest.mark.parametrize('groups', [[], (), 42, False])
    def test_invalid_group_configuration(self, groups):
        with pytest.raises(ValueError, match='group_col'):
            BallSpeed(group_col=groups)

    def test_empty_frame(self):
        df = self.frame().iloc[:0]
        result = BallSpeed().transform(df)
        pd.testing.assert_frame_equal(result[df.columns], df)
        assert result.ball_speed.dtype == np.float64

    def test_helper_names_are_not_overwritten(self):
        df = self.frame()
        for col in ['time', 'x', 'y', 'group', 'row_order', 'event_order']:
            df[col] = 'keep'
        result = BallSpeed().transform(df)
        pd.testing.assert_frame_equal(result[df.columns], df)

    def test_randomized_reference_with_shuffled_events(self):
        import math
        rng = np.random.default_rng(713)
        df = pd.DataFrame({
            'match_id': np.repeat([1, 2], 40),
            'period': np.tile(np.repeat([1, 2], 20), 2),
            'event_time_seconds': np.tile(np.arange(20) * 2., 4),
            'ball_x_start': rng.uniform(0, 105, 80),
            'ball_y_start': rng.uniform(0, 68, 80),
        })
        expected = []
        for i in range(len(df)):
            if i % 20 == 0:
                expected.append(np.nan)
            else:
                previous = df.iloc[i - 1]
                current = df.iloc[i]
                expected.append(math.dist(
                    (previous.ball_x_start, previous.ball_y_start),
                    (current.ball_x_start, current.ball_y_start),
                ) / 2)
        shuffled = df.sample(frac=1, random_state=8)
        result = BallSpeed().transform(shuffled)
        np.testing.assert_allclose(result.ball_speed, np.asarray(expected)[shuffled.index], equal_nan=True)

# ---------------------------------------------------------------------
# DistNearestDefender
# ---------------------------------------------------------------------

class TestDistNearestDefender:
    def test_empty_opponent_list_produces_nan(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[]],
        })
        result = DistNearestDefender().transform(df)
        assert pd.isna(result['dist_nearest_defender'].iloc[0])

    def test_single_opponent_correct_distance(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(3.0, 4.0)]],
        })
        result = DistNearestDefender().transform(df)
        assert result['dist_nearest_defender'].iloc[0] == pytest.approx(5.0)

    def test_picks_nearest_of_multiple_opponents(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(10.0, 0.0), (1.0, 0.0), (5.0, 0.0)]],
        })
        result = DistNearestDefender().transform(df)
        assert result['dist_nearest_defender'].iloc[0] == pytest.approx(1.0)

    def test_invalid_opponent_entry_is_filtered_out(self):
        # a (nan, nan) tuple, as FreezeFrameExtractor can produce, should
        # be ignored rather than corrupting the min.
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(np.nan, np.nan), (2.0, 0.0)]],
        })
        result = DistNearestDefender().transform(df)
        assert result['dist_nearest_defender'].iloc[0] == pytest.approx(2.0)

    def test_all_invalid_opponents_produces_nan(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(np.nan, np.nan)]],
        })
        result = DistNearestDefender().transform(df)
        assert pd.isna(result['dist_nearest_defender'].iloc[0])

    def test_nan_ball_position_produces_nan(self):
        df = pd.DataFrame({
            'ball_x_start': [np.nan], 'ball_y_start': [0.0],
            'opponent_locations': [[(2.0, 0.0)]],
        })
        result = DistNearestDefender().transform(df)
        assert pd.isna(result['dist_nearest_defender'].iloc[0])

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [0.0], 'ball_y_start': [0.0]})
        with pytest.raises(KeyError):
            DistNearestDefender().transform(df)

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(2.0, 0.0)]],
        })
        DistNearestDefender().transform(df)
        assert 'dist_nearest_defender' not in df.columns


# ---------------------------------------------------------------------
# DefendersIn3m
# ---------------------------------------------------------------------

class TestDefendersIn3m:
    def test_empty_opponent_list_produces_zero(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[]],
        })
        result = DefendersIn3m().transform(df)
        assert result['defenders_in_3m'].iloc[0] == 0

    def test_counts_only_opponents_within_radius(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(1.0, 0.0), (2.0, 0.0), (5.0, 0.0)]],
        })
        result = DefendersIn3m().transform(df)
        assert result['defenders_in_3m'].iloc[0] == 2

    def test_radius_boundary_is_inclusive(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(3.0, 0.0)]],
        })
        result = DefendersIn3m().transform(df)
        assert result['defenders_in_3m'].iloc[0] == 1

    def test_invalid_opponent_entry_is_filtered_out(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(np.nan, np.nan), (1.0, 0.0)]],
        })
        result = DefendersIn3m().transform(df)
        assert result['defenders_in_3m'].iloc[0] == 1

    def test_nan_ball_position_produces_nan(self):
        df = pd.DataFrame({
            'ball_x_start': [np.nan], 'ball_y_start': [0.0],
            'opponent_locations': [[(1.0, 0.0)]],
        })
        result = DefendersIn3m().transform(df)
        assert pd.isna(result['defenders_in_3m'].iloc[0])

    def test_custom_radius(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(4.0, 0.0)]],
        })
        result = DefendersIn3m(radius=5).transform(df)
        assert result['defenders_in_3m'].iloc[0] == 1

    def test_invalid_radius_raises_valueerror(self):
        with pytest.raises(ValueError):
            DefendersIn3m(radius=-1)
        with pytest.raises(ValueError):
            DefendersIn3m(radius=np.inf)
        with pytest.raises(ValueError):
            DefendersIn3m(radius=True)

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'ball_x_start': [0.0], 'ball_y_start': [0.0]})
        with pytest.raises(KeyError):
            DefendersIn3m().transform(df)

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({
            'ball_x_start': [0.0], 'ball_y_start': [0.0],
            'opponent_locations': [[(1.0, 0.0)]],
        })
        DefendersIn3m().transform(df)
        assert 'defenders_in_3m' not in df.columns



# ---------------------------------------------------------------------
# BallSpeedImputer
# ---------------------------------------------------------------------

class TestBallSpeedImputer:
    def test_adds_missingness_flag_column(self):
        df = pd.DataFrame({'ball_speed': [1.0, np.nan, 3.0]})
        result = BallSpeedImputer().transform(df)
        assert 'ball_speed_missing' in result.columns

    def test_flag_is_1_for_missing_0_for_present(self):
        df = pd.DataFrame({'ball_speed': [1.0, np.nan, 3.0]})
        result = BallSpeedImputer().transform(df)
        assert result['ball_speed_missing'].tolist() == [0, 1, 0]

    def test_fills_nan_with_median_of_present_values(self):
        # median of [1.0, 3.0, 5.0] (NaN excluded) is 3.0
        df = pd.DataFrame({'ball_speed': [1.0, np.nan, 3.0, 5.0]})
        result = BallSpeedImputer().transform(df)
        assert result['ball_speed'].tolist() == [1.0, 3.0, 3.0, 5.0]

    def test_no_missing_values_leaves_column_unchanged(self):
        df = pd.DataFrame({'ball_speed': [1.0, 2.0, 3.0]})
        result = BallSpeedImputer().transform(df)
        assert result['ball_speed'].tolist() == [1.0, 2.0, 3.0]
        assert result['ball_speed_missing'].tolist() == [0, 0, 0]

    def test_all_values_missing_flag_all_1_and_stays_nan(self):
        # median of an all-NaN series is NaN, so fillna has nothing to fill with
        df = pd.DataFrame({'ball_speed': [np.nan, np.nan]})
        result = BallSpeedImputer().transform(df)
        assert result['ball_speed_missing'].tolist() == [1, 1]
        assert result['ball_speed'].isna().all()

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'ball_speed': [1.0, np.nan, 3.0]})
        BallSpeedImputer().transform(df)
        assert df['ball_speed'].isna().sum() == 1
        assert 'ball_speed_missing' not in df.columns

    def test_missing_column_raises_keyerror(self):
        df = pd.DataFrame({'not_ball_speed': [1.0, 2.0]})
        with pytest.raises(KeyError):
            BallSpeedImputer().transform(df)


# ---------------------------------------------------------------------
# OpponentDataDropper
# ---------------------------------------------------------------------

class TestOpponentDataDropper:
    REQUIRED = ['dist_nearest_defender', 'team_centroid_distance', 'open_angle_goal']

    def _full_frame(self):
        return pd.DataFrame({
            'dist_nearest_defender': [1.0, np.nan, 3.0],
            'team_centroid_distance': [10.0, 20.0, 30.0],
            'open_angle_goal': [0.1, 0.2, 0.3],
        })

    def test_drops_rows_with_nan_dist_nearest_defender(self):
        df = self._full_frame()
        result = OpponentDataDropper().transform(df)
        assert result['dist_nearest_defender'].tolist() == [1.0, 3.0]

    def test_keeps_rows_where_other_required_columns_are_nan(self):
        # only dist_nearest_defender governs the drop
        df = pd.DataFrame({
            'dist_nearest_defender': [1.0, 2.0],
            'team_centroid_distance': [np.nan, 20.0],
            'open_angle_goal': [0.1, np.nan],
        })
        result = OpponentDataDropper().transform(df)
        assert len(result) == 2

    def test_no_missing_values_returns_all_rows(self):
        df = self._full_frame().dropna()
        result = OpponentDataDropper().transform(df)
        assert len(result) == len(df)

    def test_preserves_index_of_kept_rows(self):
        df = self._full_frame()
        result = OpponentDataDropper().transform(df)
        assert result.index.tolist() == [0, 2]

    def test_does_not_mutate_original_df(self):
        df = self._full_frame()
        OpponentDataDropper().transform(df)
        assert df['dist_nearest_defender'].isna().sum() == 1
        assert len(df) == 3

    @pytest.mark.parametrize('missing_col', REQUIRED)
    def test_missing_required_column_raises_keyerror(self, missing_col):
        df = self._full_frame().drop(columns=[missing_col])
        with pytest.raises(KeyError):
            OpponentDataDropper().transform(df)

    def test_error_message_lists_all_missing_columns(self):
        df = self._full_frame().drop(columns=['team_centroid_distance', 'open_angle_goal'])
        with pytest.raises(KeyError) as excinfo:
            OpponentDataDropper().transform(df)
        message = str(excinfo.value)
        assert 'team_centroid_distance' in message
        assert 'open_angle_goal' in message


# ---------------------------------------------------------------------
# ShotFilter
# ---------------------------------------------------------------------

class TestShotFilter:
    def test_filters_to_shot_rows_only(self):
        df = pd.DataFrame({'type': ['Pass', 'Shot', 'Carry', 'Shot']})
        result = ShotFilter().transform(df)
        assert result['type'].tolist() == ['Shot', 'Shot']

    def test_no_shots_returns_empty_df(self):
        df = pd.DataFrame({'type': ['Pass', 'Carry']})
        result = ShotFilter().transform(df)
        assert result.empty

    def test_all_shots_returns_all_rows_unchanged(self):
        df = pd.DataFrame({'type': ['Shot', 'Shot']})
        result = ShotFilter().transform(df)
        assert len(result) == 2

    def test_preserves_other_columns(self):
        df = pd.DataFrame({
            'type': ['Pass', 'Shot'],
            'ball_x_start': [1.0, 2.0],
        })
        result = ShotFilter().transform(df)
        assert result['ball_x_start'].tolist() == [2.0]

    def test_preserves_index_of_kept_rows(self):
        df = pd.DataFrame({'type': ['Pass', 'Shot', 'Shot']})
        result = ShotFilter().transform(df)
        assert result.index.tolist() == [1, 2]

    def test_does_not_mutate_original_df(self):
        df = pd.DataFrame({'type': ['Pass', 'Shot']})
        ShotFilter().transform(df)
        assert len(df) == 2

    def test_missing_type_column_raises_keyerror(self):
        df = pd.DataFrame({'not_type': ['Shot']})
        with pytest.raises(KeyError):
            ShotFilter().transform(df)


# ---------------------------------------------------------------------
# xgboost model
# ---------------------------------------------------------------------

class TestXGBoostXGModel:
    @staticmethod
    def sample():
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            weights=[0.8, 0.2], random_state=42,
        )
        return pd.DataFrame(X, columns=list('abcde')), pd.Series(y)

    def test_pipeline_contract_and_probability_objective(self):
        from sklearn.pipeline import Pipeline as ModelPipeline
        from xgboost import XGBClassifier
        from src.models.xg_model.xgboost_model import XGBoostXGModel
        pipeline = XGBoostXGModel().build_pipeline()
        assert isinstance(pipeline, ModelPipeline)
        assert list(pipeline.named_steps) == ['clf']
        classifier = pipeline.named_steps['clf']
        assert isinstance(classifier, XGBClassifier)
        assert classifier.objective == 'binary:logistic'
        assert classifier.eval_metric == 'logloss'
        assert classifier.scale_pos_weight == 1
        assert classifier.n_jobs == 1
        assert classifier.random_state == 42

    def test_fresh_unfitted_pipeline_and_independent_grid(self):
        from sklearn.exceptions import NotFittedError
        from src.models.xg_model.xgboost_model import XGBoostXGModel
        model = XGBoostXGModel()
        first, second = model.build_pipeline(), model.build_pipeline()
        first.set_params(clf__max_depth=9)
        assert second.named_steps['clf'].max_depth != 9
        with pytest.raises(NotFittedError):
            second.predict_proba(self.sample()[0])
        grid = model.get_param_grid()
        grid['clf__max_depth'].append(99)
        assert 99 not in model.get_param_grid()['clf__max_depth']

    def test_all_grid_candidates_use_supported_parameters(self):
        from sklearn.base import clone
        from sklearn.model_selection import ParameterGrid
        from src.models.xg_model.xgboost_model import XGBoostXGModel
        model = XGBoostXGModel()
        pipeline = model.build_pipeline()
        grid = model.get_param_grid()
        assert len(list(ParameterGrid(grid))) == 729
        assert all(name.startswith('clf__') for name in grid)
        assert 'clf__scale_pos_weight' not in grid
        for candidate in ParameterGrid(grid):
            configured = clone(pipeline).set_params(**candidate)
            assert configured.named_steps['clf'].max_depth > 0
            assert 0 < configured.named_steps['clf'].subsample <= 1

    def test_fit_predict_with_missing_numeric_data_preserves_input(self):
        from src.models.xg_model.xgboost_model import XGBoostXGModel
        X, y = self.sample()
        X.loc[::7, 'a'] = np.nan
        original = X.copy(deep=True)
        pipeline = XGBoostXGModel().build_pipeline().set_params(clf__n_estimators=10)
        pipeline.fit(X, y)
        probabilities = pipeline.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        pd.testing.assert_frame_equal(X, original)

    def test_seed_reproduces_fit(self):
        from src.models.xg_model.xgboost_model import XGBoostXGModel
        X, y = self.sample()
        first = XGBoostXGModel().build_pipeline().set_params(clf__n_estimators=10)
        second = XGBoostXGModel().build_pipeline().set_params(clf__n_estimators=10)
        first.fit(X, y)
        second.fit(X, y)
        np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))

    def test_shared_train_and_prediction_interface(self, monkeypatch):
        from joblib import parallel_backend
        from src.models.xg_model.xgboost_model import XGBoostXGModel
        model = XGBoostXGModel()
        X, y = self.sample()
        with pytest.raises(RuntimeError, match='train'):
            model.predict_proba(X)
        # keep this integration test small while exercising the real base trainer.
        monkeypatch.setattr(model, 'get_param_grid', lambda: {
            'clf__n_estimators': [5, 10], 'clf__max_depth': [2],
        })
        with parallel_backend('threading', n_jobs=2):
            best = model.train(X=X, y=y, cv=2)
        assert best is model.best_estimator_
        assert model.search.scoring == 'neg_log_loss'
        assert len(model.X_test) == 25
        predictions = model.predict_proba(model.X_test)
        assert predictions.shape == (25,)
        np.testing.assert_allclose(predictions, best.predict_proba(model.X_test)[:, 1])

    def test_serialized_pipeline_predictions_match(self, tmp_path):
        import joblib
        from src.models.xg_model.xgboost_model import XGBoostXGModel
        X, y = self.sample()
        pipeline = XGBoostXGModel().build_pipeline().set_params(clf__n_estimators=5)
        pipeline.fit(X, y)
        path = tmp_path / 'model.joblib'
        joblib.dump(pipeline, path)
        np.testing.assert_allclose(
            pipeline.predict_proba(X), joblib.load(path).predict_proba(X)
        )


class TestLogisticRegressionEventModel:
    @staticmethod
    def sample():
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=250, n_features=8, n_informative=6,
                                   n_classes=5, n_clusters_per_class=1, random_state=42)
        names = np.array(['Carry', 'Foul', 'Other', 'Pass', 'Shot'])
        return pd.DataFrame(X), pd.Series(names[y]), pd.Series(np.repeat(np.arange(25), 10))

    def test_pipeline_and_search_contract(self):
        from sklearn.model_selection import ParameterGrid
        from sklearn.preprocessing import StandardScaler
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        model = LogisticRegressionEventModel()
        pipeline = model.build_pipeline()
        assert isinstance(pipeline.named_steps['scaler'], StandardScaler)
        clf = pipeline.named_steps['clf']
        assert clf.class_weight == 'balanced'
        assert clf.solver == 'lbfgs'
        assert clf.max_iter == 2000
        assert clf.random_state == 42
        grid = model.get_param_grid()
        assert grid['clf__C'] == [0.001, 0.01, 0.1, 1, 10, 100]
        assert len(list(ParameterGrid(grid))) == 6
        for params in ParameterGrid(grid):
            model.build_pipeline().set_params(**params)

    def test_fresh_objects_and_unfitted_errors(self):
        from sklearn.exceptions import NotFittedError
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        model = LogisticRegressionEventModel()
        X, _, _ = self.sample()
        with pytest.raises(NotFittedError):
            model.build_pipeline().predict(X)
        for method in [model.predict, model.predict_proba]:
            with pytest.raises(RuntimeError, match='train'):
                method(X)
        with pytest.raises(RuntimeError, match='train'):
            model.evaluate(plot=False)
        first = model.build_pipeline().set_params(clf__C=100)
        assert model.build_pipeline().named_steps['clf'].C == 1
        grid = model.get_param_grid()
        grid['clf__C'].append(999)
        assert 999 not in model.get_param_grid()['clf__C']

    def test_speed_fill_cannot_reveal_label_or_mutate_input(self):
        from src.models.event_classification_model.event_logistic_regression import _neutralize_missing_speed
        X = pd.DataFrame({'ball_speed': [2.5, 10.9, 3.0], 'ball_speed_missing': [1, 1, 0]})
        original = X.copy(deep=True)
        safe = _neutralize_missing_speed(X)
        assert safe.ball_speed.tolist() == [0.0, 0.0, 3.0]
        pd.testing.assert_frame_equal(X, original)
        with pytest.raises(ValueError, match='indicator'):
            _neutralize_missing_speed(X.drop(columns='ball_speed_missing'))
        with pytest.raises(TypeError, match='dataframe'):
            _neutralize_missing_speed(X.to_numpy())
        with pytest.raises(ValueError, match='target'):
            _neutralize_missing_speed(X.assign(match_id=1))

    def test_multiclass_probabilities_and_serialization(self, tmp_path):
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        X, y, _ = self.sample()
        original = X.copy(deep=True)
        model = LogisticRegressionEventModel()
        model.best_estimator_ = model.build_pipeline().fit(X, y)
        p = model.predict_proba(X)
        assert p.shape == (250, 5)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(X), model.best_estimator_.classes_[p.argmax(axis=1)])
        pd.testing.assert_frame_equal(X, original)
        model.save(tmp_path / 'event.pkl')
        loaded = LogisticRegressionEventModel()
        loaded.load(tmp_path / 'event.pkl')
        np.testing.assert_allclose(loaded.predict_proba(X), p)

    def test_shared_grouped_training_scaler_and_evaluator(self):
        from joblib import parallel_backend
        from sklearn.model_selection import GroupShuffleSplit
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        X, y, groups = self.sample()
        model = LogisticRegressionEventModel()
        with parallel_backend('threading', n_jobs=1):
            model.train(X, y, groups, n_iter=6, n_splits=2)
        train, test = next(GroupShuffleSplit(test_size=.2, random_state=42).split(X, y, groups))
        assert set(groups.iloc[train]).isdisjoint(model.groups_test)
        pd.testing.assert_frame_equal(model.X_test, X.iloc[test])
        np.testing.assert_allclose(model.best_estimator_.named_steps['scaler'].mean_, X.iloc[train].mean())
        for a, b in model.search.cv.split(X.iloc[train], y.iloc[train], groups.iloc[train]):
            assert set(groups.iloc[train].iloc[a]).isdisjoint(groups.iloc[train].iloc[b])
        assert model.search.scoring == 'f1_macro'
        assert len(model.search.cv_results_['params']) == 6
        result = model.evaluate(plot=False)
        assert 0 <= result['macro_f1'] <= 1
        assert result['confusion_matrix'].shape == (5, 5)

    def test_public_model_imports_remain_compatible(self):
        from src.models import LogisticRegressionModel, RandomForestModel, XGBoostModel
        assert LogisticRegressionModel.__name__ == 'LogisticRegressionXGModel'
        assert RandomForestModel.__name__ == 'RandomForestXGModel'
        assert XGBoostModel.__name__ == 'XGBoostXGModel'
