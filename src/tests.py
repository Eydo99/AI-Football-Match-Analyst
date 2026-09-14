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