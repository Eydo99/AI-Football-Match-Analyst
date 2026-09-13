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