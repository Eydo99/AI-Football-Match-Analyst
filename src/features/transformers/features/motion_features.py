from typing import override

import numpy as np
import pandas as pd

from src.features.base import Transformer


class BallSpeed(Transformer):

    def __init__(self, time_col=None, x_col=None, y_col=None, group_col=None,
                 min_elapsed=0.1, max_speed=60.0):
        # keep the existing constructor and allow alternate input schemas.
        self.time_col = 'event_time_seconds' if time_col is None else time_col
        self.x_col = 'ball_x_start' if x_col is None else x_col
        self.y_col = 'ball_y_start' if y_col is None else y_col
        if group_col is None:
            group_col = ['match_id', 'period']
        elif isinstance(group_col, str):
            group_col = [group_col]
        if not isinstance(group_col, (list, tuple)) or not group_col:
            raise ValueError('group_col must be a column name or a non-empty list of names')
        self.group_col = list(group_col)

        # min_elapsed guards against near-zero time denominators amplifying
        # ordinary positional noise into implausible speeds. max_speed is a
        # final physical sanity cap applied after that guard, not a substitute
        # for it — clipping alone would still fabricate a value for an
        # interval that was never trustworthy to begin with.
        if min_elapsed <= 0:
            raise ValueError('min_elapsed must be positive')
        if max_speed <= 0:
            raise ValueError('max_speed must be positive')
        self.min_elapsed = min_elapsed
        self.max_speed = max_speed

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()

        # selecting required columns raises keyerror before any calculation.
        required = [self.time_col, self.x_col, self.y_col] + self.group_col
        source = transformed_df[required].reset_index(drop=True)
        numeric = source[[self.time_col, self.x_col, self.y_col]].apply(
            pd.to_numeric, errors='coerce'
        ).to_numpy(dtype=float, na_value=np.nan, copy=True)
        numeric[~np.isfinite(numeric)] = np.nan

        # use a separate table so helper names cannot overwrite project columns.
        work = pd.DataFrame(numeric, columns=['time', 'x', 'y'])
        work['group'] = source.groupby(self.group_col, sort=False).ngroup()
        work['row_order'] = np.arange(len(source))
        if 'index' in transformed_df.columns:
            # statsbomb's event index breaks timestamp ties chronologically.
            work['event_order'] = pd.to_numeric(
                transformed_df['index'].reset_index(drop=True), errors='coerce'
            )
        else:
            work['event_order'] = work['row_order']
        if 'team' in transformed_df.columns:
            work['team'] = transformed_df['team'].to_numpy()

        # compute within groups, without changing the caller's dataframe order.
        work = work.sort_values(['group', 'time', 'event_order', 'row_order'])
        grouped = work.groupby('group', sort=False)
        deltas = grouped[['time', 'x', 'y']].diff()

        # below min_elapsed, the denominator is unreliable enough that any
        # resulting "speed" is more likely to reflect positional noise than
        # real movement — treat it as unknown rather than computing it.
        elapsed = deltas['time'].where(deltas['time'] > self.min_elapsed)
        speed = np.hypot(deltas['x'], deltas['y']) / elapsed

        if 'team' in work.columns:
            # do not mistake a change in team-relative coordinates for movement.
            previous_team = grouped['team'].shift()
            same_team = work['team'].notna() & previous_team.notna()
            same_team &= work['team'].eq(previous_team).fillna(False)
            speed = speed.where(same_team)

        # invalid intervals remain unknown, and positional alignment handles
        # duplicate dataframe indices without mixing up the results.
        speed = speed.where(np.isfinite(speed))

        # final physical sanity cap on whatever remains, as a second line of
        # defense against noise the min_elapsed guard doesn't catch.
        speed = speed.clip(upper=self.max_speed)

        transformed_df['ball_speed'] = speed.reindex(range(len(source))).to_numpy()
        return transformed_df