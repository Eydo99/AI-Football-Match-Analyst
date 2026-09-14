from typing import override

import numpy as np
import pandas as pd

from src.features.base import Transformer


class BallSpeed(Transformer):
    """estimate event-to-event movement in metres per second.

    default groups are match_id and period because timestamps reset each period.
    custom group_col accepts a column name or a non-empty sequence of names;
    callers must include all boundaries in custom groups. coordinates must already
    be in metres and time in seconds. when team is available, transitions between
    teams are unknown because event coordinates use team-relative orientation.

    this is displacement between recorded event starts, not tracking-derived
    instantaneous ball velocity. first rows, missing inputs, and non-positive
    time differences yield nan. short intervals can amplify location noise;
    finite values are not clipped or smoothed. the original row order and index
    are preserved.
    """

    def __init__(self, time_col=None, x_col=None, y_col=None, group_col=None):
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
        elapsed = deltas['time'].where(deltas['time'] > 0)
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
        transformed_df['ball_speed'] = speed.reindex(range(len(source))).to_numpy()
        return transformed_df
