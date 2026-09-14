from typing import override

import numpy as np
import pandas as pd

from src.features.base import Transformer

DEFENDER_RADIUS_M = 3


def _finite_point(value):
    """return a finite coordinate pair, or none for unusable coordinates."""
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return point


def _opponent_distances(x, y, opponents):
    """distances in metres; none means the ball position is unknown.

    ignore invalid opponent entries, including nan pairs produced by
    freezeframeextractor. missing collections contain no known opponents.
    """
    # an unknown ball position cannot give a meaningful distance or count.
    ball = _finite_point((x, y))
    if ball is None:
        return None
    distances = []
    if isinstance(opponents, (list, tuple, np.ndarray)):
        if isinstance(opponents, np.ndarray) and opponents.ndim == 0:
            return np.asarray(distances, dtype=float)
        for location in opponents:
            point = _finite_point(location)
            if point is not None:
                distances.append(np.hypot(point[0] - ball[0], point[1] - ball[1]))
    return np.asarray(distances, dtype=float)


class DistNearestDefender(Transformer):
    """distance to the nearest recorded opponent, in metres.

    inputs must already use metric coordinates. goalkeepers are included when
    supplied upstream. missing ball coordinates or no valid opponents produce
    nan, rather than a misleading zero distance.
    """

    def __init__(self, x_col=None, y_col=None, opponents_col=None):
        self.x_col = 'ball_x_start' if x_col is None else x_col
        self.y_col = 'ball_y_start' if y_col is None else y_col
        self.opponents_col = 'opponent_locations' if opponents_col is None else opponents_col

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # preserve existing features and raise keyerror for missing inputs.
        transformed_df = df.copy()
        rows = transformed_df[[self.x_col, self.y_col, self.opponents_col]]
        values = []
        for x, y, opponents in rows.itertuples(index=False, name=None):
            distances = _opponent_distances(x, y, opponents)
            values.append(distances.min() if distances is not None and distances.size else np.nan)
        # no valid opponents means unknown distance, not zero distance.
        transformed_df['dist_nearest_defender'] = np.asarray(values, dtype=float)
        return transformed_df


class DefendersIn3m(Transformer):
    """count recorded opponents at distance <= radius (default 3 metres).

    empty/missing collections or collections with no valid opponents yield zero.
    this counts known positions, not proof that no unseen defenders are nearby.
    missing ball coordinates yield nan. goalkeepers supplied upstream are counted.
    counts use float dtype to accommodate nan without an object column.
    """

    def __init__(self, x_col=None, y_col=None, opponents_col=None, radius=None):
        self.x_col = 'ball_x_start' if x_col is None else x_col
        self.y_col = 'ball_y_start' if y_col is None else y_col
        self.opponents_col = 'opponent_locations' if opponents_col is None else opponents_col
        radius = DEFENDER_RADIUS_M if radius is None else radius
        if isinstance(radius, (bool, np.bool_)) or not np.isscalar(radius):
            raise ValueError('radius must be a finite, non-negative number')
        try:
            self.radius = float(radius)
        except (TypeError, ValueError):
            raise ValueError('radius must be a finite, non-negative number') from None
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError('radius must be a finite, non-negative number')

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # preserve existing features and raise keyerror for missing inputs.
        transformed_df = df.copy()
        rows = transformed_df[[self.x_col, self.y_col, self.opponents_col]]
        values = []
        for x, y, opponents in rows.itertuples(index=False, name=None):
            distances = _opponent_distances(x, y, opponents)
            values.append(np.nan if distances is None else np.count_nonzero(distances <= self.radius))
        # include the radius boundary and count only recorded valid opponents.
        transformed_df['defenders_in_3m'] = np.asarray(values, dtype=float)
        return transformed_df
