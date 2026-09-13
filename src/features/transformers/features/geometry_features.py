from typing import override

import numpy as np
import pandas as pd

from src.features.base import Transformer

GOAL_X = 105
GOAL_Y = 34
GOALPOST_1 = (105, 30.34)
GOALPOST_2 = (105, 37.66)
PENALTY_BOX_X_MIN = 88.5
PENALTY_BOX_Y_MIN = 13.84
PENALTY_BOX_Y_MAX = 54.16


class DistToGoal(Transformer):
    def __init__(self, x_col=None, y_col=None):
        # TODO: set defaults ('ball_x_start', 'ball_y_start') if None,
        # following the same param-then-self.attr pattern as TimeParser
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: df.copy()
        # TODO: KeyError check for both columns
        # TODO: compute sqrt((GOAL_X - x)^2 + (GOAL_Y - y)^2) -> new column 'dist_to_goal'
        # TODO: return
        pass


class ShotAngle(Transformer):
    def __init__(self, x_col=None, y_col=None):
        # TODO: same pattern as above
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: df.copy(), KeyError check
        # TODO: angle to GOALPOST_1 via atan2, angle to GOALPOST_2 via atan2
        # TODO: shot_angle = abs(angle1 - angle2) -> decide radians or degrees
        # TODO: return
        pass


class InPenaltyBox(Transformer):
    def __init__(self, x_col=None, y_col=None):
        # TODO: same pattern
        pass

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: df.copy(), KeyError check
        # TODO: boolean condition using PENALTY_BOX_X_MIN/Y_MIN/Y_MAX -> astype(int)
        # TODO: return
        pass