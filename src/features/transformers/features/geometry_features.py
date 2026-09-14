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
        
        if x_col is None:
            self.x_col = 'ball_x_start'
        else:
            self.x_col  = x_col

        if y_col is None:
            self.y_col = 'ball_y_start'
        else:
            self.y_col = y_col            
        # following the same param-then-self.attr pattern as TimeParser
        

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
       
        df = df.copy()
        if self.x_col not in df.columns or self.y_col not in df.columns:
            raise KeyError(f"columns{self.x_col} or {self.y_col} or both of them not found in DataFrame")
        df['dist_to_goal'] = np.sqrt((GOAL_X - df[self.x_col])**2 + (GOAL_Y - df[self.y_col])**2)

        return df
        


class ShotAngle(Transformer):
    def __init__(self, x_col=None, y_col=None):
        if x_col is None:
            self.x_col = 'ball_x_start'
        else:
            self.x_col  = x_col
        
        if y_col is None:
            self.y_col = 'ball_y_start'
        else:
            self.y_col = y_col   
       

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.x_col not in df.columns or self.y_col not in df.columns:
            raise KeyError(f"columns{self.x_col} or {self.y_col} or both of them not found in DataFrame")
        
        angle_1 = np.arctan2(GOALPOST_1[1] - df[self.y_col] , GOALPOST_1[0] - df[self.x_col])
        angle_2 = np.arctan2(GOALPOST_2[1] - df[self.y_col] , GOALPOST_2[0] - df[self.x_col])
        
        diff = angle_1 - angle_2
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        df['shot_angle'] = np.abs(diff)

        return df
       


class InPenaltyBox(Transformer):
    def __init__(self, x_col=None, y_col=None):
        if x_col is None:
            self.x_col = 'ball_x_start'
        else:
            self.x_col  = x_col
        
        if y_col is None:
            self.y_col = 'ball_y_start'
        else:
            self.y_col = y_col        
       
    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.x_col not in df.columns or self.y_col not in df.columns:
            raise KeyError(f"columns{self.x_col} or {self.y_col} or both of them not found in DataFrame")
        df['in_penalty_box'] = (
            (df[self.x_col] >= PENALTY_BOX_X_MIN)&
            (df[self.y_col] >= PENALTY_BOX_Y_MIN)&
            (df[self.y_col] <= PENALTY_BOX_Y_MAX)
        ).astype(int)    

        return df    

       