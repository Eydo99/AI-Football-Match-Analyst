from typing import override
import pandas as pd
import numpy as np
from src.features.base import Transformer



class TeamCentroidDistance(Transformer):

    Requires = ['opponent_locations', 'ball_x_start', 'ball_y_start']

    def __init__(self):
        pass

    def compute_centroid_dist(self, row):
        if (row['opponent_locations'] and isinstance(row['opponent_locations'], list) and len(row['opponent_locations']) > 0 
            and pd.notna(row['ball_x_start']) and pd.notna(row['ball_y_start'])):
            opp_x = [loc[0] for loc in row['opponent_locations']]
            opp_y = [loc[1] for loc in row['opponent_locations']]
            return np.sqrt((np.mean(opp_x) - row['ball_x_start'])**2 + (np.mean(opp_y) - row['ball_y_start'])**2)
        return np.nan

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        missing_columns = [col for col in self.Requires if col not in df.columns]
        if missing_columns:
            raise KeyError(f"TeamCentroidDistance requires columns {missing_columns}, not found in DataFrame")
        
        transformed_df = df.copy()
        transformed_df['team_centroid_distance'] = transformed_df.apply(self.compute_centroid_dist,axis=1)
        return transformed_df


class OpenAngleGoal(Transformer):

    Requires = ['opponent_locations', 'ball_x_start', 'ball_y_start']

    def __init__(self , player_width = None):
        if player_width is None:
            player_width = 0.5
        self.player_width = player_width

    def compute_open_angle_goal(self, row):
        if(row['opponent_locations'] and isinstance(row['opponent_locations'], list) and len(row['opponent_locations']) > 0
           and pd.notna(row['ball_x_start']) and pd.notna(row['ball_y_start'])):
            
            angle_left_post = np.arctan2(30.34 - row['ball_y_start'], 105 - row['ball_x_start'])
            angle_right_post = np.arctan2(37.66 - row['ball_y_start'], 105 - row['ball_x_start'])

            opponent_angles = []
            for opp in row['opponent_locations']:
                opp_angle = np.arctan2(opp[1] - row['ball_y_start'], opp[0] - row['ball_x_start'])
                if angle_left_post < opp_angle < angle_right_post:
                    if opp[0] < 105 and opp[0] > row['ball_x_start']:

                        distance = np.sqrt((opp[0] - row['ball_x_start'])**2 + (opp[1] - row['ball_y_start'])**2)
                        half_angle = np.arctan2(self.player_width , distance)

                        start = opp_angle - half_angle
                        end = opp_angle + half_angle

                        clipped_start = max(start, angle_left_post)
                        clipped_end = min(end, angle_right_post)

                        opponent_angles.append((clipped_start, clipped_end))
                    else:
                        continue
                else:
                    continue

            merged_intervals = []
            current_angle = None

            for interval in sorted(opponent_angles):
                if current_angle is None:
                    current_angle = interval

                elif current_angle and interval[0] <= current_angle[1] and interval[1] >= current_angle[0]:
                    current_angle = (min(current_angle[0], interval[0]), max(current_angle[1], interval[1]))

                else:
                    merged_intervals.append(current_angle)
                    current_angle = interval

            if current_angle:
                merged_intervals.append(current_angle)
            
            total_blocked_angle = sum(end - start for start, end in merged_intervals)
            open_angle = (angle_right_post - angle_left_post) - total_blocked_angle

            return max(open_angle, 0)
        
        return np.nan

    @override
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        missing_columns = [col for col in self.Requires if col not in df.columns]
        if missing_columns:
            raise KeyError(f"OpenAngleGoal requires columns {missing_columns}, not found in DataFrame")

        transformed_df = df.copy()
        transformed_df['open_angle_goal'] = transformed_df.apply(self.compute_open_angle_goal, axis=1)
        return transformed_df
