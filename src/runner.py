import os
import pandas as pd
from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.column_pruner import ColumnPruner
from src.features.transformers.cleaning.boolean_encoder import BooleanEncoder
from src.features.transformers.features.full_tier_features import TeamCentroidDistance, OpenAngleGoal
from src.features.transformers.features.geometry_features import DistToGoal, ShotAngle, InPenaltyBox
from src.features.transformers.features.motion_features import BallSpeed
from src.features.transformers.features.pressure_features import DistNearestDefender, DefendersIn3m
from src.features.transformers.features.trajectory_features import GoalOutcomeTransformer, TrajectoryAngle, \
    TrajectoryLength
from src.features.transformers.cleaning.location_splitter import LocationSplitter
from src.features.transformers.cleaning.coordinate_rescaler import CoordinateRescaler
from src.features.transformers.cleaning.time_parser import TimeParser
from src.features.transformers.cleaning.freeze_frame_extractor import FreezeFrameExtractor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_features(input_path: str = None, output_dir: str = None):
    """
    Main orchestration function to load raw event data, execute feature extraction pipelines,
    and save processed datasets.
    """
    if input_path is None:
        input_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'master_events.parquet')
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')

    os.makedirs(output_dir, exist_ok=True)

    # Load all individual match event files iteratively to avoid massive memory duplication spikes
    events_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'events')
    event_files = [os.path.join(events_dir, f) for f in os.listdir(events_dir) if f.endswith('.parquet')]

    req = ['Pass', 'Pressure', 'Foul Won', 'Shot', 'Goal Keeper', 'Foul Committed', 'Carry']
    filtered_dfs = []

    for file in event_files:
        try:
            df = pd.read_parquet(file)
            subset = df[df['type'].isin(req)]
            if not subset.empty:
                filtered_dfs.append(subset)
        except Exception as e:
            print(f"Error reading {file}")

    # Concatenate only the filtered subset of relevant events
    events_df = pd.concat(filtered_dfs, ignore_index=True)


    pipeline=Pipeline([ColumnPruner(flag="PRE"),LocationSplitter(),CoordinateRescaler(),BooleanEncoder(),TimeParser()
                        ,FreezeFrameExtractor(),DistToGoal(),ShotAngle(),InPenaltyBox(),GoalOutcomeTransformer(),
                        TrajectoryAngle(),TrajectoryLength(),DistNearestDefender(),DefendersIn3m(),BallSpeed(),
                       TeamCentroidDistance(),OpenAngleGoal(),ColumnPruner(flag="POST")])

    statsbomb_xg_validation=events_df[events_df['type']=='Shot'][['id','match_id','shot_statsbomb_xg']]
    master_df=pipeline.transform(events_df)

    print(master_df.columns)
    print(master_df.dtypes)
    print(master_df.shape)


    master_df_output_path = os.path.join(output_dir, 'master_df.parquet')
    master_df.to_parquet(master_df_output_path, index=False)

    statsbomb_xg_validation_output_path = os.path.join(output_dir, 'statsbomb_xg.parquet')
    statsbomb_xg_validation.to_parquet(statsbomb_xg_validation_output_path, index=False)
    print(f"Saved master df to {master_df_output_path}")
    print(f"Saved statsbomb_xg to {statsbomb_xg_validation_output_path}")



if __name__ == "__main__":
    build_features()