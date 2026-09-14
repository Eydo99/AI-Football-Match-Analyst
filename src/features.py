import os
import pandas as pd
from src.features.pipeline import Pipeline
from src.features.transformers.cleaning.column_pruner import ColumnPruner
from src.features.transformers.cleaning.boolean_encoder import BooleanEncoder
from src.features.transformers.features.trajectory_features import GoalOutcomeTransformer
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

    pipeline=Pipeline([ColumnPruner(),LocationSplitter(),CoordinateRescaler(),BooleanEncoder(),TimeParser(),FreezeFrameExtractor(),GoalOutcomeTransformer()])
    master_df=pipeline.transform(events_df)


    # Save the processed subset to disk

    # lsa e7na 3ayzen n3ml feature eng a3tkd elawl abl 3lshan ntl3 final filtered data ely
    # e7na 3awzenha w b3den n save it in dir

    # output_path = os.path.join(output_dir, 'filtered_required_events.parquet')
    # events_df.to_parquet(output_path, index=False)
    # print(f"Saved filtered events to {output_path}")



if __name__ == "__main__":
    build_features()