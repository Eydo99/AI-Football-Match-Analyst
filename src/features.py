import os
import pandas as pd
import numpy as np


def build_features(input_path: str = 'data/raw/master_events.parquet', output_dir: str = 'data/processed'):
    """
    Main orchestration function to load raw event data, execute feature extraction pipelines,
    and save processed datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all individual match event files iteratively to avoid massive memory duplication spikes
    events_dir = 'data/raw/events'
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
    print(f"Filtered events shape: {events_df.shape}")

    # Save the processed subset to disk

    # lsa e7na 3ayzen n3ml feature eng a3tkd elawl abl 3lshan ntl3 final filtered data ely
    # e7na 3awzenha w b3den n save it in dir
    
    # output_path = os.path.join(output_dir, 'filtered_required_events.parquet')
    # events_df.to_parquet(output_path, index=False)
    # print(f"Saved filtered events to {output_path}")
    


if __name__ == "__main__":
    build_features()