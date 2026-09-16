import os
import warnings
import pandas as pd
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

warnings.simplefilter('ignore', category=NoAuthWarning)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fetch_and_save_data(output_dir=None):

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')

    os.makedirs(output_dir, exist_ok=True)
    events_dir = os.path.join(output_dir, 'events')
    os.makedirs(events_dir, exist_ok=True)

    comp_dict = [
        {
        # LA LIGA
        'competition_id': 11
        , 'season_ids': [90,42,4,1,2,27,26,25,24,23,22,21,41,40,39,38]
        },
        {
        # World Cup
        'competition_id': 43
        , 'season_ids': [106,3]
        },
        {
        # Champions League
        'competition_id': 16
        , 'season_ids': [4,1,2,27,26,25,24,23,22,21,41,39,37,44,76,71,276,277]  
        }
    ]

    matches = []

    for comp in comp_dict:
        for season in comp['season_ids']:
            try:
                matches_df_single = sb.matches(competition_id=comp['competition_id'], season_id=season)
                matches.append(matches_df_single)
            except Exception as e:
                print(f"Skipping competition {comp['competition_id']}, season {season}")

        

    matches_df = pd.concat(matches, ignore_index=True)

    for col in matches_df.select_dtypes(include=['object']).columns:
        matches_df[col] = matches_df[col].astype(str)

    matches_df.to_parquet(os.path.join(output_dir, 'master_matches.parquet'), index=False)

   
    total_matches = len(matches_df['match_id'])
    success_count = 0

    for idx, match_id in enumerate(matches_df['match_id']):
        match_file = os.path.join(events_dir, f'match_{match_id}.parquet')

        if os.path.exists(match_file):
            success_count += 1
            continue

        try:
            print(f"Fetching events [{idx + 1}/{total_matches}] for match_id: {match_id}")
            df_match = sb.events(match_id=match_id)
            df_match['match_id'] = match_id
            df_match.to_parquet(match_file, index=False) # type: ignore
            success_count += 1
        except Exception as e:
            print(f"Failed to fetch events for match {match_id}")

    print(f"Successfully saved events for {success_count}/{total_matches} matches into {events_dir}")

if __name__ == "__main__":
    fetch_and_save_data()