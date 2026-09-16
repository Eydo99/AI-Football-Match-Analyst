"""
Chains MatchInferenceRunner (feature engineering) -> MatchEventCleanRunner
(cleaning) -> best_event_model.pkl prediction, for a single match.

Requires: MatchInferenceRunner's load_input() sets df['event_id'] = df['id']
right after sb.events() returns, so it survives every downstream drop step
and can be used as the merge key for attaching predictions back — do NOT
rely on row position alone, since it's fragile if pipeline order ever
changes.

Usage:
    python test_match_pipeline_chained.py <match_id>

Run from the project root (same as events_evaluation.py) so `src` imports
resolve.
"""

import os
import pickle

import pandas as pd

from src.runners.match_test_runners.match_runner import MatchInferenceRunner
from src.runners.match_test_runners.match_event_clean_runner import MatchEventCleanRunner  # adjust import path/name if yours differs
from src.runners.match_test_runners.match_shot_clean_runner import MatchShotCleanRunner  # adjust import path/name if yours differs
from src.runners.train_runners.event_clean_runner import EventCleaningRunner
from src.features.transformers.cleaning.target_encoder import EVENT_CLASS_MAPPING  # adjust path if yours differs
from src.models.xg_model.xgboost_model import XGBoostXGModel

INV_EVENT_CLASS_MAPPING = {v: k for k, v in EVENT_CLASS_MAPPING.items()}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'saved_models', 'best_event_model.pkl')
XG_MODEL_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'saved_models', 'best_xg_model.pkl')

# Same drop_cols as match_aggregation_results.py, plus 'event_id' — the xG
# model was never trained with it, it's an inference-only addition here.
XG_DROP_COLS = ['match_id', 'team', 'type', 'ball_x_start', 'ball_y_start', 'event_id']

# Draw threshold tuned via match_aggregation_results.py's sweep over 1003
# historical matches (value that maximized overall_accuracy: 70.2% overall,
# 80.9% on decisive matches, but only 33.8% on actual draws — a single-match
# 'Draw' call from this pipeline is the least trustworthy part of it).
DRAW_XG_THRESHOLD = 0.3


def predict_shot_xg(shot_df: pd.DataFrame, model_path: str = XG_MODEL_PATH) -> pd.DataFrame:
    model = XGBoostXGModel()
    model.load(model_path)

    feature_cols = [c for c in shot_df.columns if c not in XG_DROP_COLS]
    X = shot_df[feature_cols]

    feature_names = getattr(model.best_estimator_, 'feature_names_in_', None)
    if feature_names is not None:
        missing = set(feature_names) - set(X.columns)
        extra = set(X.columns) - set(feature_names)
        if missing:
            raise ValueError(f"shot_df is missing columns the xG model expects: {missing}")
        if extra:
            raise ValueError(f"shot_df has columns the xG model was never trained on: {extra}")
        X = X[feature_names]
    else:
        print("Warning: fitted xG model has no feature_names_in_ — proceeding with shot_df's "
              "current column order unverified.")

    result_df = shot_df.copy()
    result_df['predicted_xg'] = model.predict_proba(X)  # already [:, 1] per XGBaseModel.predict_proba
    return result_df


def aggregate_predicted_xg(shot_xg_df: pd.DataFrame, team_col='team',
                            draw_threshold: float = DRAW_XG_THRESHOLD) -> pd.DataFrame:
    team_agg = shot_xg_df.groupby(team_col).agg(
        shots=('predicted_xg', 'size'),
        total_xg=('predicted_xg', 'sum'),
    ).reset_index()

    if len(team_agg) != 2:
        raise ValueError(f"Expected exactly 2 teams in shot_xg_df, got {len(team_agg)}: "
                          f"{team_agg[team_col].tolist()}")

    t1, t2 = team_agg.iloc[0], team_agg.iloc[1]
    xg_diff = t1['total_xg'] - t2['total_xg']

    if draw_threshold is not None and abs(xg_diff) <= draw_threshold:
        predicted_winner = 'Draw'
    else:
        predicted_winner = t1[team_col] if xg_diff > 0 else t2[team_col]

    return pd.DataFrame([{
        'team_1': t1[team_col], 'total_xg_1': t1['total_xg'], 'shots_1': t1['shots'],
        'team_2': t2[team_col], 'total_xg_2': t2['total_xg'], 'shots_2': t2['shots'],
        'xg_diff': xg_diff,
        'predicted_winner': predicted_winner,
    }])


def predict_match_events(clean_df: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    drop_for_model = [c for c in ('match_id', 'event_id') if c in clean_df.columns]
    X = clean_df.drop(columns=drop_for_model)

    feature_names = getattr(model.best_estimator_, 'feature_names_in_', None)
    if feature_names is not None:
        missing = set(feature_names) - set(X.columns)
        extra = set(X.columns) - set(feature_names)
        if missing:
            raise ValueError(f"clean_df is missing columns the model expects: {missing}")
        if extra:
            raise ValueError(f"clean_df has columns the model was never trained on: {extra}")
        X = X[feature_names]
    else:
        print("Warning: fitted model has no feature_names_in_ — proceeding with clean_df's "
              "current column order unverified.")

    proba = model.predict_proba(X)
    classes = model.best_estimator_.classes_

    pred_encoded = classes[proba.argmax(axis=1)]
    pred_labels = [INV_EVENT_CLASS_MAPPING[c] for c in pred_encoded]

    result_df = clean_df.copy()
    result_df['predicted_event_class'] = pred_encoded
    result_df['predicted_type'] = pred_labels
    for idx, cls in enumerate(classes):
        result_df[f'proba_{INV_EVENT_CLASS_MAPPING[cls]}'] = proba[:, idx]

    return result_df


def main():
    match_id=int(input("Match ID: "))

    print(f"--- Stage 1: MatchInferenceRunner (feature engineering) for match_id={match_id} ---")
    feature_runner = MatchInferenceRunner(match_id)
    features_df = feature_runner.run()  # writes match_{match_id}_features.parquet to disk
    print(f"features_df shape: {features_df.shape}")
    print(f"features_df columns ({len(features_df.columns)}): {sorted(features_df.columns.tolist())}")

    print(f"\n--- Stage 2: MatchEventCleanRunner (cleaning) ---")
    # Reads the file MatchInferenceRunner just wrote (by match_id), same
    # pattern as EventCleaningRunner reading master_df.parquet by convention
    # rather than being handed a dataframe directly.
    clean_runner = MatchEventCleanRunner(match_id)
    clean_df = clean_runner.run()  # swap for clean_runner.load_output() if run() returns None

    print(f"clean_df shape: {clean_df.shape}")
    print(f"clean_df columns ({len(clean_df.columns)}): {sorted(clean_df.columns.tolist())}")

    print("\n--- Comparing against events_df.parquet schema ---")
    try:
        events_df = EventCleaningRunner().load_output()
    except Exception as e:
        print(f"Could not load events_df for comparison: {e}")
        events_df = None

    if events_df is not None:
        events_cols = set(events_df.columns)
        clean_cols = set(clean_df.columns)

        only_in_events = events_cols - clean_cols
        only_in_clean = clean_cols - events_cols

        # These are expected to be missing from inference output: the
        # prediction targets and grouping keys that only exist post-training.
        expected_missing = {'type', 'ends_in_goal', 'event_class', 'match_id'}
        unexpected_missing = only_in_events - expected_missing

        print(f"In events_df but not clean_df: {sorted(only_in_events)}")
        print(f"In clean_df but not events_df: {sorted(only_in_clean)}")

        if unexpected_missing:
            print(f"\n*** UNEXPECTED missing columns: {sorted(unexpected_missing)} ***")
        else:
            print("\nColumn diff looks as expected.")

        if only_in_clean:
            print(f"\n*** UNEXPECTED extra columns: {sorted(only_in_clean)} ***")

    print("\n--- Row count check ---")
    print(f"features_df rows: {len(features_df)}")
    print(f"clean_df rows:    {len(clean_df)}  "
          f"(diff: {len(features_df) - len(clean_df)} — expected from NaDropper/DuplicateDropper)")

    print("\n--- null counts (post-cleaning) ---")
    print(clean_df.isna().sum())

    print("\n--- head (pre-prediction) ---")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(clean_df.head(10))

    if 'event_id' not in clean_df.columns:
        print("\n*** 'event_id' not found in clean_df — add "
              "df['event_id'] = df['id'] to MatchInferenceRunner.load_input() "
              "before this stage can produce a mergeable prediction output. "
              "Skipping Stage 3. ***")
        return

    print("\n--- Stage 3: predicting event type with best_event_model.pkl ---")
    predicted_df = predict_match_events(clean_df)

    print(f"predicted_df shape: {predicted_df.shape}")
    print("\nPredicted type value counts:")
    print(predicted_df['predicted_type'].value_counts())

    print("\n--- head (with predictions) ---")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        cols_to_show = ['event_id', 'predicted_type'] + [
            c for c in predicted_df.columns if c.startswith('proba_')
        ]
        print(predicted_df[cols_to_show].head(10))

    print("\n--- Merging predicted type back onto full match_events_df ---")
    match_events_df = features_df.merge(
        predicted_df[['event_id', 'predicted_type']].rename(columns={'predicted_type': 'type'}),
        on='event_id',
        how='left',
    )
    print(f"match_events_df shape: {match_events_df.shape}")
    print(f"rows with no prediction (dropped during cleaning): {match_events_df['type'].isna().sum()}")

    out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'inference', f'match_{match_id}_events_with_type.parquet')
    match_events_df.to_parquet(out_path, index=False)
    print(f"\nSaved match_events_df with predicted type to {out_path}")

    print("\n--- Stage 4: MatchShotCleanRunner (shots-only cleaning) ---")
    # Reads the file just saved above (match_{match_id}_events_with_type.parquet),
    # same pattern as ShotCleaningRunner reading master_df.parquet by convention.
    shot_runner = MatchShotCleanRunner(match_id)
    shot_df = shot_runner.run()  # swap for shot_runner.load_output() if run() returns None

    print(f"shot_df shape: {shot_df.shape}")
    print(f"shot_df columns ({len(shot_df.columns)}): {sorted(shot_df.columns.tolist())}")

    print(f"\nRows filtered from {len(match_events_df)} match events down to {len(shot_df)} shots "
          f"(ShotFilter on predicted type=='Shot', then OpponentDataDropper/BallSpeedImputer).")

    print("\n--- null counts (shot_df) ---")
    print(shot_df.isna().sum())

    print("\n--- head (shot_df) ---")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(shot_df.head(10))

    print("\n--- Stage 5: predicting xG with best_xg_model.pkl ---")
    shot_xg_df = predict_shot_xg(shot_df)

    print(f"shot_xg_df shape: {shot_xg_df.shape}")
    print("\n--- head (with predicted_xg) ---")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(shot_xg_df[['event_id', 'team', 'predicted_xg']].head(10))

    print("\n--- Merging predicted_xg back onto match_events_df ---")
    match_events_df = match_events_df.merge(
        shot_xg_df[['event_id', 'predicted_xg']],
        on='event_id',
        how='left',
    )
    print(f"match_events_df shape: {match_events_df.shape}")
    print(f"non-null predicted_xg rows (shots only): {match_events_df['predicted_xg'].notna().sum()}")

    final_out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'inference',
                                   f'match_{match_id}_events_final.parquet')
    match_events_df.to_parquet(final_out_path, index=False)
    print(f"\nSaved match_events_df with type + predicted_xg to {final_out_path}")

    with pd.option_context('display.max_columns', None, 'display.max_rows', 200, 'display.width', None):
        print(match_events_df)

    print("\n--- Match aggregation: predicted winner ---")
    match_result = aggregate_predicted_xg(shot_xg_df)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(match_result)


if __name__ == "__main__":
    main()