import pandas as pd


def aggregate_match_xg(model, shot_df, drop_cols, target_col='ends_in_goal',
                        match_col='match_id', team_col='team'):
    """Predict xG for every shot, then aggregate to one row per (match, team):
    shot count, actual goals, and total predicted xG.
    """
    feature_cols = [c for c in shot_df.columns
                     if c not in drop_cols + [target_col]]
    X_all = shot_df[feature_cols]

    shot_df = shot_df.copy()
    shot_df['predicted_xg'] = model.best_estimator_.predict_proba(X_all)[:, 1]

    match_agg = shot_df.groupby([match_col, team_col]).agg(
        shots=(target_col, 'size'),
        actual_goals=(target_col, 'sum'),
        total_xg=('predicted_xg', 'sum'),
    ).reset_index()

    return match_agg


def build_match_comparison(match_agg, match_col='match_id', team_col='team'):
    """Turn the long (match, team) aggregation into one row per match with
    both teams side by side, plus xG diff, actual goal diff, and a
    predicted-vs-actual winner comparison.

    Matches without exactly two teams (data issue, or a team with zero
    shots not present in shot_df) are dropped and reported separately.
    """
    valid_matches = match_agg.groupby(match_col).filter(lambda g: len(g) == 2)
    dropped = match_agg[~match_agg[match_col].isin(valid_matches[match_col])]
    if not dropped.empty:
        print(f"Dropped {dropped[match_col].nunique()} match(es) without exactly "
              f"two teams in shot_df (likely a team recorded zero shots).")

    # assign a stable team_1 / team_2 position within each match
    valid_matches = valid_matches.sort_values([match_col, team_col]).copy()
    valid_matches['team_slot'] = valid_matches.groupby(match_col).cumcount() + 1

    wide = valid_matches.pivot(index=match_col, columns='team_slot',
                                values=[team_col, 'total_xg', 'actual_goals']).reset_index()
    wide.columns = [match_col] + [f'{col}_{slot}' for col, slot in wide.columns[1:]]

    wide['xg_diff'] = wide['total_xg_1'] - wide['total_xg_2']
    wide['actual_diff'] = wide['actual_goals_1'] - wide['actual_goals_2']

    wide['predicted_winner'] = wide.apply(
        lambda r: r[f'{team_col}_1'] if r['xg_diff'] > 0
        else (r[f'{team_col}_2'] if r['xg_diff'] < 0 else 'Draw (xG tied)'),
        axis=1,
    )
    wide['actual_winner'] = wide.apply(
        lambda r: r[f'{team_col}_1'] if r['actual_diff'] > 0
        else (r[f'{team_col}_2'] if r['actual_diff'] < 0 else 'Draw'),
        axis=1,
    )
    wide['correct'] = wide['predicted_winner'] == wide['actual_winner']

    return wide


def summarize_win_prediction_accuracy(match_comparison: pd.DataFrame) -> dict:
    """Accuracy of the xG-based predicted winner vs the actual result,
    computed both over all matches and over decisive (non-draw) matches
    only, since xG rarely predicts an exact draw.
    """
    decisive = match_comparison[match_comparison['actual_winner'] != 'Draw']
    return {
        'total_matches': len(match_comparison),
        'overall_accuracy': match_comparison['correct'].mean(),
        'decisive_matches': len(decisive),
        'decisive_accuracy': decisive['correct'].mean() if len(decisive) else float('nan'),
        'draw_matches': len(match_comparison) - len(decisive),
        'draw_accuracy': match_comparison[match_comparison['actual_winner'] == 'Draw']['correct'].mean() if len(match_comparison[match_comparison['actual_winner'] == 'Draw']) else float('nan')
    }

def add_draw_band(match_comparison, draw_threshold: float):
    """Recompute predicted_winner allowing a 'Draw' call when the two
    teams' total xG are within draw_threshold of each other, instead of
    only on an exact tie.
    """
    match_comparison = match_comparison.copy()
    match_comparison['predicted_winner'] = match_comparison.apply(
        lambda r: 'Draw' if abs(r['xg_diff']) <= draw_threshold
        else (r['team_1'] if r['xg_diff'] > 0 else r['team_2']),
        axis=1,
    )
    match_comparison['correct'] = match_comparison['predicted_winner'] == match_comparison['actual_winner']
    return match_comparison