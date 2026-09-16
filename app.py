"""
Streamlit frontend for the xG pipeline - live version.

Browses StatsBomb competitions/seasons/matches that were NOT used for
training (see live_match_loader.TRAINED_COMPETITION_IDS), fetches a
selected match's events on demand, runs them through the Stage 2
feature pipeline, and predicts xG - all in-process, no separate
backend needed.

Run with:
    streamlit run app.py
(from the project root, so the `src.*` imports resolve correctly.)
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.models.xg_model.xgboost_model import XGBoostXGModel
from live_match_loader import (
    get_available_competitions, get_matches_for, fetch_and_process_match,
)

MODEL_PATH = Path(__file__).resolve().parent / "src" / "models" / "saved_models" / "best_xg_model.pkl"
DROP_COLS = ['match_id', 'team', 'type', 'ball_x_start', 'ball_y_start']
TARGET_COL = 'ends_in_goal'

st.set_page_config(page_title="Match xG Explorer", layout="wide")


# ---------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------

def load_css(path: Path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model() -> XGBoostXGModel:
    model = XGBoostXGModel()
    model.load(str(MODEL_PATH))
    return model


@st.cache_data
def cached_competitions() -> pd.DataFrame:
    return get_available_competitions()


@st.cache_data
def cached_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    return get_matches_for(competition_id, season_id)


@st.cache_data(show_spinner=False)
def cached_fetch_and_process(match_id: int):
    return fetch_and_process_match(match_id)


def predict_xg(model: XGBoostXGModel, shots: pd.DataFrame) -> pd.Series:
    # Ask the fitted pipeline exactly which columns (and order) it was
    # trained on, rather than inferring the feature set by excluding
    # DROP_COLS + TARGET_COL. The live feature pipeline is a reconstruction
    # (see live_match_loader.py's docstring) and currently leaves a large
    # number of raw, non-numeric StatsBomb event columns in the output
    # (pass_outcome, foul_committed_card, tactics, etc.) that the fixed
    # drop-list was never meant to cover, and XGBoost rejects anything
    # object-dtype outright.
    feature_cols = list(model.best_estimator_.feature_names_in_)
    X = shots[feature_cols]
    return pd.Series(model.best_estimator_.predict_proba(X)[:, 1], index=shots.index)


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------

load_css(Path(__file__).resolve().parent / "style.css")

st.title("⚽ Match xG Explorer")
st.caption("Browsing competitions/seasons not used in training - matches are fetched and processed live.")

model = load_model()
competitions = cached_competitions()

required_cols = {'competition_id', 'competition_name', 'season_id', 'season_name'}
missing = required_cols - set(competitions.columns)
if missing:
    st.error(
        f"sb.competitions() is missing expected columns: {missing}. "
        f"Columns found: {list(competitions.columns)}"
    )
    st.stop()

with st.sidebar:
    st.header("1. Competition")
    comp_options = competitions[['competition_id', 'competition_name']].drop_duplicates()
    comp_options = comp_options.sort_values('competition_name')
    selected_competition_id = st.selectbox("Competition", comp_options['competition_id'].tolist(),format_func=lambda cid: comp_options.set_index('competition_id').loc[cid, 'competition_name'])

    st.header("2. Season")
    season_options = competitions[competitions['competition_id'] == selected_competition_id][
        ['season_id', 'season_name']
    ].drop_duplicates().sort_values('season_name', ascending=False)
    selected_season_id = st.selectbox(
        "Season", season_options['season_id'].tolist(),
        format_func=lambda sid: season_options.set_index('season_id').loc[sid, 'season_name'],
    )

    st.header("3. Match")
    matches = cached_matches(selected_competition_id, selected_season_id)
    match_options = matches['match_id'].tolist()

    def match_label(mid):
        row = matches[matches['match_id'] == mid].iloc[0]
        return f"{row['home_team']} vs {row['away_team']}"

    selected_match = st.selectbox("Match", match_options, format_func=match_label)

with st.spinner("Fetching match events and building features..."):
    shot_features, display_info = cached_fetch_and_process(selected_match)

if shot_features.empty:
    st.warning("No shots found for this match after processing.")
    st.stop()

shot_features['predicted_xg'] = predict_xg(model, shot_features)

# merge in player/minute/second for display only - not used by the model
shots = shot_features.merge(display_info, on='id', how='left')

teams = shots['team'].unique()
if len(teams) != 2:
    st.warning(f"Expected 2 teams for this match, found {len(teams)}: {list(teams)}")

# ---------------------------------------------------------------------
# Per-team summary
# ---------------------------------------------------------------------

team_summary = shots.groupby('team').agg(
    shots=(TARGET_COL, 'size'),
    actual_goals=(TARGET_COL, 'sum'),
    total_xg=('predicted_xg', 'sum'),
).reset_index()

col1, col2 = st.columns(2)
for col, (_, row) in zip([col1, col2], team_summary.iterrows()):
    with col:
        st.subheader(row['team'])
        m1, m2, m3 = st.columns(3)
        m1.metric("Shots", int(row['shots']))
        m2.metric("Actual goals", int(row['actual_goals']))
        m3.metric("Total xG", f"{row['total_xg']:.2f}")

st.divider()

# ---------------------------------------------------------------------
# Predicted result banner
# ---------------------------------------------------------------------

if len(team_summary) == 2:
    t1, t2 = team_summary.iloc[0], team_summary.iloc[1]
    xg_diff = t1['total_xg'] - t2['total_xg']

    if xg_diff > 0:
        predicted_winner = t1['team']
    elif xg_diff < 0:
        predicted_winner = t2['team']
    else:
        predicted_winner = None

    actual_diff = t1['actual_goals'] - t2['actual_goals']
    if actual_diff > 0:
        actual_winner = t1['team']
    elif actual_diff < 0:
        actual_winner = t2['team']
    else:
        actual_winner = None

    if predicted_winner:
        st.markdown(
            f'<div class="winner-banner">🔮 Predicted winner (by total xG): '
            f'<b>{predicted_winner}</b> ({abs(xg_diff):.2f} xG margin)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="draw-banner">🔮 Predicted result: Draw (xG tied)</div>', unsafe_allow_html=True)

    if actual_winner:
        st.markdown(
            f'<div class="winner-banner">✅ Actual result: <b>{actual_winner}</b> won '
            f'({int(t1["actual_goals"])} - {int(t2["actual_goals"])})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="draw-banner">✅ Actual result: Draw '
            f'({int(t1["actual_goals"])} - {int(t2["actual_goals"])})</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------
# Total xG comparison chart
# ---------------------------------------------------------------------

st.subheader("Total xG by team")
st.bar_chart(team_summary.set_index('team')['total_xg'])

st.divider()

# ---------------------------------------------------------------------
# Per-shot table - now with player and minute
# ---------------------------------------------------------------------

st.subheader("Shot-by-shot xG")

shots_display = shots.copy()
shots_display['Minute'] = shots_display.apply(
    lambda r: f"{int(r['minute'])}:{int(r['second']):02d}" if pd.notna(r['minute']) else "-",
    axis=1,
)

display_cols = ['player', 'Minute', 'team', 'predicted_xg', TARGET_COL]
st.dataframe(
    shots_display[display_cols]
    .rename(columns={'player': 'Player', 'team': 'Team', 'predicted_xg': 'Predicted xG', TARGET_COL: 'Goal?'})
    .sort_values('Predicted xG', ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
)