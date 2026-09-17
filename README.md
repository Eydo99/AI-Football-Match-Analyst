# ⚽ AI Football Match Analyst

An end-to-end **expected goals (xG)** pipeline built on StatsBomb open event
data — from raw event cleaning, through shot-level xG modeling and
match-level result prediction (win or draw), to a live Streamlit app that
runs the whole thing on matches the models have never seen.

## What this project does

1. **Cleans and engineers features** from raw StatsBomb event data at the
   shot level (location, angle, trajectory, pressure, and team-shape
   features).
2. **Trains an XGBoost classifier** to estimate the probability that a given
   shot ends in a goal (its xG value).
3. **Trains a separate event-type classifier** that predicts each event's
   type (Shot, Pass, ...) from its features — needed because the real
   StatsBomb `type` column has to be dropped before feature engineering to
   avoid leaking the answer on genuinely new matches.
4. **Aggregates shot-level xG to the match level** to compare two teams'
   total xG and predict the match outcome — a win for one side or a draw —
   evaluated against real historical results.
5. **Reconstructs the same pipeline for genuinely new matches**, using the
   event-type classifier to find the shots in a match no one has labeled.
6. **Serves it all through a Streamlit app** ("Match xG Explorer") that lets
   you browse competitions and seasons that were *not* used in training,
   pick a match, and see it scored live.

## Data

Event data comes from [StatsBomb's open data](https://github.com/statsbomb/open-data)
via the [`statsbombpy`](https://github.com/statsbomb/statsbombpy) package.
Training used a fixed set of competitions and seasons — La Liga (id 11),
the FIFA World Cup (id 43), and the UEFA Champions League (id 16), each
restricted to specific season IDs. The live app deliberately excludes
exactly those (competition, season) pairs when listing what's browsable, so
you're always scoring a match the models haven't seen.

## Project structure

```
AI-Football-Match-Analyst/
├── app.py                     # Streamlit app: "Match xG Explorer"
├── live_match_loader.py       # Sidebar data: competitions/seasons/matches not used in training
├── style.css                  # App styling
├── main.py                    # Reserved entry point (currently unused)
├── data/
│   ├── raw/                   # Raw StatsBomb pulls (gitignored)
│   └── processed/             # Cleaned parquet outputs (events, master, shot-level, xG)
├── notebooks/                 # Exploratory analysis
├── src/
│   ├── features/
│   │   ├── pipeline.py         # Transformer/Pipeline framework
│   │   └── transformers/
│   │       ├── cleaning/       # Location splitting, rescaling, imputing, filtering, pruning, ...
│   │       └── features/       # Geometry, trajectory, pressure, motion, team-shape features
│   ├── models/
│   │   ├── xg_model/            # XGBoost xG model wrapper
│   │   ├── match_agg/           # Match-level aggregation & win-prediction evaluation
│   │   └── saved_models/        # best_xg_model.pkl, best_event_model.pkl
│   └── runners/
│       ├── base_runner.py       # Shared read -> pipeline -> write convention
│       ├── train_runners/       # Offline pipeline: builds the training parquet files
│       └── match_test_runners/  # Live-inference pipeline: same pipeline, run per match_id
└── tests/
```

## Pipeline in detail

**Offline (training) pipeline** — turns the full StatsBomb event history
into a clean, shot-level feature table:
- Location splitting, coordinate rescaling, time parsing, and freeze-frame
  extraction on raw events.
- Feature engineering: distance/angle to goal, whether a shot is in the
  penalty box, ball trajectory length/angle, distance to the nearest
  defender, defenders within 3m, ball speed, team-centroid distance, and
  open angle to goal.
- Cleaning: duplicate removal, column pruning, imputing missing ball speed
  and trajectory values, and dropping rows missing core geometric features.
- Filtering down to shots only, dropping opponent-only columns, and writing
  the final `shot_df_clean.parquet` used to train the xG model.

**Match aggregation** (`src/models/match_agg/`) — sums each team's
predicted shot xG per match, compares the two totals, and predicts the
match outcome: a win for one side, or a draw.

**Live inference pipeline** (`src/runners/match_test_runners/`) — the same
transformations, reconstructed to run on a single `match_id` fetched live:
1. `MatchInferenceRunner` — fetches the match's events and runs the same
   feature-engineering pipeline (with `type` and outcome-revealing columns
   dropped up front, since they don't exist for a genuinely unscored match).
2. `MatchEventCleanRunner` — cleans the engineered features the same way
   the training pipeline does.
3. **Event-type prediction** — since the real `type` column was removed,
   the event classification model (below) predicts each event's type from
   its features, which is what lets the next step find the shots.
4. `MatchShotCleanRunner` — filters to predicted-`Shot` rows and finishes
   cleaning them into the exact feature set the xG model expects.
5. The xG model scores each shot, and the same match-aggregation logic as
   above produces a live predicted result.

## Models

### Expected goals (xG) model

An XGBoost classifier (selected via `GridSearchCV`, saved as
`best_xg_model.pkl`) trained on cleaned, shot-level features to predict
`ends_in_goal` per shot. Its output probability is used directly as that
shot's xG value, and shot-level xG values are summed per team to drive
match aggregation.

### Event classification model

A second classifier (saved as `best_event_model.pkl`) trained to predict
an event's *type* — Shot, Pass, Dribble, and so on — from its engineered
features, using the class mapping in
`src/features/transformers/cleaning/target_encoder.py`. It exists purely
for the live-inference path: the real StatsBomb `type` column is dropped
before feature engineering on a new match (otherwise the pipeline would be
leaking the exact thing it's meant to infer), so this model reconstructs
"which events are shots" from features alone, which is what lets
`MatchShotCleanRunner` filter down to shots for the xG model to score.

## Getting started

```bash
git clone https://github.com/Eydo99/AI-Football-Match-Analyst.git
cd AI-Football-Match-Analyst

python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows

pip install -r requirements.txt          # or requirements-dev.txt for tests/plots
```

Run the live app:

```bash
streamlit run app.py
```

Run the historical match-level win-prediction evaluation:

```bash
python -m src.models.match_agg.match_aggregation_results
```

## Known limitations

- **Live pipeline parity isn't fully verified end-to-end.** The live
  inference runners mirror the training pipeline's logic but should be
  validated by running them against a match already in the training data
  and diffing the output column-for-column against the offline pipeline.
- **Depends on StatsBomb's public API availability** for both training
  data refreshes and live match lookups.

## Future work

- **Computer-vision-based xG estimation.** StatsBomb's event data only
  covers a limited set of competitions and seasons. A natural extension is
  a CV-based shot-outcome/xG estimator that works directly from match
  footage or broadcast screenshots — using an object detector (e.g.
  YOLOv8) to extract visual features like shot angle, defender positions,
  and goalkeeper positioning, and feeding those into the same class of
  classical ML models (XGBoost, Random Forest, SVM, logistic regression)
  used here. This would let the pipeline estimate xG for any match with
  video available, not just competitions StatsBomb has annotated —
  and could eventually be combined with the existing event-based features
  in an ensemble for matches where both are available.
- **Broader competition coverage** and a proper multi-league generalization
  test, rather than evaluating within the same competitions used for
  training.
- **A lightweight API layer** alongside the Streamlit app, so the pipeline
  can be called programmatically rather than only through the UI.

## Tech stack

Python · pandas · NumPy · XGBoost · scikit-learn · statsbombpy · pyarrow ·
Streamlit

## Acknowledgments

Built on [StatsBomb's open event data](https://github.com/statsbomb/open-data).