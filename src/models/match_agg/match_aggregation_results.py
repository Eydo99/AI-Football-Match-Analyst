from src.models.match_agg.match_aggregation import aggregate_match_xg, build_match_comparison, summarize_win_prediction_accuracy , add_draw_band
from src.models.xg_model.xgboost_model import XGBoostXGModel
from src.runners.shot_clean_runner import ShotCleaningRunner
from pathlib import Path
import numpy as np
import pandas as pd

drop_cols = ['match_id', 'team', 'type', 'ball_x_start', 'ball_y_start']
target_col = 'ends_in_goal'

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "saved_models" / "best_xg_model.pkl"

model = XGBoostXGModel()
model.load(str(MODEL_PATH))


shot_df = ShotCleaningRunner().load_output()
match_agg = aggregate_match_xg(model, shot_df, drop_cols=drop_cols, target_col=target_col)

match_comparison = build_match_comparison(match_agg)

thresholds = np.arange(0.0, 0.51, 0.005)
results = []

for t in thresholds:
    mc = add_draw_band(match_comparison, draw_threshold=t)
    summary = summarize_win_prediction_accuracy(mc)
    results.append({'threshold': t, **summary})

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['overall_accuracy'].idxmax()]

threshold = best_row['threshold']

match_comparison = add_draw_band(match_comparison, draw_threshold=threshold)

print(match_comparison[['match_id', 'team_1', 'team_2', 'total_xg_1', 'total_xg_2',
                         'xg_diff', 'predicted_winner', 'actual_winner', 'correct']])
                         

print(summarize_win_prediction_accuracy(match_comparison))

