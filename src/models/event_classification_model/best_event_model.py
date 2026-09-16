import os
import pickle

from src.runners.train_runners.event_clean_runner import EventCleaningRunner
from src.models.event_classification_model import XGBoostEventModel
from src.models.evaluation import EventEvaluator


TARGET_COL = 'event_class'
GROUP_COL = 'match_id'

# Best params found by the narrowed RandomizedSearchCV (TRAIN_MATCH_FRAC=0.5,
# N_ITER=60) — macro-F1 0.9263 on that run. Locking these in for the final,
# full-data fit.
BEST_PARAMS = {
    'clf__subsample': 0.7,
    'clf__n_estimators': 400,
    'clf__min_child_weight': 2,
    'clf__max_depth': 14,
    'clf__learning_rate': 0.1,
    'clf__colsample_bytree': 0.8,
}

# Mirrors src/models/xg_model/saved_models/best_xg_model.pkl's layout
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
OUTPUT_PATH = os.path.join(SAVE_DIR, 'best_event_model.pkl')


def load_X_y_groups():
    events_df = EventCleaningRunner().load_output()
    y = events_df[TARGET_COL]
    groups = events_df[GROUP_COL]
    X = events_df.drop(columns=[TARGET_COL, GROUP_COL])
    return X, y, groups


def main():
    X, y, groups = load_X_y_groups()

    xgb = XGBoostEventModel()

    pinned_grid = {k: [v] for k, v in BEST_PARAMS.items()}
    xgb.get_param_grid = lambda: pinned_grid  # monkey-patch for this one run

    xgb.train(
        X=X, y=y, groups=groups,
        n_iter=1,           # nothing to search, pinned_grid has 1 combo
        n_splits=5,
        subsample_fn=None,  # full training split — no subsampling this time
    )

    print("Refit on full training data complete.")
    print("Best estimator params:", xgb.best_estimator_.get_params()['clf'])

    del xgb.get_param_grid

    evaluator = EventEvaluator()
    results = evaluator.evaluate(xgb, 'XGBoost (final, full-data fit)')

    print(f"\nmacro-F1: {results['macro_f1']:.4f}")
    print(f"weighted-F1: {results['weighted_f1']:.4f}")
    print(f"log loss: {results['log_loss']:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(xgb, f)

    print(f"Saved best event model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()