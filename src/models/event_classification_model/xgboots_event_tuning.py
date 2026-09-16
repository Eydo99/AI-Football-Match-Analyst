
from functools import partial

from src.runners.train_runners.event_clean_runner import EventCleaningRunner
from src.data.subsampling import subsample_by_match, check_min_class_count, describe_balance
from src.models.event_classification_model import XGBoostEventModel
from src.models.evaluation import EventEvaluator

TARGET_COL = 'event_class'
GROUP_COL = 'match_id'


TRAIN_MATCH_FRAC = 0.3
N_SPLITS = 5
N_ITER = 35

NARROW_GRID = {
    'clf__n_estimators': [400, 600, 800],
    'clf__max_depth': [8, 10, 12, 14],
    'clf__learning_rate': [0.03, 0.05, 0.1],
    'clf__subsample': [0.7, 0.8],
    'clf__colsample_bytree': [0.7, 0.8],
    'clf__min_child_weight': [2, 3, 5],
}


def load_X_y_groups():
    events_df = EventCleaningRunner().load_output()
    y = events_df[TARGET_COL]
    groups = events_df[GROUP_COL]
    X = events_df.drop(columns=[TARGET_COL, GROUP_COL])
    return X, y, groups


def main():
    X, y, groups = load_X_y_groups()
    describe_balance(y, 'Full dataset')

    subsample_fn = partial(subsample_by_match, match_frac=TRAIN_MATCH_FRAC)

    xgb = XGBoostEventModel()
    xgb.get_param_grid = lambda: NARROW_GRID  # override the wider original grid

    print("starting narrowed xgb search")
    xgb.train(X=X, y=y, groups=groups, n_iter=N_ITER, n_splits=N_SPLITS,
               subsample_fn=subsample_fn)
    print("finished narrowed xgb search")

    describe_balance(xgb.y_test, 'Held-out test set (full size, untouched)')
    check_min_class_count(xgb.y_test, n_splits=1)

    evaluator = EventEvaluator()
    result = evaluator.evaluate(xgb, 'XGBoost (narrowed search)')

    print(f"\nBest params: {result['best_params']}")
    print(f"macro-F1: {result['macro_f1']:.4f}")
    print(f"weighted-F1: {result['weighted_f1']:.4f}")
    print(f"log loss: {result['log_loss']:.4f}")

    return result


if __name__ == "__main__":
    main()