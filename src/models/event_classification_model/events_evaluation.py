
from functools import partial

from src.runners.train_runners.event_clean_runner import EventCleaningRunner
from src.data.subsampling import subsample_by_match, check_min_class_count, describe_balance
from src.models.event_classification_model import (
    LogisticRegressionEventModel,
    RandomForestEventModel,
    XGBoostEventModel,
)
from src.models.evaluation import EventEvaluator

TARGET_COL = 'event_class'
GROUP_COL = 'match_id'

# Applied to the TRAINING split only, inside train() — the test set is
# always the full, untouched ~20% split. Adjust to trade off speed vs. how
# much training data each model actually sees.
TRAIN_MATCH_FRAC = 0.3
N_SPLITS = 5


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


    print("starting lr")
    lr = LogisticRegressionEventModel()
    lr.train(X=X, y=y, groups=groups, n_iter=6, n_splits=N_SPLITS,
              subsample_fn=subsample_fn)
    print("finished lr")

    print("starting rf")
    rf = RandomForestEventModel()
    rf.train(X=X, y=y, groups=groups, n_iter=40, n_splits=N_SPLITS,
              subsample_fn=subsample_fn)
    print("finished rf")

    print("starting xgb")
    xgb = XGBoostEventModel()
    xgb.train(X=X, y=y, groups=groups, n_iter=40, n_splits=N_SPLITS,
               subsample_fn=subsample_fn)
    print("finished xgb")


    describe_balance(lr.y_test, 'Held-out test set (full size, untouched)')
    check_min_class_count(lr.y_test, n_splits=1)

    evaluator = EventEvaluator()
    results = [
        evaluator.evaluate(lr, 'Logistic Regression'),
        evaluator.evaluate(rf, 'Random Forest'),
        evaluator.evaluate(xgb, 'XGBoost'),
    ]

    print(evaluator.compare(results))
    for result in results:
        print(f"\n{result['model']} best params: {result['best_params']}")
        print(f"{result['model']} macro-F1: {result['macro_f1']:.4f}")
        print(f"{result['model']} weighted-F1: {result['weighted_f1']:.4f}")
        print(f"{result['model']} log loss: {result['log_loss']:.4f}")

    best = max(results, key=lambda r: r['macro_f1'])
    print(f"\nBest model based on macro-F1: {best['model']} ({best['macro_f1']:.4f})")
    return results


if __name__ == "__main__":
    main()