"""
Trains all three event classification models and compares them via EventEvaluator.

Subsampling: the full events_df (~2.27M rows) makes the RandomForest and
XGBoost searches impractical on 16GB. Neither estimator supports incremental
fitting, so we subsample once up front rather than trying to train on chunks.
Tune MATCH_FRAC / MAJORITY_CAP down if it is still too slow, up if you have
headroom.
"""
from src.runners.event_clean_runner import EventCleaningRunner
from src.data.subsampling import (
    subsample_by_match, cap_majority_classes, describe_balance,
)
from src.models.event_classification_model import (
    LogisticRegressionEventModel,
    RandomForestEventModel,
    XGBoostEventModel,
)
from src.models.evaluation import EventEvaluator

TARGET_COL = 'event_class'
GROUP_COL = 'match_id'

MATCH_FRAC = 0.2       # keep 20% of the 1,005 matches
MAJORITY_CAP = 150_000  # cap Pass and Carry; Shot/Foul kept in full


def load_X_y_groups():
    events_df = EventCleaningRunner().load_output()
    y = events_df[TARGET_COL]
    groups = events_df[GROUP_COL]
    X = events_df.drop(columns=[TARGET_COL, GROUP_COL])
    return X, y, groups


def main():
    X, y, groups = load_X_y_groups()
    describe_balance(y, 'Full dataset')

    X, y, groups = subsample_by_match(X, y, groups, match_frac=MATCH_FRAC)
    X, y, groups = cap_majority_classes(X, y, groups, cap=MAJORITY_CAP)
    describe_balance(y, 'After subsampling (used for training)')

    print("Starting lr training")
    lr = LogisticRegressionEventModel()
    lr.train(X=X, y=y, groups=groups, n_iter=6)
    print("finished lr training")

    print("Starting rf training")
    rf = RandomForestEventModel()
    rf.train(X=X, y=y, groups=groups, n_iter=40)
    print("finished rf training")

    print("started xgboost training")
    xgb = XGBoostEventModel()
    xgb.train(X=X, y=y, groups=groups, n_iter=40)
    print("ended xgboost training")

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