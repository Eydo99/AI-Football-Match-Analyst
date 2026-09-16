from src.models import LogisticRegressionModel, RandomForestModel, XGBoostModel
from src.models.evaluation.xg_evaluator import XGEvaluator
from pathlib import Path

drop_cols = ['match_id', 'team', 'type' , 'ball_x_start', 'ball_y_start']
target_col = 'ends_in_goal'

evaluator = XGEvaluator()

lr = LogisticRegressionModel()
lr.train(drop_cols=drop_cols, target_col=target_col)

rf = RandomForestModel()
rf.train(drop_cols=drop_cols, target_col=target_col)

xgb = XGBoostModel()
xgb.train(drop_cols=drop_cols, target_col=target_col)

results_lr = evaluator.compute_metrics(lr, 'Logistic Regression')
results_rf = evaluator.compute_metrics(rf, 'Random Forest')
results_xgb = evaluator.compute_metrics(xgb, 'XGBoost')

results = [
    results_lr,
    results_rf,
    results_xgb
]

evaluator.plot(lr, 'Logistic Regression',results_lr)
evaluator.plot(rf, 'Random Forest',results_rf)
evaluator.plot(xgb, 'XGBoost',results_xgb)

evaluator.plot_calibration_reference()


for result in results:
    print(f"\n{result['model']} best params: {result['best_params']}")
    print(f"{result['model']} log loss: {result['log_loss']:.4f}")
    print(f"{result['model']} brier score: {result['brier_score']:.4f}")
    print(f"{result['model']} roc auc: {result['roc_auc']:.4f}")

print("\nBest model based on log loss: ")

min_log_loss_model = min(results, key=lambda x: x['log_loss'])

print(f"\nModel with lowest log loss: {min_log_loss_model['model']}")
print(f"Best params: {min_log_loss_model['best_params']}")
print(f"Log loss: {min_log_loss_model['log_loss']:.4f}")
print(f"Brier score: {min_log_loss_model['brier_score']:.4f}")
print(f"ROC AUC: {min_log_loss_model['roc_auc']:.4f}")


models_dict = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}
best_model = models_dict[min_log_loss_model['model']]

MODEL_DIR = Path(__file__).resolve().parents[1] / "saved_models"   # -> src/models/saved_models
MODEL_PATH = MODEL_DIR / "best_xg_model.pkl"

best_model.save(str(MODEL_PATH))
print(f"Saved best model ({min_log_loss_model['model']}) to {MODEL_PATH}")