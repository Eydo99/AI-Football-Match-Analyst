from src.models import LogisticRegressionModel, RandomForestModel, XGBoostModel
from src.models.evaluation import (
    evaluate_model, compare_models, plot_calibration_reference,
)

drop_cols = ['match_id', 'team', 'type' , 'ball_x_start', 'ball_y_start']
target_col = 'ends_in_goal'

lr = LogisticRegressionModel()
lr.train(drop_cols=drop_cols, target_col=target_col)

rf = RandomForestModel()
rf.train(drop_cols=drop_cols, target_col=target_col)

xgb = XGBoostModel()
xgb.train(drop_cols=drop_cols, target_col=target_col)

results = [
    evaluate_model(lr, 'Logistic Regression'),
    evaluate_model(rf, 'Random Forest'),
    evaluate_model(xgb, 'XGBoost'),
]

plot_calibration_reference()
print(compare_models(results))

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