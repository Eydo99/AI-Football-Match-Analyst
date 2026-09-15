from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.models.evaluation.base_evaluator import BaseEvaluator


class XGEvaluator(BaseEvaluator):
    """Log-loss / Brier / ROC-AUC + calibration curve, for the binary xG family."""

    def compute_metrics(self, model, model_name: str) -> dict:
        y_true = model.y_test
        y_pred_proba = model.predict_proba(model.X_test)

        return {
            'model': model_name,
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'best_params': model.search.best_params_,
        }

    def plot(self, model, model_name: str, results: dict) -> None:
        import matplotlib.pyplot as plt

        y_true = model.y_test
        y_pred_proba = model.predict_proba(model.X_test)
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)

    @staticmethod
    def plot_calibration_reference():
        import matplotlib.pyplot as plt

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
        plt.xlabel('Predicted probability')
        plt.ylabel('Actual frequency of goals')
        plt.title('Calibration Curve — All Models')
        plt.legend()
        plt.show()