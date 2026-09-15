import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss

from src.models.evaluation.base_evaluator import BaseEvaluator


class EventEvaluator(BaseEvaluator):
    """Macro-F1 / weighted-F1 / log loss / confusion matrix, for the 5-class event family."""

    def compute_metrics(self, model, model_name: str) -> dict:
        y_true = model.y_test
        y_pred = model.best_estimator_.predict(model.X_test)
        y_proba = model.predict_proba(model.X_test)
        labels = model.best_estimator_.classes_

        return {
            'model': model_name,
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'log_loss': log_loss(y_true, y_proba, labels=labels),
            'best_params': model.search.best_params_,
            'classification_report': classification_report(y_true, y_pred, digits=3),
            'confusion_matrix': np.round(
                confusion_matrix(y_true, y_pred, labels=labels, normalize='true'), 2
            ),
        }

    def plot(self, model, model_name: str, results: dict) -> None:
        print(f"--- {model_name} ---")
        print(results['classification_report'])
        print(results['confusion_matrix'])

    def _excluded_columns(self) -> set:
        return {'classification_report', 'confusion_matrix'}