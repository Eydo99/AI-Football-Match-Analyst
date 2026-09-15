import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def evaluate_model(model, model_name: str, plot: bool = True) -> dict:
    
    if model.best_estimator_ is None:
        raise RuntimeError(f"{model_name}: call train() before evaluate_model()")

    y_true = model.y_test
    y_pred_proba = model.predict_proba(model.X_test)

    results = {
        'model': model_name,
        'log_loss': log_loss(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'best_params': model.grid_search.best_params_,
    }

    if plot:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)

    return results


def plot_calibration_reference():
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual frequency of goals')
    plt.title('Calibration Curve — All Models')
    plt.legend()
    plt.show()


def compare_models(results_list: list) -> pd.DataFrame:
    return pd.DataFrame(results_list).set_index('model')