
from abc import ABC, abstractmethod

import pandas as pd

class BaseEvaluator(ABC):
    """Common interface every family's evaluator implements."""

    @abstractmethod
    def compute_metrics(self, model, model_name: str) -> dict:
        """Return this family's metric dict for a fitted model."""
        raise NotImplementedError

    @abstractmethod
    def plot(self, model, model_name: str, results: dict) -> None:
        """Family-specific plot/printout (calibration curve, confusion matrix, ...)."""
        raise NotImplementedError

    def evaluate(self, model, model_name: str, plot: bool = True) -> dict:
        """Shared entry point: compute metrics, optionally plot, return the dict."""
        if model.best_estimator_ is None:
            raise RuntimeError(f"{model_name}: call train() before evaluate()")

        results = self.compute_metrics(model, model_name)
        if plot:
            self.plot(model, model_name, results)
        return results

    def compare(self, results_list: list) -> pd.DataFrame:
        """Shared table builder. Override _excluded_columns() to drop non-tabular fields."""
        excluded = self._excluded_columns()
        return pd.DataFrame(
            [{k: v for k, v in r.items() if k not in excluded} for r in results_list]
        ).set_index('model')

    def _excluded_columns(self) -> set:
        """Columns to drop from compare()'s table (e.g. a confusion matrix array)."""
        return set()