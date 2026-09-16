"""
This class knows NOTHING about xG (binary) or event classification
(multiclass) specifics. It only defines the contract:

    build_pipeline()  -> sklearn Pipeline               (abstract)
    get_param_grid()  -> dict                            (abstract)
    train(...)        -> fitted best estimator           (abstract)
    evaluate(...)      -> dict of metrics                (abstract)

`train` and `evaluate` are abstract here (not shared) because the two
task families need genuinely different logic:
  - xG: binary target, plain train_test_split, GridSearchCV, log-loss/
    ROC-AUC/Brier evaluation.
  - Event classification: multiclass target, match-grouped split,
    RandomizedSearchCV, macro-F1/confusion-matrix evaluation.
"""

import pickle
from abc import ABC, abstractmethod
import os


class BaseModel(ABC):
    """Common interface every model wrapper in the project implements."""

    def __init__(self):
        self.search = None
        self.best_estimator_ = None
        self.X_test = None
        self.y_test = None

    @abstractmethod
    def build_pipeline(self):
        """Return a fresh, unfitted sklearn Pipeline for this model."""
        raise NotImplementedError

    @abstractmethod
    def get_param_grid(self):
        """Return the hyperparameter search space for this model."""
        raise NotImplementedError

    @abstractmethod
    def train(self, X=None, y=None, **kwargs):
        """Fit the search over build_pipeline()/get_param_grid() and store the result."""
        raise NotImplementedError

    @abstractmethod
    def get_evaluator(self):
        """Return this family's BaseEvaluator instance (XGEvaluator, EventEvaluator, ...)."""
        raise NotImplementedError

    def evaluate(self, plot: bool = True) -> dict:
        return self.get_evaluator().evaluate(self, self.__class__.__name__, plot=plot)


    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Call train() before predict().")
        return self.best_estimator_.predict(X)

    def save(self, path: str) -> None:
        if self.best_estimator_ is None:
            raise RuntimeError("Call train() before save().")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.best_estimator_, f)

    def load(self, path: str) -> None:
        """Load a previously-trained estimator, skipping train() entirely."""
        with open(path, "rb") as f:
            self.best_estimator_ = pickle.load(f)