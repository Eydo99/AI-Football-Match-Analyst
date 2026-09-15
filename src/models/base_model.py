"""
Shared base class for all xG model wrappers.

Subclasses only need to implement `build_pipeline` and `get_param_grid`;
`train` and `predict_proba` are shared here so partners don't duplicate
the GridSearchCV plumbing in every model file.
"""
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV


class BaseXGModel(ABC):
    """Common interface every xG model wrapper implements."""

    def __init__(self):
        self.grid_search = None
        self.best_estimator_ = None

    @abstractmethod
    def build_pipeline(self):
        """Return an sklearn Pipeline for this model.

        Implemented per-model in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_param_grid(self):
        """Return the hyperparameter grid dict for GridSearchCV.

        Implemented per-model in the subclass.
        """
        raise NotImplementedError

    def train(self, X_train, y_train, scoring='neg_log_loss', cv=5):
        """Run GridSearchCV over this model's pipeline and store the result.

        Shared logic - no TODO here, this should work as-is once a
        subclass fills in build_pipeline() and get_param_grid().
        """
        pipeline = self.build_pipeline()
        param_grid = self.get_param_grid()
        self.grid_search = GridSearchCV(
            pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1
        )
        self.grid_search.fit(X_train, y_train)
        self.best_estimator_ = self.grid_search.best_estimator_
        return self.best_estimator_

    def predict_proba(self, X):
        """Return predicted goal probabilities for X.

        Shared logic - no TODO here. Requires train() to have been
        called first.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Call train() before predict_proba().")
        return self.best_estimator_.predict_proba(X)[:, 1]
