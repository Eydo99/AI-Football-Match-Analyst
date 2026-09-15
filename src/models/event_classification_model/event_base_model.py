
from abc import abstractmethod

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV

from src.models.base_model import BaseModel

class EventClassificationBaseModel(BaseModel):
    """Common interface for every event classification model wrapper (5-class target)."""

    def __init__(self):
        super().__init__()
        self.groups_test = None

    def train(self, X=None, y=None, groups=None, n_iter=25,
              scoring='f1_macro', n_splits=5, test_size=0.2):

        if X is None or y is None or groups is None:
            raise ValueError("train() requires X, y, and groups (match_id) for event classification.")

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        self.X_test, self.y_test = X_test, y_test
        self.groups_test = groups.iloc[test_idx]

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        pipeline = self.build_pipeline()
        param_distributions = self.get_param_grid()
        fit_params = self.get_fit_params(X_train, y_train)

        self.search = RandomizedSearchCV(
            pipeline, param_distributions, n_iter=n_iter,
            scoring=scoring, cv=cv, n_jobs=-1, random_state=42,
        )
        self.search.fit(X_train, y_train, groups=groups.iloc[train_idx], **fit_params)
        self.best_estimator_ = self.search.best_estimator_
        return self.best_estimator_

    def get_fit_params(self, X_train, y_train) -> dict:
        """
        Extra kwargs to pass into search.fit(), e.g. {'clf__sample_weight': ...}.

        Default is no extra params (model handles imbalance via class_weight
        in build_pipeline instead). Override in models that need
        sample_weight, e.g. XGBoost — see XGBoostEventModel.
        """
        return {}

    def predict_proba(self, X):
        """Return the full 5-class probability distribution for X."""
        if self.best_estimator_ is None:
            raise RuntimeError("Call train() before predict_proba().")
        return self.best_estimator_.predict_proba(X)

    def evaluate(self, plot: bool = True) -> dict:
        from src.models.event_evaluation import evaluate_event_model
        return evaluate_event_model(self, self.__class__.__name__, plot=plot)