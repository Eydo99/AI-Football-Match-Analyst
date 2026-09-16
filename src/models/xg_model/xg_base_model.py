"""
Shared logic for the xG (probability-that-a-shot-becomes-a-goal) model family.

This is exactly the training/prediction plumbing that used to live in
base_model.py, moved down one level now that BaseModel is generic.
Subclasses (LogisticRegressionXGModel, RandomForestXGModel, XGBoostXGModel)
only implement build_pipeline() and get_param_grid().
"""
from sklearn.model_selection import GridSearchCV, train_test_split

from src.runners.train_runners.shot_clean_runner import ShotCleaningRunner
from src.models.base_model import BaseModel


class XGBaseModel(BaseModel):
    """Common interface for every xG model wrapper (binary target)."""

    def train(self, X=None, y=None, drop_cols=None, target_col=None,
              path=None, scoring='neg_log_loss', cv=5, test_size=0.25):
        if X is None or y is None:
            if target_col is None:
                raise ValueError(
                    "Either pass X and y directly, or pass "
                    "drop_cols and target_col so train() can load the data itself."
                )
            X, y = ShotCleaningRunner(output_path=path).load_X_y(drop_cols=drop_cols or [], target_col=target_col)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.X_test, self.y_test = X_test, y_test

        pipeline = self.build_pipeline()
        param_grid = self.get_param_grid()
        self.search = GridSearchCV(
            pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1
        )
        self.search.fit(X_train, y_train)
        self.best_estimator_ = self.search.best_estimator_
        return self.best_estimator_

    def predict_proba(self, X):
        """Return predicted goal probabilities for X."""
        if self.best_estimator_ is None:
            raise RuntimeError("Call train() before predict_proba().")
        return self.best_estimator_.predict_proba(X)[:, 1]

    def get_evaluator(self):
        from src.models.evaluation import XGEvaluator
        return XGEvaluator()