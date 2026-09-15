import numpy as np
import pandas as pd
import pytest


def sample_event_data(n_samples=300, n_groups=30, random_state=42):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples, n_features=8, n_informative=6,
        n_classes=5, n_clusters_per_class=1, random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(8)])
    rng = np.random.RandomState(random_state)
    groups = pd.Series(rng.randint(0, n_groups, size=n_samples), name='match_id')
    return df, pd.Series(y), groups


class TestXGBoostEventModel:
    def test_is_instantiable(self):
        # regression test: same abstract-method bug as RandomForestEventModel
        from src.models.event_classification_model.event_xg_boost import XGBoostEventModel
        XGBoostEventModel()

    def test_pipeline_contract(self):
        from sklearn.pipeline import Pipeline as ModelPipeline
        from xgboost import XGBClassifier
        from src.models.event_classification_model.event_xg_boost import XGBoostEventModel
        pipeline = XGBoostEventModel().build_pipeline()
        assert isinstance(pipeline, ModelPipeline)
        assert list(pipeline.named_steps) == ['clf']
        clf = pipeline.named_steps['clf']
        assert isinstance(clf, XGBClassifier)
        assert clf.objective == 'multi:softprob'
        assert clf.eval_metric == 'mlogloss'
        assert clf.tree_method == 'hist'
        assert clf.random_state == 42

    def test_param_grid_matches_pipeline_guide(self):
        from src.models.event_classification_model.event_xg_boost import XGBoostEventModel
        grid = XGBoostEventModel().get_param_grid()
        assert grid['clf__n_estimators'] == [200, 300, 400]
        assert grid['clf__max_depth'] == [4, 5, 6, 8]
        assert grid['clf__learning_rate'] == [0.01, 0.05, 0.1]
        assert grid['clf__subsample'] == [0.7, 0.8, 1.0]
        assert grid['clf__colsample_bytree'] == [0.7, 0.8, 1.0]
        assert grid['clf__min_child_weight'] == [1, 5, 10]
        assert all(name.startswith('clf__') for name in grid)

    def test_fresh_unfitted_pipeline_and_independent_grid(self):
        from sklearn.exceptions import NotFittedError
        from src.models.event_classification_model.event_xg_boost import XGBoostEventModel
        model = XGBoostEventModel()
        first, second = model.build_pipeline(), model.build_pipeline()
        first.set_params(clf__max_depth=99)
        assert second.named_steps['clf'].max_depth != 99
        with pytest.raises(NotFittedError):
            second.predict_proba(sample_event_data()[0])
        grid = model.get_param_grid()
        grid['clf__max_depth'].append(1)
        assert 1 not in model.get_param_grid()['clf__max_depth']

    def test_get_fit_params_returns_balanced_sample_weight(self):
        from sklearn.utils.class_weight import compute_sample_weight
        from src.models.event_classification_model.event_xg_boost import XGBoostEventModel
        X, y, _ = sample_event_data()
        fit_params = XGBoostEventModel().get_fit_params(X, y)
        assert set(fit_params) == {'clf__sample_weight'}
        expected = compute_sample_weight('balanced', y)
        np.testing.assert_allclose(fit_params['clf__sample_weight'], expected)

    def test_get_evaluator_returns_event_evaluator(self):
        from src.models.evaluation import EventEvaluator
        from src.models.event_classification_model.event_xg_boost import XGBoostEventModel
        assert isinstance(XGBoostEventModel().get_evaluator(), EventEvaluator)

    def test_shared_train_and_prediction_interface(self, monkeypatch):
        from src.models.event_classification_model.event_xg_boost import XGBoostEventModel
        model = XGBoostEventModel()
        X, y, groups = sample_event_data()
        with pytest.raises(RuntimeError, match='train'):
            model.predict_proba(X)
        monkeypatch.setattr(model, 'get_param_grid', lambda: {
            'clf__n_estimators': [20, 40], 'clf__max_depth': [2],
        })
        best = model.train(X=X, y=y, groups=groups, n_iter=2, n_splits=2)
        assert best is model.best_estimator_
        assert model.search.scoring == 'f1_macro'
        proba = model.predict_proba(model.X_test)
        assert proba.shape == (len(model.X_test), 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)