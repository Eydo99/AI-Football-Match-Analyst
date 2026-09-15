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


class TestRandomForestEventModel:
    def test_is_instantiable(self):
        # regression test: EventClassificationBaseModel must implement
        # get_evaluator(), not override evaluate() directly, or this
        # subclass stays abstract and raises TypeError.
        from src.models.event_classification_model.event_random_forest import RandomForestEventModel
        RandomForestEventModel()

    def test_pipeline_contract(self):
        from sklearn.pipeline import Pipeline as ModelPipeline
        from sklearn.ensemble import RandomForestClassifier
        from src.models.event_classification_model.event_random_forest import RandomForestEventModel
        pipeline = RandomForestEventModel().build_pipeline()
        assert isinstance(pipeline, ModelPipeline)
        assert list(pipeline.named_steps) == ['clf']
        clf = pipeline.named_steps['clf']
        assert isinstance(clf, RandomForestClassifier)
        assert clf.class_weight == 'balanced_subsample'
        assert clf.n_jobs == -1
        assert clf.random_state == 42

    def test_param_grid_matches_pipeline_guide(self):
        from src.models.event_classification_model.event_random_forest import RandomForestEventModel
        grid = RandomForestEventModel().get_param_grid()
        assert grid['clf__n_estimators'] == [200, 300, 400]
        assert grid['clf__max_depth'] == [8, 12, 16, 20]
        assert grid['clf__min_samples_leaf'] == [5, 10, 25]
        assert grid['clf__max_features'] == ['sqrt', 'log2']
        assert None not in grid['clf__max_depth']  # guide deliberately drops unbounded depth

    def test_fresh_unfitted_pipeline_and_independent_grid(self):
        from sklearn.exceptions import NotFittedError
        from src.models.event_classification_model.event_random_forest import RandomForestEventModel
        model = RandomForestEventModel()
        first, second = model.build_pipeline(), model.build_pipeline()
        first.set_params(clf__n_estimators=999)
        assert second.named_steps['clf'].n_estimators != 999
        with pytest.raises(NotFittedError):
            second.predict(sample_event_data()[0])
        grid = model.get_param_grid()
        grid['clf__n_estimators'].append(1)
        assert 1 not in model.get_param_grid()['clf__n_estimators']

    def test_default_fit_params_are_empty(self):
        # RF corrects imbalance via class_weight, not sample_weight
        from src.models.event_classification_model.event_random_forest import RandomForestEventModel
        X, y, _ = sample_event_data()
        assert RandomForestEventModel().get_fit_params(X, y) == {}

    def test_get_evaluator_returns_event_evaluator(self):
        from src.models.evaluation import EventEvaluator
        from src.models.event_classification_model.event_random_forest import RandomForestEventModel
        assert isinstance(RandomForestEventModel().get_evaluator(), EventEvaluator)

    def test_shared_train_and_prediction_interface(self, monkeypatch):
        from src.models.event_classification_model.event_random_forest import RandomForestEventModel
        model = RandomForestEventModel()
        X, y, groups = sample_event_data()
        with pytest.raises(RuntimeError, match='train'):
            model.predict_proba(X)
        monkeypatch.setattr(model, 'get_param_grid', lambda: {'clf__n_estimators': [20, 40]})
        best = model.train(X=X, y=y, groups=groups, n_iter=2, n_splits=2)
        assert best is model.best_estimator_
        assert model.search.scoring == 'f1_macro'
        proba = model.predict_proba(model.X_test)
        assert proba.shape == (len(model.X_test), 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)