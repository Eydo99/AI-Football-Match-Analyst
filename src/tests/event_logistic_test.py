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
    df['ball_speed_missing'] = 0
    df.loc[::4, 'ball_speed_missing'] = 1
    df['ball_speed'] = df['f0'].abs()
    df.loc[df['ball_speed_missing'].eq(1), 'ball_speed'] = 999  # would leak if not neutralized
    rng = np.random.RandomState(random_state)
    groups = pd.Series(rng.randint(0, n_groups, size=n_samples), name='match_id')
    return df, pd.Series(y), groups


class TestNeutralizeMissingSpeed:
    def test_rejects_non_dataframe(self):
        from src.models.event_classification_model.event_logistic_regression import _neutralize_missing_speed
        with pytest.raises(TypeError):
            _neutralize_missing_speed(np.array([[1, 2], [3, 4]]))

    def test_rejects_grouping_or_target_columns(self):
        from src.models.event_classification_model.event_logistic_regression import _neutralize_missing_speed
        df = pd.DataFrame({'match_id': [1, 2], 'f0': [0.1, 0.2]})
        with pytest.raises(ValueError, match='grouping keys'):
            _neutralize_missing_speed(df)

    def test_requires_missing_flag_alongside_ball_speed(self):
        from src.models.event_classification_model.event_logistic_regression import _neutralize_missing_speed
        df = pd.DataFrame({'ball_speed': [1.0, 2.0]})
        with pytest.raises(ValueError, match='ball_speed_missing'):
            _neutralize_missing_speed(df)

    def test_zeroes_speed_only_where_flag_is_set(self):
        from src.models.event_classification_model.event_logistic_regression import _neutralize_missing_speed
        df = pd.DataFrame({
            'ball_speed': [10.0, 20.0, 30.0],
            'ball_speed_missing': [0, 1, 0],
            'other': [1, 2, 3],
        })
        result = _neutralize_missing_speed(df)
        assert result['ball_speed'].tolist() == [10.0, 0.0, 30.0]
        assert result['other'].tolist() == [1, 2, 3]
        assert df['ball_speed'].tolist() == [10.0, 20.0, 30.0]

    def test_passthrough_when_ball_speed_absent(self):
        from src.models.event_classification_model.event_logistic_regression import _neutralize_missing_speed
        df = pd.DataFrame({'f0': [1.0, 2.0]})
        result = _neutralize_missing_speed(df)
        pd.testing.assert_frame_equal(result, df)


class TestLogisticRegressionEventModel:
    def test_pipeline_contract(self):
        from sklearn.pipeline import Pipeline as ModelPipeline
        from sklearn.linear_model import LogisticRegression
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        pipeline = LogisticRegressionEventModel().build_pipeline()
        assert isinstance(pipeline, ModelPipeline)
        assert list(pipeline.named_steps) == ['safe_speed', 'scaler', 'clf']
        clf = pipeline.named_steps['clf']
        assert isinstance(clf, LogisticRegression)
        assert clf.solver == 'lbfgs'
        assert clf.class_weight == 'balanced'
        assert clf.max_iter == 2000
        assert clf.random_state == 42

    def test_param_grid_has_six_C_candidates_and_one_regularization_key(self):
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        grid = LogisticRegressionEventModel().get_param_grid()
        assert grid['clf__C'] == [0.001, 0.01, 0.1, 1, 10, 100]
        assert grid['clf__solver'] == ['lbfgs']
        assert ('clf__penalty' in grid) != ('clf__l1_ratio' in grid)

    def test_fresh_unfitted_pipeline_and_independent_grid(self):
        from sklearn.exceptions import NotFittedError
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        model = LogisticRegressionEventModel()
        first, second = model.build_pipeline(), model.build_pipeline()
        first.set_params(clf__C=99)
        assert second.named_steps['clf'].C != 99
        with pytest.raises(NotFittedError):
            second.predict(sample_event_data()[0])
        grid = model.get_param_grid()
        grid['clf__C'].append(999)
        assert 999 not in model.get_param_grid()['clf__C']

    def test_get_evaluator_returns_event_evaluator(self):
        from src.models.evaluation import EventEvaluator
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        assert isinstance(LogisticRegressionEventModel().get_evaluator(), EventEvaluator)

    def test_shared_train_and_prediction_interface(self, monkeypatch):
        from src.models.event_classification_model.event_logistic_regression import LogisticRegressionEventModel
        model = LogisticRegressionEventModel()
        X, y, groups = sample_event_data()
        with pytest.raises(RuntimeError, match='train'):
            model.predict_proba(X)
        monkeypatch.setattr(model, 'get_param_grid', lambda: {'clf__C': [0.1, 1]})
        best = model.train(X=X, y=y, groups=groups, n_iter=2, n_splits=2)
        assert best is model.best_estimator_
        assert model.search.scoring == 'f1_macro'
        proba = model.predict_proba(model.X_test)
        assert proba.shape == (len(model.X_test), 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
        assert set(model.groups_test).isdisjoint(set(groups) - set(model.groups_test))