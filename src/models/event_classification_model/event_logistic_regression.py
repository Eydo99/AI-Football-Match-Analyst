"""balanced logistic regression for the five event classes."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.models.event_classification_model.event_base_model import EventClassificationBaseModel


def _neutralize_missing_speed(X):
    """remove label-dependent speed fills while keeping the missingness flag."""
    if not hasattr(X, 'columns'):
        raise TypeError('event features must be a dataframe with named columns')
    if {'match_id', 'event_class', 'type', 'ends_in_goal'}.intersection(X.columns):
        raise ValueError('drop grouping keys and target columns before fitting or predicting')
    result = X.copy()
    if 'ball_speed' in result:
        if 'ball_speed_missing' not in result:
            raise ValueError('ball_speed requires the ball_speed_missing indicator')
        result.loc[result['ball_speed_missing'].eq(1), 'ball_speed'] = 0.0
    return result


class LogisticRegressionEventModel(EventClassificationBaseModel):
    """use the shared match-grouped trainer with scaled, balanced logistic regression."""

    def build_pipeline(self):
        classifier = LogisticRegression(
            solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=42,
        )
        return Pipeline([
            ('safe_speed', FunctionTransformer(_neutralize_missing_speed,
                                               feature_names_out='one-to-one')),
            ('scaler', StandardScaler()),
            ('clf', classifier),
        ])

    def get_param_grid(self):
        grid = {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__solver': ['lbfgs'],
        }
        if LogisticRegression().get_params().get('penalty') == 'l2':
            grid['clf__penalty'] = ['l2']
        else:
            grid['clf__l1_ratio'] = [0.0]
        return grid