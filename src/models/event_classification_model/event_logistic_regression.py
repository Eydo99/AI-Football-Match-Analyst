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
        # the cleaner fills speed by the true class, which is unknown at prediction.
        # zero plus the existing flag gives the same rule during fit and prediction.
        result.loc[result['ball_speed_missing'].eq(1), 'ball_speed'] = 0.0
    return result


class LogisticRegressionEventModel(EventClassificationBaseModel):
    """use the shared match-grouped trainer with scaled, balanced logistic regression."""

    def build_pipeline(self):
        """return an independent pipeline for numeric event features."""
        # lbfgs uses multinomial loss for five classes; multi_class was removed in 1.8.
        classifier = LogisticRegression(
            solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=42,
        )
        return Pipeline([
            ('safe_speed', FunctionTransformer(_neutralize_missing_speed,
                                               feature_names_out='one-to-one')),
            # fitting the scaler inside the pipeline keeps each cv fold separate.
            ('scaler', StandardScaler()),
            ('clf', classifier),
        ])

    def get_param_grid(self):
        """return the guide's six regularization candidates; use n_iter=6 in train."""
        grid = {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__solver': ['lbfgs'],
        }
        # sklearn 1.8+ spells l2 as l1_ratio=0; older supported releases use penalty.
        if LogisticRegression().get_params().get('penalty') == 'l2':
            grid['clf__penalty'] = ['l2']
        else:
            grid['clf__l1_ratio'] = [0.0]
        return grid

    def get_evaluator(self):
        """connect this model to the current event evaluator interface."""
        from src.models.evaluation.event_evaluator import EventEvaluator
        return EventEvaluator()

    def evaluate(self, plot=True):
        """use the current evaluator instead of the base's obsolete module path."""
        return self.get_evaluator().evaluate(self, self.__class__.__name__, plot=plot)
