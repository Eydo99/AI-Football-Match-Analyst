"""
Logistic Regression — interpretable baseline for the 5-class event classifier.

See Section 2 of the Event Classification Pipeline Guide.
"""

from src.models.event_classification_model.event_base_model import EventClassificationBaseModel

class LogisticRegressionEventModel(EventClassificationBaseModel):
    """See EventClassificationBaseModel for train()/predict_proba()/evaluate()."""

    def build_pipeline(self):
        # TODO: scaler + LogisticRegression(multi_class='multinomial',
        #       class_weight='balanced', max_iter=2000, random_state=42).
        # class_weight='balanced' matters here (unlike the xG version):
        # without it this will essentially never predict Shot/Foul.
        raise NotImplementedError

    def get_param_grid(self):
        # TODO: per the guide, n_iter=6 in RandomizedSearchCV effectively
        # covers this whole grid, so it's a full search in practice:
        #   'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
        #   'clf__penalty': ['l2']
        #   'clf__solver': ['lbfgs']
        raise NotImplementedError