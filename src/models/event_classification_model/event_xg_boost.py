from src.models.event_classification_model.event_base_model import EventClassificationBaseModel
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


class XGBoostEventModel(EventClassificationBaseModel):
    def build_pipeline(self):
        pipeline_xgb = Pipeline([
            ('clf',XGBClassifier(objective='multi:softprob' , num_class=5 , eval_metric='mlogloss' , tree_method='hist' , random_state=42))

        ])
        return pipeline_xgb

    def get_param_grid(self):
        param_xgb={
            'clf__n_estimators': [200, 300, 400],
            'clf__max_depth': [4, 5, 6, 8],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.7, 0.8, 1.0],
            'clf__colsample_bytree': [0.7, 0.8, 1.0],
            'clf__min_child_weight': [1, 5, 10]
        }
        return param_xgb

    def get_fit_params(self, X_train, y_train) -> dict:
         weights = compute_sample_weight('balanced',y_train)
         return {'clf__sample_weight':weights}
        