from src.models.event_classification_model.event_base_model import EventClassificationBaseModel

class RandomForestEventModel(EventClassificationBaseModel):

    def build_pipeline(self):
        # TODO: RandomForestClassifier(class_weight='balanced_subsample',
        #       n_jobs=-1, random_state=42). No scaler needed.
        # balanced_subsample (not plain 'balanced') tends to work better
        # here since Shot is only ~1.1% of rows.
        raise NotImplementedError

    def get_param_grid(self):
        # TODO: per the guide (max_depth=None dropped deliberately — too
        # expensive/overfits Carry/Pass at 2.27M rows):
        #   'clf__n_estimators': [200, 300, 400]
        #   'clf__max_depth': [8, 12, 16, 20]
        #   'clf__min_samples_leaf': [5, 10, 25]
        #   'clf__max_features': ['sqrt', 'log2']
        raise NotImplementedError

    # NOTE: if search.fit() is too slow on the full ~1.8M-row training
    # split, the guide suggests fitting on a match-grouped ~300k-row
    # subsample first to shortlist 2-3 configs, then refitting the winner
    # on the full set. That would be a good place for a small override of
    # train() here later, if needed — not required for the skeleton.