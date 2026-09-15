from src.features.base import Transformer

class ShotFilter(Transformer):
    def transform(self, df):
        return df[df['type'] == 'Shot'].copy()