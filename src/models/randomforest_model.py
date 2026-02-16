import joblib
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, random_state=None):
        self.model = RandomForestClassifier(class_weight='balanced', max_depth=15, n_estimators=200, random_state=random_state)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def export(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)