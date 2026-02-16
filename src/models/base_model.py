# create base model template

class BaseModel:
    
    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def export(self, path):
        pass