class NotFittedError(Exception):
    pass


class MLModel:
    def __init__(self, model, name, fitted=False):
        self.fitted = fitted
        self.model = model
        self.name = name

    def fit(self, *args, **kwargs):
        if not self.fitted:
            res = self.model.fit(*args, **kwargs)
            self.fitted = True
            return res

    def predict(self, *args, **kwargs):
        if self.fitted:
            return self.model.predict(*args, **kwargs)
        raise NotFittedError()
