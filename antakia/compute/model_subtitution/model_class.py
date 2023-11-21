import pandas as pd


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

    def predict(self, X, *args, **kwargs):
        if self.fitted:
            pred = self.model.predict(X, *args, **kwargs)
            if isinstance(pred, (pd.DataFrame, pd.Series)):
                return pred
            else:
                return pd.Series(pred, index=X.index)
        raise NotFittedError()
