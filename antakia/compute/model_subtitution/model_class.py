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

class LinearMLModel(MLModel):

    def fit(self, X, *args, **kwargs):
        super().fit(X, *args, **kwargs)
        self.means = X.mean()

    def global_explanation(self):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        coefs['intercept'] = self.model.intercept_
        return {
            'type': 'table',
            'value': coefs
        }

    def local_explanation(self, x):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        exp = coefs * x - coefs * self.means
        return {
            'type': 'table',
            'prior': self.predict(x),
            'value': exp
        }


class GAMMLMdel(MLModel):
    def global_explanation(self):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        coefs['intercept'] = self.model.intercept_
        return {
            'type': 'table',
            'value': coefs
        }

    def local_explanation(self, x):
        coefs = pd.Series(self.model.coef_, index=self.model.features_names_in_)
        exp = coefs * x - coefs * self.means
        return {
            'type': 'table',
            'prior': self.predict(x),
            'value': exp
        }