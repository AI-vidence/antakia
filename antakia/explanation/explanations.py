import lime
import numpy as np
import pandas as pd
import shap

from antakia.explanation.explanation_method import ExplanationMethod


# ===========================================================
#              Explanations implementations
# ===========================================================


class SHAPExplanation(ExplanationMethod):
    """
    SHAP computation class.
    """

    def __init__(self, X: pd.DataFrame, model, progress_updated: callable = None):
        super().__init__(ExplanationMethod.SHAP, X, model, progress_updated)

    def compute(self) -> pd.DataFrame:
        self.publish_progress(0)
        explainer = shap.TreeExplainer(self.model)
        chunck_size = 200
        shap_val_list = []
        for i in range(0, len(self.X), chunck_size):
            explanations = explainer.shap_values(self.X.iloc[i:i + chunck_size])
            shap_val_list.append(
                pd.DataFrame(explanations, columns=self.X.columns, index=self.X.index[i:i + chunck_size]))
            self.publish_progress(int(100 * i / len(self.X)))
        shap_values = pd.concat(shap_val_list)
        self.publish_progress(100)
        return shap_values


class LIMExplanation(ExplanationMethod):
    """
    LIME computation class.
    """

    def __init__(self, X: pd.DataFrame, model, progress_updated: callable = None):
        super().__init__(ExplanationMethod.LIME, X, model, progress_updated)

    def compute(self) -> pd.DataFrame:
        progress = 0

        self.publish_progress(progress)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X.sample(min(len(self.X), 500)).values,
            feature_names=self.X.columns,
            verbose=False,
            mode='regression',
            discretize_continuous=True
        )

        values_lime = pd.DataFrame(
            np.zeros(self.X.shape),
            index=self.X.index,
            columns=self.X.columns
        )
        for index, row in self.X.iterrows():
            exp = explainer.explain_instance(row.values, self.model.predict)
            values_lime.loc[index] = pd.Series(exp.local_exp[0], index=explainer.feature_names).str[1]
            progress += 100 / len(self.X)
            self.publish_progress(progress)
        self.publish_progress(100)
        return values_lime


def compute_explanations(X: pd.DataFrame, model, explanation_method: int, callback: callable) -> pd.DataFrame:
    """ Generic method to compute explanations, SHAP or LIME
    """
    if explanation_method == ExplanationMethod.SHAP:
        return SHAPExplanation(X, model, callback).compute()
    elif explanation_method == ExplanationMethod.LIME:
        return LIMExplanation(X, model, callback).compute()
    else:
        raise ValueError(f"This explanation method {explanation_method} is not valid!")
