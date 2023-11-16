import lime
import numpy as np
import pandas as pd
import shap

from antakia.data import ExplanationMethod


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

        # TODO : It seems we defined class_name in order to work with California housing dataset. We should find a way to generalize this.
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(self.X), feature_names=self.X.columns,
                                                           class_names=['price'], verbose=False, mode='regression',
                                                           discretize_continuous=True)

        N = self.X.shape[0]
        values_lime = pd.DataFrame(np.zeros((N, self.X.shape[-1])))

        for j in range(N):
            l = []
            exp = explainer.explain_instance(
                self.X.values[j], self.model.predict
            )
            l = []
            size = self.X.shape[-1]
            for ii in range(size):
                exp_map = exp.as_map()[0]
                l.extend(exp_map[ii][1] for jj in range(size) if ii == exp_map[jj][0])

            values_lime.iloc[j] = pd.Series(l)
            progress += 100 / len(self.X)
            self.publish_progress(progress)
        j = list(self.X.columns)
        for i in range(len(j)):
            j[i] = j[i] + "_lime"
        values_lime.columns = j
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
