import pandas as pd

from antakia_core.utils.long_task import LongTask


class ExplanationMethod(LongTask):
    """
    Abstract class (see Long Task) to compute explaination values for the Explanation Space (ES)

    Attributes
    model : the model to explain
    explanation_method : SHAP or LIME
    """

    # Class attributes
    NONE = 0  # no explanation, ie: original values
    SHAP = 1
    LIME = 2

    def __init__(
            self,
            explanation_method: int,
            X: pd.DataFrame,
            model,
            task_type,
            progress_updated: callable = None,
    ):
        if not ExplanationMethod.is_valid_explanation_method(explanation_method):
            raise ValueError(explanation_method, " is a bad explanation method")
        self.explanation_method = explanation_method
        super().__init__(X, progress_updated)
        self.task_type = task_type
        self.model = model

    @staticmethod
    def is_valid_explanation_method(method: int) -> bool:
        """
        Returns True if this is a valid explanation method.
        """
        return (
                method == ExplanationMethod.SHAP
                or method == ExplanationMethod.LIME
                or method == ExplanationMethod.NONE
        )

    @staticmethod
    def explanation_methods_as_list() -> list:
        return [ExplanationMethod.SHAP, ExplanationMethod.LIME]

    @staticmethod
    def explain_method_as_str(method: int) -> str:
        if method == ExplanationMethod.SHAP:
            return "SHAP"
        elif method == ExplanationMethod.LIME:
            return "LIME"
        else:
            raise ValueError(method, " is a bad explanation method")

    @staticmethod
    def explain_method_as_int(method: str) -> int:
        if method.upper() == "SHAP":
            return ExplanationMethod.SHAP
        elif method.upper() == "LIME":
            return ExplanationMethod.LIME
        else:
            raise ValueError(method, " is a bad explanation method")
