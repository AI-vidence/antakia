import pandas as pd

from antakia import config
from antakia.compute.explanation.explanations import compute_explanations, ExplanationMethod
from antakia.data_handler.projected_values import ProjectedValues
from antakia.gui.widgets import get_widget, app_widget


class ExplanationValues:
    """

    """
    available_exp = ['Imported', 'SHAP', 'LIME']

    def __init__(self, X: pd.DataFrame, y: pd.Series, model, on_change_callback: callable, X_exp: pd.DataFrame = None):
        self.X = X
        self.y = y
        self.model = model

        self.explanations: dict[str, ProjectedValues | None] = {
            exp: None for exp in self.available_exp
        }

        if X_exp is not None:
            self.explanations[self.available_exp[0]] = ProjectedValues(X_exp, y)

        self.update_explanation_select()
        self.get_explanation_select().on_event("change", self.explanation_select_changed)

        get_widget(app_widget, "13000203").on_event("click", self.compute_btn_clicked)
        get_widget(app_widget, "13000303").on_event("click", self.compute_btn_clicked)
        self.update_compute_menu()
        if X_exp is not None:
            self.current_exp = self.available_exp[0]
        else:
            self.current_exp = self.available_exp[1]

        self.on_change_callback = on_change_callback

    def initialize(self, progress_callback):
        if not self.has_user_exp:
            self.compute_explanation(config.DEFAULT_EXPLANATION_METHOD, progress_callback)
            self.on_change_callback(progress_callback)
        self.get_explanation_select().v_model = self.current_exp

    @property
    def current_pv(self) -> ProjectedValues:
        """
        Returns the ProjectedValue
        """
        return self.explanations[self.current_exp]

    @property
    def has_user_exp(self) -> bool:
        return self.explanations[self.available_exp[0]] is not None

    def update_explanation_select(self):
        """
       Called at startup by the GUI (only ES HE)
       """
        self.get_explanation_select().items = [
            {"text": exp, "disabled": self.explanations[exp] is None} for exp in self.available_exp
        ]

    def get_compute_menu(self):
        """
       Called at startup by the GUI (only ES HDE)
       """
        return get_widget(app_widget, "13")

    def get_explanation_select(self):
        """
       Called at startup by the GUI (only ES HE)
       """
        return get_widget(app_widget, "12")

    def compute_explanation(self, explanation_method: int, progress_bar: callable):
        self.current_exp = self.available_exp[explanation_method]
        X_exp = compute_explanations(self.X, self.model, explanation_method, progress_bar)
        self.explanations[self.current_exp] = ProjectedValues(X_exp, self.y)
        # We compute proj for this new PV :
        self.update_explanation_select()
        self.update_compute_menu()

    def update_compute_menu(self):
        is_shap_computed = self.explanations[self.available_exp[1]] is not None
        get_widget(app_widget, "130000").disabled = is_shap_computed
        get_widget(app_widget, "13000203").disabled = is_shap_computed

        is_lime_computed = self.explanations[self.available_exp[2]] is not None
        get_widget(app_widget, "130001").disabled = is_lime_computed
        get_widget(app_widget, "13000303").disabled = is_lime_computed

    def compute_btn_clicked(self, widget, event, data):
        """
        Called when new explanation computed values are wanted
        """
        # This compute btn is no longer useful / clickable
        widget.disabled = True

        if widget == get_widget(app_widget, "13000203"):
            desired_explain_method = ExplanationMethod.SHAP
        else:
            desired_explain_method = ExplanationMethod.LIME
        self.compute_explanation(desired_explain_method, self.update_progress_linear)

    def disable_selection(self, is_disabled: bool):
        """

        """

        self.get_compute_menu().disabled = is_disabled
        self.get_explanation_select().disabled = is_disabled

    def update_progress_linear(self, method: ExplanationMethod, progress: int, duration: float = None):
        """
        Called by the computation process (SHAP or LUME) to udpate the progress linear
        """

        if method.explanation_method == ExplanationMethod.SHAP:
            progress_linear = get_widget(app_widget, "13000201")
            progress_linear.indeterminate = True
        else:
            progress_linear = get_widget(app_widget, "13000301")

        progress_linear.v_model = progress

        if progress == 100:
            if method.explanation_method == ExplanationMethod.SHAP:
                tab = get_widget(app_widget, "130000")
                progress_linear.indeterminate = False
            else:
                tab = get_widget(app_widget, "130001")
                progress_linear.v_model = progress
            tab.disabled = True

    def explanation_select_changed(self, widget, event, data):
        """
        Called when the user chooses another dataframe
        """
        # Remember : impossible items ine thee Select are disabled = we have the desired values
        self.current_exp = data

        self.on_change_callback()
