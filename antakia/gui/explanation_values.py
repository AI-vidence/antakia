import pandas as pd

from antakia import config
from antakia.compute.explanation.explanations import compute_explanations, ExplanationMethod
from antakia.data_handler.projected_values import ProjectedValues
from antakia.gui.progress_bar import ProgressBar
from antakia.gui.widgets import get_widget, app_widget


class ExplanationValues:
    """
    Widget to manage explanation values
    in charge on computing them when necessary
    """
    available_exp = ['Imported', 'SHAP', 'LIME']

    def __init__(self, X: pd.DataFrame, y: pd.Series, model, on_change_callback: callable, X_exp=None):
        """

        Parameters
        ----------
        X: orignal train DataFrame
        y: target variable
        model: customer model
        on_change_callback: callback to notify explanation change
        X_exp: user provided explanations
        """
        self.X = X
        self.y = y
        self.model = model
        self.on_change_callback = on_change_callback
        self.initialized = False

        # init dict of explanations
        self.explanations: dict[str, ProjectedValues | None] = {
            exp: None for exp in self.available_exp
        }

        if X_exp is not None:
            self.explanations[self.available_exp[0]] = ProjectedValues(X_exp, y)

        # set up compute menu
        get_widget(app_widget, "13000203").on_event("click", self.compute_btn_clicked)
        get_widget(app_widget, "13000303").on_event("click", self.compute_btn_clicked)
        self.update_compute_menu()

        # init selected explanation
        if X_exp is not None:
            self.current_exp = self.available_exp[0]
        else:
            self.current_exp = self.available_exp[1]

        # refresh select menu
        self.update_explanation_select()
        # set up callback
        self.get_explanation_select().on_event("change", self.explanation_select_changed)

    def initialize(self, progress_callback):
        """
        initialize class (compute explanation if necessary)
        Parameters
        ----------
        progress_callback : callback to notify progress

        Returns
        -------

        """
        if not self.has_user_exp:
            # compute explanation if not provided
            self.compute_explanation(config.DEFAULT_EXPLANATION_METHOD, progress_callback, auto_update=False)
        # ensure progess is at 100%
        progress_callback(100, 0)
        self.initialized = True

    @property
    def current_pv(self) -> ProjectedValues:
        """
        currently selected explanation projected values instance
        Returns
        -------

        """
        return self.explanations[self.current_exp]

    @property
    def has_user_exp(self) -> bool:
        """
        has the user provided an explanation
        Returns
        -------

        """
        return self.explanations[self.available_exp[0]] is not None

    def update_explanation_select(self):
        """
        refresh explanation select menu
        Returns
        -------

        """
        self.get_explanation_select().items = [
            {"text": exp, "disabled": self.explanations[exp] is None} for exp in self.available_exp
        ]
        self.get_explanation_select().v_model = self.current_exp

    def get_compute_menu(self):
        """
        returns the compute menu widget
        Returns
        -------

        """
        return get_widget(app_widget, "13")

    def get_explanation_select(self):
        """
        returns the explanation select menu
        Returns
        -------

        """
        return get_widget(app_widget, "12")

    def compute_explanation(self, explanation_method: int, progress_bar: callable, auto_update: bool = True):
        """
        compute explanation and refresh widgets (select the new explanation method)
        Parameters
        ----------
        explanation_method: desired explanation
        progress_bar : progress bar to notify progress to
        auto_update : should trigger the on change callback

        Returns
        -------

        """
        self.current_exp = self.available_exp[explanation_method]
        # We compute proj for this new PV :
        X_exp = compute_explanations(self.X, self.model, explanation_method, progress_bar)
        # update explanation
        self.explanations[self.current_exp] = ProjectedValues(X_exp, self.y)
        # refresh front
        self.update_explanation_select()
        self.update_compute_menu()
        # call callback
        if auto_update:
            self.on_change_callback(self.current_pv, progress_bar)

    def update_compute_menu(self):
        """
        refresh compute menu
        Returns
        -------

        """
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
            progress_widget = get_widget(app_widget, "13000201")
        else:
            desired_explain_method = ExplanationMethod.LIME
            progress_widget = get_widget(app_widget, "13000301")

        progress_bar = ProgressBar(progress_widget)
        self.compute_explanation(desired_explain_method, progress_bar.update)

    def disable_selection(self, is_disabled: bool):
        """
        disable widgets
        Parameters
        ----------
        is_disabled = should disable ?

        Returns
        -------

        """
        self.get_compute_menu().disabled = is_disabled
        self.get_explanation_select().disabled = is_disabled

    def explanation_select_changed(self, widget, event, data):
        """
        triggered on selection of new explanation by user
        explanation has already been computed (the option is enabled in select)
        Parameters
        ----------
        widget
        event
        data: explanation name

        Returns
        -------

        Called when the user chooses another dataframe
        """
        self.current_exp = data

        self.on_change_callback(self.current_pv)
