import pandas as pd
import ipyvuetify as v

from antakia import config
from antakia.explanation.explanations import compute_explanations, ExplanationMethod
from antakia.gui.progress_bar import ProgressBar


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
        X: original train DataFrame
        y: target variable
        model: customer model
        on_change_callback: callback to notify explanation change
        X_exp: user provided explanations
        """
        self.widget = None
        self.X = X
        self.y = y
        self.model = model
        self.on_change_callback = on_change_callback
        self.initialized = False

        # init dict of explanations
        self.explanations: dict[str, pd.DataFrame | None] = {
            exp: None for exp in self.available_exp
        }

        if X_exp is not None:
            self.explanations[self.available_exp[0]] = X_exp

        # init selected explanation
        if X_exp is not None:
            self.current_exp = self.available_exp[0]
        else:
            self.current_exp = self.available_exp[1]

        self.build_widget()

    def build_widget(self):
        self.widget = v.Row(children=[
            v.Select(  # Select of explanation method
                label="Explanation method",
                items=[
                    {"text": "Imported", "disabled": True},
                    {"text": "SHAP", "disabled": True},
                    {"text": "LIME", "disabled": True},
                ],
                class_="ml-6 mr-6",
                style_="width: 15%",
                disabled=False,
            ),
            v.ProgressCircular(  # exp menu progress bar
                class_="ml-6 mr-6",
                indeterminate=False,
                color="grey",
                width="6",
                size="35",
            )
        ])
        # refresh select menu
        self.update_explanation_select()
        self.get_explanation_select().on_event("change", self.explanation_select_changed)
        # set up callback
        self.get_progress_bar().reset_progress_bar()

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
            self.compute_explanation(config.DEFAULT_EXPLANATION_METHOD, progress_callback)
        # ensure progress is at 100%
        progress_callback(100, 0)
        self.initialized = True

    @property
    def current_exp_df(self) -> pd.DataFrame:
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
        exp_values = []
        for exp in self.available_exp:
            if exp == 'Imported':
                exp_values.append({
                    "text": exp,
                    'disabled': self.explanations[exp] is None
                })
            else:
                exp_values.append({
                    "text": exp + (' (compute)' if self.explanations[exp] is None else ''),
                    'disabled': False
                })
        self.get_explanation_select().items = exp_values
        self.get_explanation_select().v_model = self.current_exp

    def get_progress_bar(self):
        progress_widget = self.widget.children[1]
        progress_bar = ProgressBar(progress_widget)
        return progress_bar

    def get_explanation_select(self):
        """
        returns the explanation select menu
        Returns
        -------

        """
        return self.widget.children[0]

    def compute_explanation(self, explanation_method: int, progress_bar: callable):
        """
        compute explanation and refresh widgets (select the new explanation method)
        Parameters
        ----------
        explanation_method: desired explanation
        progress_bar : progress bar to notify progress to

        Returns
        -------

        """
        self.current_exp = self.available_exp[explanation_method]
        # We compute proj for this new PV :
        x_exp = compute_explanations(self.X, self.model, explanation_method, progress_bar)
        pd.testing.assert_index_equal(x_exp.columns, self.X.columns)

        # update explanation
        self.explanations[self.current_exp] = x_exp
        # refresh front
        self.update_explanation_select()

    def disable_selection(self, is_disabled: bool):
        """
        disable widgets
        Parameters
        ----------
        is_disabled = should disable ?

        Returns
        -------

        """
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
        if not isinstance(data, str):
            raise KeyError('invalid explanation')
        data = data.replace(' ', '').replace('(compute)', '')
        self.current_exp = data

        if self.explanations[self.current_exp] is None:
            exp_method = ExplanationMethod.explain_method_as_int(self.current_exp)
            progress_bar = self.get_progress_bar()
            self.compute_explanation(exp_method, progress_bar)

        self.on_change_callback(self.current_exp_df)
