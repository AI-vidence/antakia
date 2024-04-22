import time
from functools import partial
from typing import Callable

import pandas as pd
import ipyvuetify as v

from antakia.config import AppConfig
from antakia_core.explanation import compute_explanations, ExplanationMethod

from antakia.gui.helpers.data import DataStore
from antakia.gui.helpers.progress_bar import ProgressBar
from antakia.utils.logging_utils import Log
from antakia.utils.stats import stats_logger, log_errors


class ExplanationValues:
    """
    Widget to manage explanation values
    in charge on computing them when necessary
    """
    available_exp = ['Imported', 'SHAP', 'LIME']

    def __init__(
        self,
        data_store: DataStore,
        on_change_callback: Callable,
        disable_gui: Callable,
    ):
        """

        Parameters
        ----------
        data_store: data_store object with all data
        on_change_callback: callback to notify explanation change
        """
        self.widget = None
        self.data_store = data_store
        self.on_change_callback = partial(on_change_callback, self,
                                          'explanation_changed')
        self.disable_gui = partial(disable_gui, self)
        self.initialized = False

        # init dict of explanations
        self.explanations: dict[str, pd.DataFrame | None] = {
            exp: None
            for exp in self.available_exp
        }

        self.explanations[self.available_exp[0]] = self.data_store.X_exp

        # init selected explanation
        if self.data_store.X_exp is not None:
            self.current_exp = self.available_exp[0]
        else:
            self.current_exp = self.available_exp[1]
        stats_logger.log('exp_method_init', {'exp_method': self.current_exp})
        self._build_widget()

    def _build_widget(self):
        self.explanation_select = v.Select(  # Select of explanation method
            label="Explanation method",
            items=[
                {
                    "text": "Imported",
                    "disabled": True
                },
                {
                    "text": "SHAP",
                    "disabled": True
                },
                {
                    "text": "LIME",
                    "disabled": True
                },
            ],
            class_="ml-2 mr-2",
            style_="width: 15%",
            disabled=False,
        )
        self.progress_bar_wgt = v.ProgressCircular(  # exp menu progress bar
            class_="ml-2 mr-2 mt-2",
            indeterminate=False,
            color="grey",
            width="6",
            size="35",
        )
        self.progress_bar = ProgressBar(self.progress_bar_wgt)
        self.widget = v.Row(
            children=[self.explanation_select, self.progress_bar_wgt])
        # refresh select menu
        self.update_explanation_select()
        # set up callback
        self.explanation_select.on_event("change",
                                         self.explanation_select_changed)
        self.progress_bar.reset_progress_bar()

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
            self.compute_explanation(AppConfig.ATK_DEFAULT_EXPLANATION_METHOD,
                                     progress_callback)

        self.select_explanation(self.current_exp)
        # ensure progress is at 100%
        progress_callback(100, 0)
        self.initialized = True

    @property
    def current_exp_df(self) -> pd.DataFrame | None:
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
                    "text":
                    exp +
                    (' (compute)' if self.explanations[exp] is None else ''),
                    'disabled':
                    False
                })
        self.explanation_select.items = exp_values
        self.explanation_select.v_model = self.current_exp

    def compute_explanation(self, explanation_method: int,
                            progress_bar: ProgressBar):
        """
        compute explanation and refresh widgets (select the new explanation method)
        Parameters
        ----------
        explanation_method: desired explanation
        progress_bar : progress bar to notify progress to

        Returns
        -------

        """
        t = time.time()
        self.disable_gui('computing_explanations', True)
        # We compute proj for this new PV :
        x_exp = compute_explanations(self.data_store.X, self.data_store.model,
                                     explanation_method,
                                     self.data_store.problem_category,
                                     progress_bar)
        pd.testing.assert_index_equal(x_exp.columns, self.data_store.X.columns)

        # update explanation
        self.explanations[self.available_exp[explanation_method]] = x_exp
        # refresh front
        self.update_explanation_select()
        self.disable_gui('explanations_computed', False)
        stats_logger.log('compute_explanation', {
            'exp_method': explanation_method,
            'compute_time': time.time() - t
        })

    def disable_selection(self, is_disabled: bool):
        """
        disable widgets
        Parameters
        ----------
        is_disabled = should disable ?

        Returns
        -------

        """
        self.explanation_select.disabled = is_disabled

    @log_errors
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
        with Log('explanation_select_changed', 2):
            self.select_explanation(data)
            self.on_change_callback()

    def select_explanation(self, data):
        stats_logger.log('exp_method_changed', {'selected': data})
        if not isinstance(data, str):
            raise KeyError('invalid explanation')
        data = data.replace(' ', '').replace('(compute)', '')
        self.current_exp = data
        if self.explanations[self.current_exp] is None:
            exp_method = ExplanationMethod.explain_method_as_int(
                self.current_exp)
            self.compute_explanation(exp_method, self.progress_bar)
        self.data_store.X_exp = self.current_exp_df
