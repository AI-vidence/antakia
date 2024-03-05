from __future__ import annotations

import time

import numpy as np
import pandas as pd

import ipyvuetify as v
import IPython.display

from antakia_core.data_handler.region import ModelRegionSet, ModelRegion

from antakia.gui.antakia_logo import TopBar
from antakia.gui.explanation_values import ExplanationValues
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank
from antakia.gui.progress_bar import ProgressBar, MultiStepProgressBar
from antakia.explanation.explanation_method import ExplanationMethod
from antakia_ac.auto_cluster import AutoCluster
from antakia_core.compute.skope_rule.skope_rule import skope_rules
import antakia.config as config
from antakia_core.data_handler.rules import RuleSet

from antakia.gui.tabs.model_explorer import ModelExplorer
from antakia.gui.tabs.tab1 import Tab1
from antakia.gui.tabs.tab2 import Tab2
from antakia.gui.tabs.tab3 import Tab3
from antakia.gui.widget_utils import get_widget, change_widget
from antakia.gui.widgets import splash_widget, app_widget
from antakia.gui.high_dim_exp.highdimexplorer import HighDimExplorer
from antakia.gui.ruleswidget import RulesWidget

from antakia.gui.metadata import metadata

import logging
from antakia.utils.logging_utils import conf_logger
from antakia_core.utils.utils import boolean_mask, ProblemCategory, format_data
from antakia_core.utils.variable import DataVariables

from antakia.utils.stats import stats_logger, log_errors

logger = logging.getLogger(__name__)
conf_logger(logger)


class GUI:
    """
    GUI class.

    The GUI guides the user through the AntakIA process.
    It stores Xs, Y and the model to explain.
    It displays a UI (app_graph) and creates various UI objects, in particular
    two HighDimExplorers resposnible to compute or project values in 2 spaces.

    The interface is built using ipyvuetify and plotly.
    It heavily relies on the IPyWidgets framework.

    Instance Attributes
    ---------------------
    X : Pandas DataFrame, the orignal dataset
    y : Pandas Series, target values
    model : a model
    X_exp : a Pandas DataFrame, containing imported explanations
    variables : a list of Variable
    selection_ids : a list of a pd.DataFrame indexes, corresponding to the current selected points
        IMPORTANT : a dataframe index may differ from the row number
    vs_hde, es_hde : HighDimExplorer for the VS and ES space
    vs_rules_wgt, es_rules_wgt : RulesWidget
    region_set : a list of Region,
        a region is a dict : {'num':int, 'rules': list of rules, 'indexes', 'model': str, 'score': str}
        if the list of rules is None, the region has been defined with auto-cluster
        num start at 1
    validated_rules_region, validated_region, validated_sub_model

    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        variables: DataVariables,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_exp: pd.DataFrame | None = None,
        score: callable | str = "mse",
        problem_category: ProblemCategory = ProblemCategory.regression
    ):
        metadata.start()
        self.tab = 1
        self.X = X
        self.X_test = X_test
        self.y = y
        self.y_test = y_test
        self._y_pred = None
        self.problem_category = problem_category
        self.model = model
        self.variables: DataVariables = variables
        self.score = score
        # Init value space widgets
        self.selection_mask = boolean_mask(X, True)

        self.pv_bank = ProjectedValueBank(y)
        self.region_set = ModelRegionSet(self.X, self.y, self.X_test, self.y_test, self.model, self.score)

        # star dialog
        self.topbar = TopBar()

        # first hde
        self.vs_hde = HighDimExplorer(
            self.pv_bank,
            self.selection_changed,
            'VS'
        )

        # init Explanation space
        # first explanation getter/compute
        self.exp_values = ExplanationValues(
            self.X,
            self.y,
            self.model,
            problem_category,
            self.explanation_changed_callback,
            self.disable_hde,
            X_exp
        )
        # then hde
        self.es_hde = HighDimExplorer(
            self.pv_bank,
            self.selection_changed,
            'ES'
        )

        # init tabs
        self.tab1 = Tab1(variables, self.new_rule_selected_callback, self.validate_rules_callback, self.X, X_exp,
                         self.y)

        self.tab2 = Tab2(variables, X, self.vs_hde.projected_value_selector, self.es_hde.projected_value_selector,
                         self.region_set, self.update_region_callback, self.substitute_model_callback)

        self.tab3 = Tab3(X, problem_category, self.model_validation_callback)
        self.model_explorer = ModelExplorer(self.X)

        self.widget = app_widget.get_app_widget()
        self.splash_widget = splash_widget.get_app_widget()

    @log_errors
    def show_splash_screen(self):
        """Displays the splash screen and updates it during the first computations."""

        # We add both widgets to the current notebook cell and hide them
        t = time.time()
        IPython.display.display(self.splash_widget, self.widget)
        self.widget.hide()
        self.splash_widget.show()

        exp_progress_bar = ProgressBar(
            get_widget(self.splash_widget, "110"),
            unactive_color="light blue",
            reset_at_end=False
        )
        dimreduc_progress_bar = MultiStepProgressBar(
            get_widget(self.splash_widget, "210"),
            steps=2,
            unactive_color="light blue",
            reset_at_end=False
        )
        # We trigger VS proj computation :
        get_widget(
            self.splash_widget, "220"
        ).v_model = f"{config.ATK_DEFAULT_PROJECTION} on {self.X.shape} 1/2"

        self.vs_hde.initialize(progress_callback=dimreduc_progress_bar.get_update(1), X=self.X)

        # We trigger ES explain computation if needed :
        if not self.exp_values.has_user_exp:  # No imported explanation values
            exp_method = ExplanationMethod.explain_method_as_str(config.ATK_DEFAULT_EXPLANATION_METHOD)
            msg = f"Computing {exp_method} on {self.X.shape}"
        else:
            msg = f"Imported explained values {self.X.shape}"
        self.exp_values.initialize(exp_progress_bar.update)
        get_widget(self.splash_widget, "120").v_model = msg

        # THen we trigger ES proj computation :
        get_widget(
            self.splash_widget, "220"
        ).v_model = f"{config.ATK_DEFAULT_PROJECTION} on {self.X.shape} 2/2"
        self.es_hde.initialize(
            progress_callback=dimreduc_progress_bar.get_update(2),
            X=self.exp_values.current_exp_df
        )
        self.tab1.update_X_exp(self.exp_values.current_exp_df)
        self.selection_changed(None, boolean_mask(self.X, True))

        self.init_app()

        self.splash_widget.hide()
        self.widget.show()
        self.vs_hde.figure.create_figure()
        self.es_hde.figure.create_figure()
        self.select_tab(0)
        self.disable_hde()

        if metadata.counter == 10:
            self.topbar.open()
        stats_logger.log('gui_init_end', {'load_time': time.time() - t})

    def init_app(self):
        """
        Inits and wires the app_widget, and implements UI logic
        """

        # -------------- Dimension Switch --------------

        change_widget(self.widget, '0', self.topbar.widget)

        # -------------- Dimension Switch --------------

        get_widget(self.widget, "100").v_model = config.ATK_DEFAULT_DIMENSION == 3
        get_widget(self.widget, "100").on_event("change", self.switch_dimension)

        # -------------- ColorChoiceBtnToggle ------------

        # Set "change" event on the Button Toggle used to chose color
        get_widget(self.widget, "11").on_event("change", self.change_color)

        # -------------- ExplanationSelect ------------

        get_widget(self.widget, '12').children = [self.exp_values.widget]

        # -------------- set up VS High Dim Explorer  ------------

        get_widget(self.widget, '13').children = [self.vs_hde.projected_value_selector.widget]
        change_widget(self.widget, "201", self.vs_hde.figure.widget),

        # -------------- set up ES High Dim Explorer ------------

        get_widget(self.widget, '14').children = [self.es_hde.projected_value_selector.widget]
        change_widget(self.widget, "211", self.es_hde.figure.widget),

        # ================ Tab 1 Selection ================

        # We wire the click event on 'Tab 1'
        get_widget(self.widget, "40").on_event("click", self.select_tab_front(1))

        # We add the tab1 widget to the GUI :
        get_widget(self.widget, "43").children = self.tab1.widget

        # ================ Tab 2 : regions ===============
        # We wire the click event on 'Tab 2'
        get_widget(self.widget, "41").on_event("click", self.select_tab_front(2))
        # We add the Tab2 widget to the GUI :
        get_widget(self.widget, "44").children = self.tab2.widget

        # ============== Tab 3 : substitution ==================

        # We wire the click event on 'Tab 3'
        get_widget(self.widget, "42").on_event("click", self.select_tab_front(3))
        get_widget(self.widget, "45").children = self.tab3.widget

    # ==================== properties ==================== #

    @property
    def y_pred(self):
        if self._y_pred is None:
            pred = self.model.predict(self.X)
            if self.problem_category in [ProblemCategory.classification_with_proba]:
                pred = self.model.predict_proba(self.X)

            if len(pred.shape) > 1:
                if pred.shape[1] == 1:
                    pred = pred.squeeze()
                if pred.shape[1] == 2:
                    pred = np.array(pred)[:, 1]
                else:
                    pred = pred.argmax(axis=1)
            self._y_pred = pd.Series(pred, index=self.X.index)
        return self._y_pred

    # ==================== sync callbacks ==================== #

    @log_errors
    def explanation_changed_callback(self, current_exp_df: pd.DataFrame, progress_callback: callable = None):
        """
        on explanation change, synchronizes es_hde and tab1
        Parameters
        ----------
        current_exp_df
        progress_callback

        Returns
        -------

        """
        self.es_hde.update_X(current_exp_df, progress_callback)
        self.tab1.update_X_exp(current_exp_df)

    @log_errors
    def disable_hde(self, disable='auto'):
        if disable == 'auto':
            disable_proj = bool((self.tab == 0) and self.selection_mask.any() and not self.selection_mask.all())
            disable_figure = bool(self.tab > 1)
        else:
            disable_proj = disable
            disable_figure = disable
        self.vs_hde.disable(disable_figure, disable_proj)
        self.exp_values.disable_selection(disable_proj)
        self.es_hde.disable(disable_figure, disable_proj)

    @log_errors
    def selection_changed(self, caller: HighDimExplorer | None, new_selection_mask: pd.Series):
        """
        callback to synchronize both hdes and tab1
        Parameters
        ----------
        caller
        new_selection_mask

        Returns
        -------

        """
        """Called when the selection of one HighDimExplorer changes"""

        self.selection_mask = new_selection_mask

        # If new selection (empty or not) : if exists, we remove any 'pending rule'
        self.disable_hde()
        if new_selection_mask.all():
            # Selection is empty
            self.select_tab(0)
        else:
            stats_logger.log('selection_gui',
                             {'exp_method': self.exp_values.current_exp,
                              'vs_proj': str(self.vs_hde.projected_value_selector.current_proj),
                              'es_proj': str(self.es_hde.projected_value_selector.current_proj)})

        # We synchronize selection between the two HighDimExplorers
        if caller is None:
            self.es_hde.set_selection(self.selection_mask)
            self.vs_hde.set_selection(self.selection_mask)
        else:
            other_hde = self.es_hde if caller == self.vs_hde.figure else self.vs_hde
            other_hde.set_selection(self.selection_mask)

        # update tab1
        self.tab1.update_selection(self.selection_mask)

    # ==================== top bar ==================== #

    def set_dimension(self, dim):
        get_widget(self.widget, "100").v_model = dim == 3
        self.vs_hde.set_dim(dim)
        self.es_hde.set_dim(dim)

    @log_errors
    def switch_dimension(self, widget, event, data):
        """
        Called when the switch changes.
        We call the HighDimExplorer to update its figure and, enventually,
        compute its proj
        """
        stats_logger.log('dim_changed')
        self.set_dimension(3 if data else 2)

    @log_errors
    def change_color(self, widget, event, data):
        """
        Called with the user clicks on the colorChoiceBtnToggle
        Allows change the color of the dots
        """

        # Color : a pd.Series with one color value par row
        color = None
        stats_logger.log('color_changed', {'color': data})
        if data == "y":
            color = self.y
        elif data == "y^":
            color = self.y_pred
        elif data == "residual":
            color = self.y - self.y_pred

        self.vs_hde.figure.set_color(color, 0)
        self.es_hde.figure.set_color(color, 0)
        self.select_tab(0)

    # ==================== TAB handling ==================== #

    def select_tab_front(self, tab):
        @log_errors
        def call_fct(*args):
            stats_logger.log('tab_selected', {'tab': tab})
            self.select_tab(tab, front=True)

        return call_fct

    def select_tab(self, tab, front=False):
        if tab == 1 and (not self.selection_mask.any() or self.selection_mask.all()):
            return self.select_tab(0)
        if tab == 1:
            self.vs_hde.figure.display_selection()
            self.es_hde.figure.display_selection()
        elif tab == 2:
            self.tab2.update_region_table()
            self.update_region_callback(self, self.region_set)
        elif tab == 3:
            if len(self.tab2.selected_regions) == 0:
                self.select_tab(2)
            else:
                region = self.region_set.get(self.tab2.selected_regions[0]['Region'])
                self.tab3.update_region(region, False)
                if region is None:
                    region = ModelRegion(self.X, self.y, self.X_test, self.y_test, self.model, score=self.score)
                self.vs_hde.figure.display_region(region)
                self.es_hde.figure.display_region(region)
        if not front:
            get_widget(self.widget, "4").v_model = max(tab - 1, 0)
        self.vs_hde.set_tab(tab)
        self.es_hde.set_tab(tab)
        self.tab = tab
        self.disable_hde()

    # ==================== TAB 1 ==================== #

    def validate_rules_callback(self, rules_set: RuleSet):
        self.selection_changed(None, boolean_mask(self.X, True))

        region = self.region_set.add_region(rules=rules_set)
        region.validate()
        self.select_tab(2)

    def new_rule_selected_callback(self, selection_mask, rules_mask):
        self.select_tab(1)
        self.vs_hde.figure.display_rules(selection_mask, rules_mask)
        self.es_hde.figure.display_rules(selection_mask, rules_mask)

    # ==================== TAB 2 ==================== #

    def update_region_callback(self, caller, region_set):
        self.vs_hde.figure.display_regionset(region_set)
        self.es_hde.figure.display_regionset(region_set)

    def substitute_model_callback(self, region):
        self.select_tab(3)
        self.tab3.update_region(region)

    # ==================== TAB 3 ==================== #

    @log_errors
    def model_validation_callback(self, *args):
        self.select_tab(2)
        self.tab2.selected_regions = []
