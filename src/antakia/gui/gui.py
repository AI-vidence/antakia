from __future__ import annotations

import time
from typing import Callable

import pandas as pd

import ipyvuetify as v
import IPython.display

from antakia_core.data_handler import Region

from antakia.gui.app_bar.color_switch import ColorSwitch
from antakia.gui.app_bar.dimension_switch import DimSwitch
from antakia.gui.helpers.data import DataStore
from antakia.gui.splash_screen import SplashScreen
from antakia.gui.app_bar.top_bar import TopBar
from antakia.gui.app_bar.explanation_values import ExplanationValues
from antakia_core.explanation import ExplanationMethod
from antakia.config import AppConfig

from antakia.gui.tabs.tab1 import Tab1
from antakia.gui.tabs.tab2 import Tab2
from antakia.gui.tabs.tab3 import Tab3
from antakia.gui.high_dim_exp.highdimexplorer import HighDimExplorer

from antakia.gui.helpers.metadata import metadata

import logging
from antakia.utils.logging_utils import conf_logger, Log
from antakia_core.utils import boolean_mask, timeit

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

    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        metadata.start()
        self.tab_value = 1
        # Init value space widgets

        # star dialog
        self.topbar = TopBar()

        self.dimension_switch = DimSwitch(self.dimension_update_callback)
        self.color_switch = ColorSwitch(self.data_store,
                                        self.color_update_callback)

        # first hde
        with Log('building vs hde', 2):
            self.vs_hde = HighDimExplorer(self.data_store,
                                          self.selection_changed, 'VS')

        # init Explanation space
        # first explanation getter/compute
        with Log('building exp values', 2):
            self.exp_values = ExplanationValues(
                self.data_store, self.explanation_changed_callback,
                self.disable_hde)
        # then hde
        with Log('building es hde', 2):
            self.es_hde = HighDimExplorer(self.data_store,
                                          self.selection_changed, 'ES')

        # init tabs
        with Log('building tab1', 2):
            self.tab1 = Tab1(self.data_store, self.new_rule_selected_callback,
                             self.validate_rules_callback)

        with Log('building tab2', 2):
            self.tab2 = Tab2(self.data_store,
                             self.vs_hde.projected_value_selector,
                             self.es_hde.projected_value_selector,
                             self.edit_region_callback,
                             self.update_region_callback,
                             self.substitute_model_callback)

        with Log('building tab3', 2):
            self.tab3 = Tab3(self.data_store, self.model_validation_callback,
                             self.display_model_data)

        with Log('building widget', 2):
            self._build_widget()
            self.splash = SplashScreen()

    def _build_widget(self):
        self.widget = v.Col(children=[
            self.topbar.widget,
            v.Row(  # Top buttons bar # 1
                class_="mt-3 align-center",
                children=[
                    v.Tooltip(  # 10
                        bottom=True,
                        v_slots=[{
                            'name': 'activator',
                            'variable': 'tooltip',
                            'children': self.dimension_switch.widget
                        }  # End v_slots dict
                        ],  # End v_slots list
                        children=['Change dimensions']),  # End v.Tooltip
                    self.color_switch.widget,
                    v.Col(  # 12
                        class_="ml-4 mr-4",
                        children=[self.exp_values.widget]),
                    v.Col(  # 13 VS proj Select
                        class_="ml-6 mr-6",
                        children=[self.vs_hde.projected_value_selector.widget
                                  ]),
                    v.Col(  # 14 ES proj Select
                        class_="ml-6 mr-6",
                        children=[self.es_hde.projected_value_selector.widget
                                  ]),
                ]),
            v.Row(  # The two HighDimExplorer # 2
                class_="d-flex",
                children=[
                    v.Col(  # VS HDE # 20
                        style_="width: 50%",
                        class_="d-flex flex-column justify-center",
                        children=[
                            v.Html(  # 200
                                tag="h3",
                                style_="align-self: center",
                                class_="mb-3",
                                children=["Values space"]),
                            self.vs_hde.figure.widget,
                        ],
                    ),
                    v.Col(  # ES HDE placeholder # 21
                        style_="width: 50%",
                        class_="d-flex flex-column justify-center",
                        children=[
                            v.Html(  # 210
                                tag="h3",
                                style_="align-self: center",
                                class_="mb-3",
                                children=["Explanations space"]),
                            self.es_hde.figure.widget
                        ],
                    ),
                ],
            ),
            v.Divider(),  # 3
            v.Tabs(  # 4
                v_model=0,  # default active tab
                children=[
                             v.Tab(children=["Selection"]),  # 40
                             v.Tab(children=["Regions"]),  # 41
                             v.Tab(children=["Substitution"]),  # 42
                         ] + [
                             v.TabItem(  # Tab 1)
                                 class_="mt-2", children=self.tab1.widget),
                             v.TabItem(  # Tab 2) Regions #44
                                 children=self.tab2.widget),  # End of v.TabItem #2
                             v.TabItem(  # TabItem #3 Substitution #45
                                 children=self.tab3.widget)
                         ])  # End of v.Tabs
        ]  # End v.Col children
        )  # End of v.Col

    @timeit
    def compute_base_values(self):
        # We trigger ES explain computation if needed :
        with Log('initializing explanations', 1) as log:
            if not self.exp_values.has_user_exp:  # No imported explanation values
                exp_method = ExplanationMethod.explain_method_as_str(
                    AppConfig.ATK_DEFAULT_EXPLANATION_METHOD)
                msg = f"Computing {exp_method} on {self.data_store.X.shape}"
            else:
                msg = f"Imported explained values {self.data_store.X.shape}"
            self.splash.set_exp_msg(msg)
            self.splash.exp_progressbar.set_log(log)
            self.exp_values.initialize(self.splash.exp_progressbar)

        # We trigger VS proj computation :
        pb1, pb2 = self.splash.proj_progressbar.split(50)
        with Log('projecting Value space', 1) as log:
            self.splash.set_proj_msg(
                f"{AppConfig.ATK_DEFAULT_PROJECTION} on {self.data_store.X.shape} 1/2"
            )
            pb1.set_log(log)
            self.vs_hde.initialize(progress_callback=pb1, X=self.data_store.X)

        # THen we trigger ES proj computation :
        with Log('projecting Explanation space', 1) as log:
            self.splash.set_proj_msg(
                f"{AppConfig.ATK_DEFAULT_PROJECTION} on {self.data_store.X.shape} 2/2"
            )
            pb2.set_log(log)
            self.es_hde.initialize(progress_callback=pb2,
                                   X=self.exp_values.current_exp_df)
        step = 'updating es rules'
        with Log(step, 1):
            self.splash.set_proj_msg(step)
            self.tab1.refresh_X_exp()
        step = 'refreshing rule_widget'
        with Log(step, 1):
            self.splash.set_proj_msg(step)
            main_variables = self.exp_values.current_exp_df.abs().mean(
            ).sort_values(ascending=False).iloc[:10].index
            self.data_store.variables.set_main_variables(
                main_variables.to_list())
            self.tab1.initialize()
        self.select_tab(0)
        self.disable_hde(self, 'compute_base_values')

        if metadata.counter == 10:
            self.topbar.open()

    @log_errors
    def initialize(self):
        """Displays the splash screen and updates it during the first computations."""

        # We add both widgets to the current notebook cell and hide them
        t = time.time()
        self.widget.hide()
        self.splash.widget.show()
        IPython.display.display(self.splash.widget, self.widget)

        self.compute_base_values()
        self.wire()

        self.splash.widget.hide()
        self.widget.show()
        # redraw figures once app is displayed to be able to autosize it
        with Log('refreshing figures', 2):
            self.vs_hde.figure.rebuild()
            self.es_hde.figure.rebuild()
        stats_logger.log('gui_init_end', {'load_time': time.time() - t})

    def wire(self):
        """
        wires the app_widget, and implements UI logic
        """

        # ================ Tab Selection ================

        # We wire the click event on 'Tabs'
        tabs = self.widget.children[4].children
        tabs[0].on_event("click", self.select_tab_front(1))
        tabs[1].on_event("click", self.select_tab_front(2))
        tabs[2].on_event("click", self.select_tab_front(3))

    # ==================== sync callbacks ==================== #

    @log_errors
    @timeit
    def explanation_changed_callback(self,
                                     caller,
                                     event,
                                     progress_callback: Callable
                                                        | None = None):
        """
        on explanation change, synchronizes es_hde and tab1
        Parameters
        ----------
        current_exp_df
        progress_callback

        Returns
        -------

        """
        self.es_hde.update_X()
        self.es_hde.refresh(progress_callback)

        self.tab1.refresh_X_exp()

    @log_errors
    @timeit
    def disable_hde(self, caller, event, disable='auto'):
        if disable == 'auto':

            disable_proj = bool((self.tab_value == 0)
                                and not self.data_store.empty_selection)
            disable_figure = bool(self.tab_value > 1)
        else:
            disable_proj = disable
            disable_figure = disable
        self.vs_hde.disable(disable_figure, disable_proj)
        self.exp_values.disable_selection(disable_proj)
        self.es_hde.disable(disable_figure, disable_proj)

    @log_errors
    @timeit
    def selection_changed(self, caller, event):
        """
        callback to synchronize both hdes and tab1
        Parameters
        ----------
        caller
        event

        Returns
        -------

        """
        """Called when the selection of one HighDimExplorer changes"""
        # If new selection (empty or not) : if exists, we remove any 'pending rule'
        if self.data_store.empty_selection:
            # Selection is empty
            if self.tab1.edit_type == self.tab1.CREATE_RULE:
                self.select_tab(0, msg='unselect')
                self.tab1.reset()
            else:
                self.select_tab(1, msg='unselect')
                # reset to tab1 region
                self.tab1.refresh()
                caller = None
                stats_logger.log(
                    'deselection', {
                        'exp_method':
                            self.exp_values.current_exp,
                        'vs_proj':
                            str(self.vs_hde.projected_value_selector.current_proj),
                        'es_proj':
                            str(self.es_hde.projected_value_selector.current_proj)
                    })
        else:
            stats_logger.log(
                'selection_gui', {
                    'exp_method':
                        self.exp_values.current_exp,
                    'vs_proj':
                        str(self.vs_hde.projected_value_selector.current_proj),
                    'es_proj':
                        str(self.es_hde.projected_value_selector.current_proj)
                })
        self.disable_hde(self, 'selection_changed')
        if self.tab_value == 1:
            self.vs_hde.figure.display_rules()
            self.es_hde.figure.display_rules()
        else:
            self.vs_hde.display_selection()
            self.es_hde.display_selection()
        if caller != self.tab1:
            self.tab1.refresh()

    # ==================== top bar ==================== #

    def dimension_update_callback(self, caller, dim):
        self.vs_hde.set_dim(dim)
        self.es_hde.set_dim(dim)
        self.disable_hde(caller, 'dimension_changed')

    @log_errors
    def color_update_callback(self, caller, color):
        """
        Called with the user clicks on the colorChoiceBtnToggle
        Allows change the color of the dots
        """
        self.vs_hde.figure.set_color(color, 0)
        self.es_hde.figure.set_color(color, 0)
        self.select_tab(0, msg='color')

    # ==================== TAB handling ==================== #

    def select_tab_front(self, tab):

        @log_errors
        def call_fct(*args):
            with Log(f'front_tab_{tab}_selected', 2):
                stats_logger.log('tab_selected', {'tab': tab})
                self.select_tab(tab, front=True, msg='front')

        return call_fct

    @timeit
    def select_tab(self, tab, front=False, msg=None):
        if tab == 1 and (self.data_store.empty_selection):
            return self.select_tab(0)
        elif tab == 2:
            # refresh region set display
            self.update_region_callback(self)
        elif tab == 3:
            if self.tab3.region is not None:
                region = self.tab3.region
                self.es_hde.figure.display_region(region)
                self.vs_hde.figure.display_region(region)
            else:
                self.select_tab(2, msg='no region selected')
        if not front:
            self.widget.children[4].v_model = max(tab - 1, 0)
        self.vs_hde.set_tab(tab)
        self.es_hde.set_tab(tab)
        self.tab_value = tab
        self.disable_hde(self, 'select_tab')

    # ==================== TAB 1 ==================== #

    @timeit
    def new_rule_selected_callback(self, caller, event: str):
        if self.data_store.empty_selection:
            # no selection mode - we edit keep the self selection mask clean
            self.vs_hde.figure.display_selection()
            self.es_hde.figure.display_selection()
            self.select_tab(0, msg='new_rule')
        else:
            self.select_tab(1, msg='new_rule')
            self.vs_hde.figure.display_rules()
            self.es_hde.figure.display_rules()

    @timeit
    def validate_rules_callback(self, caller, event: str, region: Region):
        """
        Callback method for validating rules in the GUI.

        Parameters:
            caller: The object that triggered the event.
            event (str): The type of event that occurred.
            region (Region): The region object to be validated.

        Returns:
            None

        Description:
            This method is called when the user validates a set of rules in the GUI. It updates the selection mask in
            the data store to include all data points, triggers the selection_changed method to update the displayed
            selection, validates the region object, adds the region to the region set in the data store, updates the
            region table in the tab2 widget, and selects the tab2 in the GUI.

        """
        self.data_store.selection_mask = boolean_mask(self.data_store.X, True)
        self.selection_changed(caller, event)
        region.validate()
        self.data_store.region_set.add(region)
        self.tab2.update_region_table()
        self.select_tab(2, msg='validate')

    # ==================== TAB 2 ==================== #

    @timeit
    def edit_region_callback(self, caller, region):
        self.tab1.update_region(region)
        self.select_tab(1, msg='edit_Region')
        self.vs_hde.figure.display_rules()
        self.es_hde.figure.display_rules()

    @timeit
    def update_region_callback(self, caller):
        self.vs_hde.figure.display_regionset(self.data_store.region_set)
        self.es_hde.figure.display_regionset(self.data_store.region_set)
        self.tab2.update_region_table()

    @timeit
    def substitute_model_callback(self, caller, region):
        self.vs_hde.figure.display_region(region)
        self.es_hde.figure.display_region(region)
        self.select_tab(3, msg='substitute')
        self.tab3.update_region(region)

    # ==================== TAB 3 ==================== #

    @log_errors
    @timeit
    def model_validation_callback(self, *args):
        self.tab2.update_region_table()
        self.tab2.selected_regions = []
        self.select_tab(2, msg='model_validated')

    @timeit
    def display_model_data(self, region, y=None):
        if y is None:
            self.vs_hde.figure.display_region(region)
            self.es_hde.figure.display_region(region)
        else:
            self.vs_hde.figure.display_region_value(region, y)
            self.es_hde.figure.display_region_value(region, y)
