from __future__ import annotations

import numpy as np
import pandas as pd

import ipyvuetify as v
import IPython.display

from antakia_core.data_handler.region import ModelRegionSet, ModelRegion

from antakia.gui.antakia_logo import AntakiaLogo
from antakia.gui.explanation_values import ExplanationValues
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank
from antakia.gui.progress_bar import ProgressBar, MultiStepProgressBar
from antakia.explanation.explanation_method import ExplanationMethod
from antakia_ac.auto_cluster import AutoCluster
from antakia_core.compute.skope_rule.skope_rule import skope_rules
import antakia.config as config
from antakia_core.data_handler.rules import RuleSet

from antakia.gui.tabs.model_explorer import ModelExplorer
from antakia.gui.widgets import get_widget, change_widget, splash_widget, app_widget
from antakia.gui.high_dim_exp.highdimexplorer import HighDimExplorer
from antakia.gui.ruleswidget import RulesWidget

import copy
from os import path
from json import dumps, loads

import logging
from antakia.utils.logging import conf_logger
from antakia_core.utils.utils import boolean_mask, ProblemCategory
from antakia_core.utils.variable import DataVariables

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
        self.new_selection = False
        self.selection_mask = boolean_mask(X, True)

        self.pv_bank = ProjectedValueBank(y)

        # star dialog
        self.logo = AntakiaLogo()

        # first hde
        self.vs_hde = HighDimExplorer(
            self.pv_bank,
            self.selection_changed,
            'VS'
        )
        # then rules
        self.vs_rules_wgt = RulesWidget(self.X, self.y, self.variables, True, self.new_rules_defined)

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
        # finally rules
        self.es_rules_wgt = RulesWidget(X_exp, self.y, self.variables, False)

        # We set empty rules for now :
        self.vs_rules_wgt.disable()
        self.es_rules_wgt.disable()

        # init tabs
        self.model_explorer = ModelExplorer(self.X)

        self.region_num_for_validated_rules = None  # tab 1 : number of the region created when validating rules
        self.region_set = ModelRegionSet(self.X, self.y, self.X_test, self.y_test, self.model, self.score)
        self.substitute_region = None
        self.substitution_model_training = False  # tab 3 : training flag
        self.widget = app_widget.get_app_widget()
        self.splash_widget = splash_widget.get_app_widget()
        # UI rules :
        # We disable the selection datatable at startup (bottom of tab 1)
        get_widget(self.widget, "4320").disabled = True

        # We count the number of times this GUI has been initialized
        self.counter = loads(open("counter.json", "r").read()) if path.exists("counter.json") else 0
        self.counter += 1
        with open("counter.json", "w") as f:
            f.write(dumps(self.counter))
        logger.debug(f"GUI has been initialized {self.counter} times")

    def show_splash_screen(self):
        """Displays the splash screen and updates it during the first computations."""

        # We add both widgets to the current notebook cell and hide them
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
        ).v_model = f"{config.DEFAULT_PROJECTION} on {self.X.shape} 1/2"

        self.vs_hde.initialize(progress_callback=dimreduc_progress_bar.get_update(1), X=self.X)

        # We trigger ES explain computation if needed :
        if not self.exp_values.has_user_exp:  # No imported explanation values
            exp_method = ExplanationMethod.explain_method_as_str(config.DEFAULT_EXPLANATION_METHOD)
            msg = f"Computing {exp_method} on {self.X.shape}"
        else:
            msg = f"Imported explained values {self.X.shape}"
        self.exp_values.initialize(exp_progress_bar.update)
        get_widget(self.splash_widget, "120").v_model = msg

        # THen we trigger ES proj computation :
        get_widget(
            self.splash_widget, "220"
        ).v_model = f"{config.DEFAULT_PROJECTION} on {self.X.shape} 2/2"
        self.es_hde.initialize(
            progress_callback=dimreduc_progress_bar.get_update(2),
            X=self.exp_values.current_exp_df
        )
        self.es_rules_wgt.update_X(self.exp_values.current_exp_df)
        self.selection_changed(None, boolean_mask(self.X, True))

        self.init_app()

        self.splash_widget.hide()
        self.widget.show()
        self.vs_hde.figure.create_figure()
        self.es_hde.figure.create_figure()
        self.select_tab(0)
        self.disable_hde()

        if self.counter == 10:
            self.logo.open()

    def init_app(self):
        """
        Inits and wires the app_widget, and implements UI logic
        """

        # -------------- Dimension Switch --------------

        get_widget(self.widget,'00').children = [self.logo.widget]

        # -------------- Dimension Switch --------------

        get_widget(self.widget, "100").v_model = config.DEFAULT_DIMENSION == 3
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

        # We add our 2 RulesWidgets to the GUI :
        change_widget(self.widget, "4310", self.vs_rules_wgt.root_widget)
        change_widget(self.widget, "4311", self.es_rules_wgt.root_widget)

        # We wire the click event on the 'Find-rules' button
        get_widget(self.widget, "43010").on_event("click", self.compute_skope_rules)

        # We wire the ckick event on the 'Undo' button
        get_widget(self.widget, "4302").on_event("click", self.undo_rules)

        # Its enabled when rules graphs have been updated with rules
        # We wire the click event on the 'Valildate rules' button
        get_widget(self.widget, "43030").on_event("click", self.validate_rules)

        # It's enabled when a SKR rules has been found and is disabled when the selection gets empty
        # or when validated is pressed

        # ================ Tab 2 : regions ===============
        # We wire the click event on 'Tab 2'
        get_widget(self.widget, "41").on_event("click", self.select_tab_front(2))

        get_widget(self.widget, "44001").set_callback(self.region_selected)

        # We wire events on the 'substitute' button:
        get_widget(self.widget, "4401000").on_event("click", self.substitute_clicked)
        # button is disabled by default
        get_widget(self.widget, "4401000").disabled = True

        # We wire events on the 'divide' button:
        get_widget(self.widget, "4401100").on_event("click", self.divide_region_clicked)
        # button is disabled by default
        get_widget(self.widget, "4401100").disabled = True

        # We wire events on the 'merge' button:
        get_widget(self.widget, "4401200").on_event("click", self.merge_region_clicked)
        # button is disabled by default
        get_widget(self.widget, "4401200").disabled = True

        # We wire events on the 'delete' button:
        get_widget(self.widget, "4401300").on_event("click", self.delete_region_clicked)
        # The 'delete' button is disabled at startup
        get_widget(self.widget, "4401300").disabled = True

        # We wire events on the 'auto-cluster' button :
        get_widget(self.widget, "4402000").on_event("click", self.auto_cluster_clicked)

        # UI rules :
        # The 'auto-cluster' button is disabled at startup
        get_widget(self.widget, "4402000").disabled = True
        # Checkbox automatic number of cluster is set to True at startup
        get_widget(self.widget, "440211").v_model = True

        # We wire select events on this checkbox :
        get_widget(self.widget, "440211").on_event("change", self.checkbox_auto_cluster_clicked)

        def num_cluster_changed(*args):
            """
            Called when the user changes the number of clusters
            """
            # We enable the 'auto-cluster' button
            get_widget(self.widget, "4402000").disabled = False

        # We wire events on the num cluster Slider
        get_widget(self.widget, "4402100").on_event("change", num_cluster_changed)

        # UI rules : at startup, the slider is disabled and the checkbox is checked
        get_widget(self.widget, "4402100").disabled = True

        self.update_region_table()
        # At startup, REGIONSET_TRACE is not visible

        # ============== Tab 3 : substitution ==================

        # We wire the click event on 'Tab 3'
        get_widget(self.widget, "42").on_event("click", self.select_tab_front(3))

        # UI rules :
        # At startup validate sub-model btn is disabled :
        get_widget(self.widget, "4501000").disabled = True

        # We wire a select event on the 'substitution table' :
        get_widget(self.widget, "45001").set_callback(self.sub_model_selected_callback)

        # We wire a ckick event on the "validate sub-model" button :
        get_widget(self.widget, "4501000").on_event("click", self.validate_sub_model)
        get_widget(self.widget, "4502").children = [self.model_explorer.widget]

        # We disable the Substitution table at startup :
        self.update_substitution_table(None)

        self.refresh_buttons_tab_1()

    # ==================== properties ==================== #

    @property
    def selected_regions(self):
        return get_widget(self.widget, "44001").selected

    @selected_regions.setter
    def selected_regions(self, value):
        get_widget(self.widget, "44001").selected = value
        self.disable_buttons(None)

    @property
    def selected_sub_model(self):
        return get_widget(self.widget, "45001").selected

    @selected_sub_model.setter
    def selected_sub_model(self, value):
        get_widget(self.widget, "45001").selected = value

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

    def explanation_changed_callback(self, current_exp_df: pd.DataFrame, progress_callback: callable = None):
        self.es_hde.update_X(current_exp_df, progress_callback)
        self.es_rules_wgt.update_X(current_exp_df)

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

    def selection_changed(self, caller: HighDimExplorer | None, new_selection_mask: pd.Series):
        """Called when the selection of one HighDimExplorer changes"""

        # UI rules :
        # If new selection (empty or not) : if exists, we remove any 'pending rule'
        self.new_selection = True
        self.disable_hde()
        if new_selection_mask.all():
            # Selection is empty
            # we display y as color
            self.select_tab(0)
            # we reset rules_widgets
            self.vs_rules_wgt.disable()
            self.es_rules_wgt.disable()
            self.es_rules_wgt.reset_widget()
            self.vs_rules_wgt.reset_widget()
        else:
            # Selection is not empty anymore or changes
            X_rounded = copy.copy((self.X.loc[new_selection_mask])).round(3)
            change_widget(
                self.widget,
                "432010",
                v.DataTable(
                    v_model=[],
                    show_select=False,
                    headers=[{"text": column, "sortable": True, "value": column} for column in self.X.columns],
                    items=X_rounded.to_dict("records"),
                    hide_default_footer=False,
                    disable_sort=False,
                ),
            )

        # We store the new selection
        self.selection_mask = new_selection_mask
        # We synchronize selection between the two HighDimExplorers
        if caller is None:
            self.es_hde.set_selection(self.selection_mask)
            self.vs_hde.set_selection(self.selection_mask)
        else:
            other_hde = self.es_hde if caller == self.vs_hde.figure else self.vs_hde
            other_hde.set_selection(self.selection_mask)

        # We update the selection status :
        if not self.selection_mask.all():
            selection_status_str_1 = f"{self.selection_mask.sum()} point selected"
            selection_status_str_2 = f"{100 * self.selection_mask.mean():.2f}% of the  dataset"
        else:
            selection_status_str_1 = f"0 point selected"
            selection_status_str_2 = f"0% of the  dataset"
        change_widget(self.widget, "4300000", selection_status_str_1)
        change_widget(self.widget, "430010", selection_status_str_2)
        # we refresh button and enable/disable the datatable
        self.refresh_buttons_tab_1()

    def new_rules_defined(self, rules_widget: RulesWidget, df_mask: pd.Series):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        # We sent to the proper HDE the rules_indexes to render :
        self.vs_hde.figure.display_rules(selection_mask=self.selection_mask, rules_mask=df_mask)
        self.es_hde.figure.display_rules(selection_mask=self.selection_mask, rules_mask=df_mask)

        # sync selection between rules_widgets
        if rules_widget == self.vs_rules_wgt:
            self.es_rules_wgt.update_from_mask(df_mask, RuleSet(), sync=False)
        else:
            self.vs_rules_wgt.update_from_mask(df_mask, RuleSet(), sync=False)

        self.refresh_buttons_tab_1()

    # ==================== top bar ==================== #

    def set_dimension(self, dim):
        get_widget(self.widget, "100").v_model = dim == 3
        self.vs_hde.set_dim(dim)
        self.es_hde.set_dim(dim)

    def switch_dimension(self, widget, event, data):
        """
        Called when the switch changes.
        We call the HighDimExplorer to update its figure and, enventually,
        compute its proj
        """
        self.set_dimension(3 if data else 2)

    def change_color(self, widget, event, data):
        """
        Called with the user clicks on the colorChoiceBtnToggle
        Allows change the color of the dots
        """

        # Color : a pd.Series with one color value par row
        color = None

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
        def call_fct(*args):
            self.select_tab(tab, front=True)

        return call_fct

    def select_tab(self, tab, front=False):
        if tab == 1 and (not self.selection_mask.any() or self.selection_mask.all()):
            return self.select_tab(0)
        if tab == 1:
            self.vs_hde.figure.display_selection()
            self.es_hde.figure.display_selection()
        elif tab == 2:
            self.update_region_table()
            self.vs_hde.figure.display_regionset(self.region_set)
            self.es_hde.figure.display_regionset(self.region_set)
        elif tab == 3:
            if len(self.selected_regions) == 0:
                self.select_tab(2)
            else:
                region = self.region_set.get(self.selected_regions[0]['Region'])
                self.update_substitution_table(region)
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

    def refresh_buttons_tab_1(self):
        self.disable_hde()
        # data table
        get_widget(self.widget, "4320").disabled = bool(self.selection_mask.all())
        # skope_rule
        get_widget(self.widget, "43010").disabled = not self.new_selection or bool(self.selection_mask.all())
        # undo
        get_widget(self.widget, "4302").disabled = not (self.vs_rules_wgt.rules_num > 1)
        # validate rule
        get_widget(self.widget, "43030").disabled = not (self.vs_rules_wgt.rules_num > 0)

    def compute_skope_rules(self, *args):
        self.new_selection = False

        if self.tab != 1:
            self.select_tab(1)
        # compute skope rules
        skr_rules_list, skr_score_dict = skope_rules(self.selection_mask, self.vs_hde.current_X, self.variables)
        skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
        # init vs rules widget
        self.vs_rules_wgt.init_rules(skr_rules_list, skr_score_dict, self.selection_mask)
        # update VS and ES HDE
        self.vs_hde.figure.display_rules(
            selection_mask=self.selection_mask,
            rules_mask=skr_rules_list.get_matching_mask(self.X)
        )
        self.es_hde.figure.display_rules(
            selection_mask=self.selection_mask,
            rules_mask=skr_rules_list.get_matching_mask(self.X)
        )

        es_skr_rules_list, es_skr_score_dict = skope_rules(self.selection_mask, self.es_hde.current_X, self.variables)
        es_skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
        self.es_rules_wgt.init_rules(es_skr_rules_list, es_skr_score_dict, self.selection_mask)
        self.refresh_buttons_tab_1()
        self.select_tab(1)

    def undo_rules(self, *args):
        if self.tab != 1:
            self.select_tab(1)
        if self.vs_rules_wgt.rules_num > 0:
            self.vs_rules_wgt.undo()
        else:
            # TODO : pourquoi on annule d'abord le VS puis l'ES?
            self.es_rules_wgt.undo()
        self.refresh_buttons_tab_1()

    def validate_rules(self, *args):
        if self.tab != 1:
            self.select_tab(1)

        rules_list = self.vs_rules_wgt.current_rules_list
        # UI rules :
        # We clear selection
        self.selection_changed(None, boolean_mask(self.X, True))
        # We clear the RulesWidget
        self.es_rules_wgt.reset_widget()
        self.vs_rules_wgt.reset_widget()
        if len(rules_list) == 0:
            self.vs_rules_wgt.show_msg("No rules found on Value space cannot validate region", "red--text")
            return

        # We add them to our region_set
        region = self.region_set.add_region(rules=rules_list)
        self.region_num_for_validated_rules = region.num
        # lock rule
        region.validate()

        # And update the rules table (tab 2)
        # we refresh buttons
        self.refresh_buttons_tab_1()
        # We force tab 2
        self.select_tab(2)

    # ==================== TAB 2 ==================== #

    def update_region_table(self):
        """
        Called to empty / fill the RegionDataTable and refresh plots
        """
        self.region_set.sort(by='size', ascending=False)
        temp_items = self.region_set.to_dict()

        # We populate the ColorTable :
        get_widget(self.widget, "44001").items = temp_items

        region_stats = self.region_set.stats()
        str_stats = [
            f"{region_stats['regions']} {'regions' if region_stats['regions'] > 1 else 'region'}",
            f"{region_stats['points']} points",
            f"{region_stats['coverage']}% of the dataset",
            f"{region_stats['delta_score']:.2f} subst score"
        ]
        get_widget(self.widget, "44002").children = [
            ', '.join(str_stats)
        ]
        get_widget(self.widget, "4402000").disabled = False

    def checkbox_auto_cluster_clicked(self, widget, event, data):
        """
        Called when the user clicks on the 'auto-cluster' checkbox
        """
        if self.tab != 2:
            self.select_tab(2)
        # In any case, we enable the auto-cluster button
        get_widget(self.widget, "4402000").disabled = False

        # We reveive either True or {} (bool({})==False))
        data = bool(data)

        # IF true, we disable the Slider
        get_widget(self.widget, "4402100").disabled = data

    def auto_cluster_clicked(self, *args):
        """
        Called when the user clicks on the 'auto-cluster' button
        """
        # We disable the AC button. Il will be re-enabled when the AC progress is 100%
        get_widget(self.widget, "4402000").disabled = True
        if self.tab != 2:
            self.select_tab(2)
        if self.region_set.stats()["coverage"] > 80:
            # UI rules :
            # region_set coverage is > 80% : we need to clear it to do another auto-cluster
            self.region_set.clear_unvalidated()

        # We assemble indices ot all existing regions :
        region_set_mask = self.region_set.mask
        not_rules_indexes_list = ~region_set_mask
        # We call the auto_cluster with remaining X and explained(X) :
        if get_widget(self.widget, "440211").v_model:
            cluster_num = "auto"
        else:
            cluster_num = get_widget(self.widget, "4402100").v_model - len(self.region_set)
            if cluster_num <= 2:
                cluster_num = 2

        self.compute_auto_cluster(not_rules_indexes_list, cluster_num)

        # We re-enable the button
        get_widget(self.widget, "4402000").disabled = False
        self.select_tab(2)

    def compute_auto_cluster(self, not_rules_indexes_list, cluster_num='auto'):
        if len(not_rules_indexes_list) > config.MIN_POINTS_NUMBER:
            vs_compute = int(not self.vs_hde.projected_value_selector.is_computed(dim=3))
            es_compute = int(not self.es_hde.projected_value_selector.is_computed(dim=3))
            steps = 1 + vs_compute + es_compute

            progress_bar = MultiStepProgressBar(get_widget(self.widget, "440212"), steps=steps)
            step = 1
            vs_proj_3d_df = self.vs_hde.get_current_X_proj(
                3,
                progress_callback=progress_bar.get_update(step)
            )

            step += vs_compute
            es_proj_3d_df = self.es_hde.get_current_X_proj(
                3,
                progress_callback=progress_bar.get_update(step)
            )

            step += es_compute
            ac = AutoCluster(self.X, progress_bar.get_update(step))

            found_regions = ac.compute(
                vs_proj_3d_df.loc[not_rules_indexes_list],
                es_proj_3d_df.loc[not_rules_indexes_list],
                # We send 'auto' or we read the number of clusters from the Slider
                cluster_num,
            )  # type: ignore
            self.region_set.extend(found_regions)
            progress_bar.set_progress(100)
        else:
            print('not enough points to cluster')

    def disable_buttons(self, current_operation):
        selected_region_nums = [x['Region'] for x in self.selected_regions]
        if current_operation:
            if current_operation['type'] == 'select':
                selected_region_nums.append(current_operation['region_num'])
            elif current_operation['type'] == 'unselect':
                selected_region_nums.remove(current_operation['region_num'])
        num_selected_regions = len(selected_region_nums)
        if num_selected_regions:
            first_region = self.region_set.get(selected_region_nums[0])
            enable_div = (num_selected_regions == 1) and bool(first_region.num_points() >= config.MIN_POINTS_NUMBER)
        else:
            enable_div = False

        # substitute
        get_widget(self.widget, "4401000").disabled = num_selected_regions != 1

        # divide
        get_widget(self.widget, "4401100").disabled = not enable_div

        # merge
        enable_merge = (num_selected_regions > 1)
        get_widget(self.widget, "4401200").disabled = not enable_merge

        # delete
        get_widget(self.widget, "4401300").disabled = num_selected_regions == 0

    def region_selected(self, data):
        if self.tab != 2:
            self.select_tab(2)
        operation = {
            'type': 'select' if data['value'] else 'unselect',
            'region_num': data['item']['Region']
        }
        self.disable_buttons(operation)

    def clear_selected_regions(self):
        self.selected_regions = []
        self.disable_buttons(None)

    def divide_region_clicked(self, *args):
        """
        Called when the user clicks on the 'divide' (region) button
        """
        if self.tab != 2:
            self.select_tab(2)
        # we recover the region to sudivide
        region = self.region_set.get(self.selected_regions[0]['Region'])
        if region.num_points() > config.MIN_POINTS_NUMBER:
            # Then we delete the region in self.region_set
            self.region_set.remove(region.num)
            # we compute the subregions and add them to the region set
            if get_widget(self.widget, "440211").v_model:
                cluster_num = "auto"
            else:
                cluster_num = get_widget(self.widget, "4402100").v_model - len(self.region_set)
                if cluster_num <= 2:
                    cluster_num = 2
            self.compute_auto_cluster(region.mask, cluster_num)
        self.select_tab(2)
        # There is no more selected region
        self.clear_selected_regions()

    def merge_region_clicked(self, *args):
        """
        Called when the user clicks on the 'merge' (regions) button
        """

        selected_regions = [self.region_set.get(r['Region']) for r in self.selected_regions]
        mask = None
        for region in selected_regions:
            if mask is None:
                mask = region.mask
            else:
                mask |= region.mask

        # compute skope rules
        skr_rules_list, _ = skope_rules(mask, self.vs_hde.current_X, self.variables)

        # delete regions
        for region in selected_regions:
            self.region_set.remove(region.num)
        # add new region
        if len(skr_rules_list) > 0:
            r = self.region_set.add_region(rules=skr_rules_list)
        else:
            r = self.region_set.add_region(mask=mask)
        self.selected_regions = [{'Region': r.num}]
        self.select_tab(2)

    def delete_region_clicked(self, *args):
        """
        Called when the user clicks on the 'delete' (region) button
        """
        if self.tab != 2:
            self.select_tab(2)
        for selected_region in self.selected_regions:
            region = self.region_set.get(selected_region['Region'])
            # Then we delete the regions in self.region_set
            self.region_set.remove(region.num)

        self.select_tab(2)
        # There is no more selected region
        self.clear_selected_regions()

    # ==================== TAB 3 ==================== #

    def substitute_clicked(self, widget, event, data):
        region = self.region_set.get(self.selected_regions[0]['Region'])
        self.selected_sub_model = []
        if region is not None:
            # We update the substitution table once to show the name of the region
            self.substitution_model_training = True
            # show tab 3 (and update)
            self.select_tab(3)
            region.train_substitution_models(task_type=self.problem_category)

            self.substitution_model_training = False
            # We update the substitution table a second time to show the results
            self.update_substitution_table(region)

    def update_subtitution_prefix(self, region):
        # Region prefix text
        get_widget(self.widget, "450000").class_ = "mr-2 black--text" if region else "mr-2 grey--text"
        # v.Chip
        get_widget(self.widget, "450001").color = region.color if region else "grey"
        get_widget(self.widget, "450001").children = [str(region.num)] if region else ["-"]

    def update_subtitution_progress_bar(self):
        prog_circular = get_widget(self.widget, "450110")
        if self.substitution_model_training:
            prog_circular.disabled = False
            prog_circular.color = "blue"
            prog_circular.indeterminate = True
        else:
            prog_circular.disabled = True
            prog_circular.color = "grey"
            prog_circular.indeterminate = False

    def update_substitution_title(self, region: ModelRegion):
        title = get_widget(self.widget, "450002")
        title.tag = "h3"
        table = get_widget(self.widget, "45001")  # subModel table
        if self.substitution_model_training:
            # We tell to wait ...
            title.class_ = "ml-2 grey--text italic "
            title.children = [f"Sub-models are being evaluated ..."]
            # We clear items int the SubModelTable
            table.items = []
        elif not region:  # no region provided
            title.class_ = "ml-2 grey--text italic "
            title.children = [f"No region selected for substitution"]
            table.items = []
        elif region.num_points() < config.MIN_POINTS_NUMBER:  # region is too small
            title.class_ = "ml-2 red--text"
            title.children = [" Region too small for substitution !"]
            table.items = []
        elif len(region.perfs) == 0:  # model not trained
            title.class_ = "ml-2 red--text"
            title.children = [" click on substitute button to train substitution models"]
            table.items = []
        else:
            # We have results
            title.class_ = "ml-2 black--text"
            title.children = [
                f"{region.name}, "
                f"{region.num_points()} points, {100 * region.dataset_cov():.1f}% of the dataset"
            ]

            def series_to_str(series: pd.Series) -> str:
                return series.apply(lambda x: f"{x:.2f}")

            perfs = region.perfs.copy()
            for col in perfs.columns:
                if col != 'delta_color':
                    perfs[col] = series_to_str(perfs[col])
            perfs = perfs.reset_index().rename(columns={"index": "Sub-model"})
            headers = [
                {
                    "text": column,
                    "sortable": False,
                    "value": column,
                }
                for column in perfs.drop('delta_color', axis=1).columns
            ]
            table.headers = headers
            table.items = perfs.to_dict("records")
            if region.interpretable_models.selected_model:
                # we set to selected model if any
                table.selected = [
                    {'item': {'Sub-model': region.interpretable_models.selected_model}, 'value': True}]
            else:
                # clear selection if new region:
                table.selected = []

    def update_substitution_table(self, region: ModelRegion | None):
        """
        Called twice to update table
        """
        # set region to called region
        self.substitute_region = region

        self.update_subtitution_prefix(region)
        self.update_subtitution_progress_bar()
        self.update_substitution_title(region)

    def sub_model_selected_callback(self, data):
        is_selected = bool(data["value"])
        # We use this GUI attribute to store the selected sub-model
        self.selected_sub_model = [data['item']]
        get_widget(self.widget, "4501000").disabled = not is_selected
        if is_selected:
            region = self.region_set.get(self.selected_regions[0]['Region'])
            self.model_explorer.update_selected_model(region.get_model(data['item']['Sub-model']))

    def validate_sub_model(self, *args):
        # We get the sub-model data from the SubModelTable:
        # get_widget(self.widget,"45001").items[self.validated_sub_model]

        get_widget(self.widget, "4501000").disabled = True

        # We udpate the region
        region = self.region_set.get(self.selected_regions[0]['Region'])
        region.select_model(self.selected_sub_model[0]['Sub-model'])
        region.validate()
        # empty selected region
        self.selected_regions = []
        self.selected_sub_model = []
        # Show tab 2
        self.select_tab(2)
