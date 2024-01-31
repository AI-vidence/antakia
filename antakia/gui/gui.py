from __future__ import annotations
import pandas as pd

import ipyvuetify as v
import IPython.display
import ipywidgets as widgets

from antakia.data_handler.projected_values import ProjectedValues
from antakia.data_handler.region import ModelRegionSet, ModelRegion
from antakia.gui.explanation_values import ExplanationValues
from antakia.utils.long_task import LongTask
from antakia.compute.explanation.explanation_method import ExplanationMethod
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.auto_cluster.auto_cluster import AutoCluster
from antakia.compute.skope_rule.skope_rule import skope_rules
import antakia.config as config
from antakia.data_handler.rules import Rule

from antakia.gui.widgets import get_widget, change_widget, splash_widget, app_widget
from antakia.gui.highdimexplorer import HighDimExplorer
from antakia.gui.ruleswidget import RulesWidget

import copy

import logging
from antakia.utils.logging import conf_logger
from antakia.utils.utils import boolean_mask
from antakia.utils.variable import DataVariables

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
            X_exp: pd.DataFrame | None = None,
            score: callable | str = "mse",
    ):
        self.tab = 1
        self.X = X
        self.y = y
        self._y_pred = None
        self.model = model
        self.variables: DataVariables = variables
        self.score = score
        if X_exp is not None:
            if X.reindex(X_exp.index).iloc[:, 0].isna().sum() != X.iloc[:, 0].isna().sum():
                raise IndexError('X and X_exp must share the same index')
        if X.reindex(y.index).iloc[:, 0].isna().sum() != X.iloc[:, 0].isna().sum():
            raise IndexError('X and y must share the same index')
        # Init value space widgets
        init_dim = config.DEFAULT_DIMENSION
        # first hde
        self.vs_hde = HighDimExplorer(
            ProjectedValues(self.X, self.y),
            init_dim,
            int(config.INIT_FIG_WIDTH / 2),
            self.selection_changed,
            'VS',
        )
        # then rules
        self.vs_rules_wgt = RulesWidget(self.X, self.y, self.variables, True, self.new_rules_defined)

        # init Explanation space
        # first explanation getter/compute
        self.exp_values = ExplanationValues(self.X, self.y, self.model, self.explanation_changed_callback, X_exp)
        # then hde
        self.es_hde = HighDimExplorer(
            self.exp_values.current_pv,
            init_dim,
            int(config.INIT_FIG_WIDTH / 2),
            self.selection_changed,
            'ES'
        )
        # finally rules
        self.es_rules_wgt = RulesWidget(X_exp, self.y, self.variables, False)

        # init selection to all points

        # We set empty rules for now :
        self.vs_rules_wgt.disable()
        self.es_rules_wgt.disable()

        # init tabs
        self.region_num_for_validated_rules = None  # tab 1 : number of the region created when validating rules
        self.region_set = ModelRegionSet(self.X, self.y, self.model, self.score)
        self.selected_region_num = None  # tab 2 :  num of the region selected for substitution
        self.validated_sub_model_dict = None  # tab 3 : num of the sub-model validated for the region
        self.substitution_model_training = False  # tab 3 : training flag

        # UI rules :
        # We disable the selection datatable at startup (bottom of tab 1)
        get_widget(app_widget, "4320").disabled = True

        # We add both widgets to the current notebook cell and hide them
        IPython.display.display(splash_widget, app_widget)
        splash_widget.hide()
        app_widget.hide()

    @property
    def y_pred(self):
        if self._y_pred is None:
            self._y_pred = pd.Series(self.model.predict(self.X), index=self.X.index)
        return self._y_pred

    def show_splash_screen(self):
        """Displays the splash screen and updates it during the first computations."""
        get_widget(splash_widget, "110").color = "light blue"
        get_widget(splash_widget, "110").v_model = 0
        get_widget(splash_widget, "210").color = "light blue"
        get_widget(splash_widget, "210").v_model = 0
        splash_widget.show()

        # We trigger VS proj computation :
        get_widget(
            splash_widget, "220"
        ).v_model = f"{DimReducMethod.dimreduc_method_as_str(config.DEFAULT_PROJECTION)} on {self.X.shape} x 4"
        self.vs_hde.initialize(progress_callback=self.update_splash_screen)
        # self.vs_hde.compute_projs(False, self.update_splash_screen)

        # We trigger ES explain computation if needed :
        if not self.exp_values.has_user_exp:  # No imported explanation values
            msg = f"Computing {ExplanationMethod.explain_method_as_str(config.DEFAULT_EXPLANATION_METHOD)} on {self.X.shape}"
        else:
            msg = f"Imported explained values {self.X.shape}"
        self.exp_values.initialize(self.update_splash_screen)
        get_widget(splash_widget, "120").v_model = msg

        # THen we trigger ES proj computation :
        # self.es_hde.compute_projs(False, self.update_splash_screen)
        self.es_hde.initialize(progress_callback=self.update_splash_screen)

        self.selection_changed(None, boolean_mask(self.X, True))

        self.setup_app()

    def update_splash_screen(self, caller: LongTask, progress: int, duration: float):
        """
        Updates progress bars of the splash screen
        """
        # We select the proper progress bar :
        if isinstance(caller, ExplanationMethod):
            # It's an explanation
            progress_linear = get_widget(splash_widget, "110")
            number = 1
        else:  # It's a projection
            progress_linear = get_widget(splash_widget, "210")
            number = 2  # (VS/ES) x (2D/3D)

        if progress_linear.color == "light blue":
            progress_linear.color = "blue"
            progress_linear.v_model = 0

        if isinstance(caller, ExplanationMethod):
            progress_linear.v_model = round(progress / number)
            logger.debug(f"explain progress : {progress_linear.v_model} %")
        else:
            progress_linear.v_model += round(progress / number)
            logger.debug(f"proj progress : {progress_linear.v_model} %")

        if progress_linear.v_model == 100:
            progress_linear.color = "light blue"

        # We check if we're finished
        if get_widget(splash_widget, "110").v_model == 100 and get_widget(splash_widget, "210").v_model == 100:
            # We hide the splash screen and show the app :
            splash_widget.hide()
            app_widget.show()

    def explanation_changed_callback(self, progress_callback=None):
        self.es_hde.update_pv(self.exp_values.current_pv, progress_callback)
        self.es_rules_wgt.update_X(self.exp_values.current_pv.X)

    def disable_hde(self, disable):
        self.vs_hde.disable_widgets(disable)
        self.exp_values.disable_selection(disable)
        self.es_hde.disable_widgets(disable)

    def set_dimension(self, dim):
        self.vs_hde.set_dimension(dim)
        self.es_hde.set_dimension(dim)

    def selection_changed(self, caller: HighDimExplorer, new_selection_mask: pd.Series):
        """Called when the selection of one HighDimExplorer changes"""

        # UI rules :
        # If new selection (empty or not) : if exist, we remove any 'pending rule'

        if new_selection_mask.all():
            # Selection is empty
            # UI rules :
            # We disable the 'Find-rules' button
            get_widget(app_widget, "43010").disabled = True
            # We disable 'undo' and 'validate rules' buttons
            get_widget(app_widget, "4302").disabled = True
            get_widget(app_widget, "43030").disabled = True

            # We enable both HDEs (proj select, explain select etc.)
            self.vs_hde.display_rules()
            self.es_hde.display_rules()
            self.disable_hde(False)

            # we reset rules_widgets
            self.vs_rules_wgt.disable()
            self.es_rules_wgt.disable()
            self.es_rules_wgt.reset_widget()
            self.vs_rules_wgt.reset_widget()

            # We disable the selection datatable :
            get_widget(app_widget, "4320").disabled = True

        else:
            # Selection is not empty anymore or changes
            # UI rules :
            # We enable the 'Find-rules' button
            get_widget(app_widget, "43010").disabled = False
            # We disable HDEs (proj select, explain select etc.)
            self.disable_hde(True)
            # We show and fill the selection datatable :
            get_widget(app_widget, "4320").disabled = False
            X_rounded = copy.copy((self.X.loc[new_selection_mask])).round(3)
            change_widget(
                app_widget,
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
            other_hde = self.es_hde if caller == self.vs_hde else self.vs_hde
            other_hde.set_selection(self.selection_mask)

        # We update the selection status :
        if not self.selection_mask.all():
            selection_status_str_1 = f"{self.selection_mask.sum()} point selected"
            selection_status_str_2 = f"{100 * self.selection_mask.mean():.2f}% of the  dataset"
        else:
            selection_status_str_1 = f"0 point selected"
            selection_status_str_2 = f"0% of the  dataset"
        change_widget(app_widget, "4300000", selection_status_str_1)
        change_widget(app_widget, "430010", selection_status_str_2)

    def fig_size_changed(self, widget, event, data):
        """Called when the figureSizeSlider changed"""
        self.vs_hde.fig_width = self.es_hde.fig_width = round(widget.v_model / 2)
        self.vs_hde.update_fig_size()
        self.es_hde.update_fig_size()

    def new_rules_defined(self, rules_widget: RulesWidget, df_mask: pd.Series, skr: bool = False):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        # We make sure we're in 2D :
        get_widget(app_widget, "100").v_model == 2  # Switch button
        # TODO : pourquoi on passe en dim 2 ici ?
        self.set_dimension(2)

        # We sent to the proper HDE the rules_indexes to render :
        self.vs_hde.display_rules(df_mask)
        self.es_hde.display_rules(df_mask)

        # sync selection between rules_widgets
        if rules_widget == self.vs_rules_wgt:
            self.es_rules_wgt.update_from_mask(df_mask, [])
        else:
            self.vs_rules_wgt.update_from_mask(df_mask, [])

        # We disable the 'undo' button if RsW has less than 2 rules
        get_widget(app_widget, "4302").disabled = rules_widget.rules_num <= 1
        # We disable the 'validate rules' button if RsW has less than 1 rule
        get_widget(app_widget, "43030").disabled = rules_widget.rules_num <= 0

    def setup_app(self):
        """
        Inits and wires the app_widget, and implements UI logic
        """
        # =================== AppBar ===================

        # ------------------Figure size -----------------

        # We wire the input event on the figureSizeSlider (050100)
        get_widget(app_widget, "03000").on_event("input", self.fig_size_changed)
        # We set the init value to default :
        get_widget(app_widget, "03000").v_model = config.INIT_FIG_WIDTH

        # -------------- Dimension Switch --------------

        get_widget(app_widget, "100").v_model == config.DEFAULT_DIMENSION
        get_widget(app_widget, "100").on_event("change", self.switch_dimension)

        # -------------- ColorChoiceBtnToggle ------------

        # Set "change" event on the Button Toggle used to chose color
        get_widget(app_widget, "11").on_event("change", self.change_color)

        # ============== HighDimExplorers ===============

        # We attach each HighDimExplorers component to the app_graph:
        change_widget(app_widget, "201", self.vs_hde.figure_container),
        change_widget(app_widget, "14", self.vs_hde.get_projection_select())
        change_widget(app_widget, "16", self.vs_hde.get_projection_prog_circ())
        change_widget(app_widget, "211", self.es_hde.figure_container)
        change_widget(app_widget, "17", self.es_hde.get_projection_select())
        change_widget(app_widget, "19", self.es_hde.get_projection_prog_circ())
        change_widget(app_widget, "12", self.exp_values.get_explanation_select())
        change_widget(app_widget, "13", self.exp_values.get_compute_menu())

        # ================ Tab 1 Selection ================

        # We wire the click event on 'Tab 1'
        get_widget(app_widget, "40").on_event("click", self.select_tab_front(1))

        # We add our 2 RulesWidgets to the GUI :
        change_widget(app_widget, "4310", self.vs_rules_wgt.root_widget)
        change_widget(app_widget, "4311", self.es_rules_wgt.root_widget)

        # We wire the click event on the 'Find-rules' button
        get_widget(app_widget, "43010").on_event("click", self.compute_skope_rules)
        # UI rules :
        # The Find-rules button is disabled at startup. It's only enabled when a selection occurs - when the selection is empty, it's disabled again
        get_widget(app_widget, "43010").disabled = True

        # We wire the ckick event on the 'Undo' button
        get_widget(app_widget, "4302").on_event("click", self.undo_rules)
        # At start the button is disabled
        get_widget(app_widget, "4302").disabled = True

        # Its enabled when rules graphs have been updated with rules
        # We wire the click event on the 'Valildate rules' button
        get_widget(app_widget, "43030").on_event("click", self.validate_rules)

        # UI rules :
        # The 'validate rules' button is disabled at startup
        get_widget(app_widget, "43030").disabled = True

        # It's enabled when a SKR rules has been found and is disabled when the selection gets empty
        # or when validated is pressed

        # ================ Tab 2 : regions ===============
        # We wire the click event on 'Tab 2'
        get_widget(app_widget, "41").on_event("click", self.select_tab_front(2))

        get_widget(app_widget, "440010").set_callback(self.region_selected)

        # We wire events on the 'substitute' button:
        get_widget(app_widget, "4401000").on_event("click", self.substitute_clicked)
        # button is disabled by default
        get_widget(app_widget, "4401000").disabled = True

        # We wire events on the 'subdivide' button:
        get_widget(app_widget, "440110").on_event("click", self.subdivide_region_clicked)
        # button is disabled by default
        get_widget(app_widget, "440110").disabled = True

        # We wire events on the 'delete' button:
        get_widget(app_widget, "440120").on_event("click", self.delete_region_clicked)
        # The 'delete' button is disabled at startup
        get_widget(app_widget, "440120").disabled = True

        # We wire events on the 'auto-cluster' button :
        get_widget(app_widget, "4402000").on_event("click", self.auto_cluster_clicked)

        # UI rules :
        # The 'auto-cluster' button is disabled at startup
        get_widget(app_widget, "4402000").disabled = True
        # Checkbox automatic number of cluster is set to True at startup
        get_widget(app_widget, "440211").v_model = True

        # We wire select events on this checkbox :
        get_widget(app_widget, "440211").on_event("change", self.checkbox_auto_cluster_clicked)

        def num_cluster_changed(widget, event, data):
            """
            Called when the user changes the number of clusters
            """
            # We enable the 'auto-cluster' button
            get_widget(app_widget, "4402000").disabled = False

        # We wire events on the num cluster Slider
        get_widget(app_widget, "4402100").on_event("change", num_cluster_changed)

        # UI rules : at startup, the slider is is disabled and the checkbox is checked
        get_widget(app_widget, "4402100").disabled = True

        self.update_region_table()
        # At startup, REGIONSET_TRACE is not visible

        # ============== Tab 3 : substitution ==================

        # We wire the click event on 'Tab 3'
        get_widget(app_widget, "42").on_event("click", self.select_tab_front(3))

        # UI rules :
        # At startup the validate sub-model btn is disabled :
        get_widget(app_widget, "450100").disabled = True

        # We wire a select event on the 'substitution table' :
        get_widget(app_widget, "45001").set_callback(self.sub_model_selected)

        # We wire a ckick event on the "validate sub-model" button :
        get_widget(app_widget, "450100").on_event("click", self.validate_sub_model)

        # We disable the Substitution table at startup :
        self.update_substitution_table(None)

        self.select_tab(1)

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

        self.vs_hde.set_color(color, 0)
        self.es_hde.set_color(color, 0)
        self.select_tab(0)

    def select_tab_front(self, tab):
        def call_fct(*args):
            self.select_tab(tab, front=True)

        return call_fct

    def select_tab(self, tab, front=False):
        if self.tab == 1:
            self.vs_hde._deselection_event()
            self.vs_hde.create_figure()
            self.es_hde._deselection_event()
            self.es_hde.create_figure()
        if tab == 2:
            self.update_region_table()
            self.vs_hde.display_regionset(self.region_set)
            self.es_hde.display_regionset(self.region_set)
        elif tab == 3:
            region = self.region_set.get(self.selected_region_num)
            self.update_substitution_table(region)
            if region is None:
                region = ModelRegion(self.X, self.y, self.model, score=self.score)
            self.vs_hde.display_region(region)
            self.es_hde.display_region(region)
        if not front:
            get_widget(app_widget, "4").v_model = tab - 1
        self.vs_hde.set_tab(tab)
        self.es_hde.set_tab(tab)
        self.tab = tab

    # ==================== TAB 1 ==================== #

    def compute_skope_rules(self, *args):
        if self.tab != 1:
            self.select_tab(1)
        # if clicked, selection can't be empty
        assert not self.selection_mask.all()
        # Let's disable the Skope button. It will be re-enabled if a new selection occurs
        get_widget(app_widget, "43010").disabled = True
        self.compute_skr_display(self.vs_hde, self.vs_rules_wgt)
        self.compute_skr_display(self.es_hde, self.es_rules_wgt)

    def compute_skr_display(self, hde, rules_widget):
        skr_rules_list, skr_score_dict = skope_rules(self.selection_mask, hde.current_X, self.variables)
        skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
        if len(skr_rules_list) > 0:  # SKR rules found
            # UI rules :
            # We enable the 'validate rule' button
            if self.vs_hde == hde:
                get_widget(app_widget, "43030").disabled = False
            # We enable RulesWidet and init it wit the rules
            rules_widget.enable()
            rules_widget.init_rules(skr_rules_list, skr_score_dict, self.selection_mask)
        else:
            # No skr found
            rules_widget.show_msg("No rules found", "red--text")
            if self.vs_hde == hde:
                # we disable validation if no rules are found in value space
                get_widget(app_widget, "43030").disabled = True

    def undo_rules(self, *args):
        if self.tab != 1:
            self.select_tab(1)
        if self.vs_rules_wgt.rules_num > 0:
            self.vs_rules_wgt.undo()
            if self.vs_rules_wgt.rules_num == 1:
                # We disable the 'undo' button
                get_widget(app_widget, "4302").disabled = True
        else:
            # TODO : pourquoi on annule d'abord le VS puis l'ES?
            self.es_rules_wgt.undo()
            if self.es_rules_wgt.rules_num == 1:
                # We disable the 'undo' button
                get_widget(app_widget, "4302").disabled = True

    def validate_rules(self, *args):
        if self.tab != 1:
            self.select_tab(1)

        rules_list = self.vs_rules_wgt.current_rules_list
        # We add them to our region_set
        if len(rules_list) == 0:
            self.es_rules_wgt.reset_widget()
            self.vs_rules_wgt.reset_widget()
            self.vs_rules_wgt.show_msg("No rules found on Value space cannot validate region", "red--text")
            return

        region = self.region_set.add_region(rules=rules_list)
        self.region_num_for_validated_rules = region.num
        # lock rule
        region.validate()

        # And update the rules table (tab 2)
        # UI rules :
        # We clear selection
        # We clear the RulesWidget
        self.vs_rules_wgt.reset_widget()
        self.es_rules_wgt.reset_widget()
        # We force tab 2
        self.select_tab(2)

        # We disable the 'undo' button
        get_widget(app_widget, "4302").disabled = True
        # We disable the 'validate rules' button
        get_widget(app_widget, "43030").disabled = True

    # ==================== TAB 2 ==================== #

    def update_region_table(self):
        """
        Called to empty / fill the RegionDataTable and refresh plots
        """
        self.region_set.sort(by='size', ascending=False)
        temp_items = self.region_set.to_dict()

        # We populate the ColorTable :
        get_widget(app_widget, "440010").items = temp_items

        region_stats = self.region_set.stats()
        str_stats = [
            f"{region_stats['regions']} {'regions' if region_stats['regions'] > 1 else 'region'}",
            f"{region_stats['points']} points",
            f"{region_stats['coverage']}% of the dataset",
            f"{region_stats['delta_score']:.2f} subst score"
        ]
        get_widget(app_widget, "44002").children = [
            ', '.join(str_stats)
        ]
        get_widget(app_widget, "4402000").disabled = False

    def checkbox_auto_cluster_clicked(self, widget, event, data):
        """
        Called when the user clicks on the 'auto-cluster' checkbox
        """
        if self.tab != 2:
            self.select_tab(2)
        # In any case, we enable the auto-cluster button
        get_widget(app_widget, "4402000").disabled = False

        # We reveive either True or {}
        if data != True:
            data = False

        # IF true, we disable the Slider
        get_widget(app_widget, "4402100").disabled = data

    def auto_cluster_clicked(self, widget, event, data):
        """
        Called when the user clicks on the 'auto-cluster' button
        """
        get_widget(app_widget, "4402000").disabled = True
        if self.tab != 2:
            self.select_tab(2)
        if self.region_set.stats()["coverage"] > 80:
            # UI rules :
            # region_set coverage is > 80% : we need to clear it to do another auto-cluster
            self.region_set.clear_unvalidated()

        # We disable the AC button. Il will be re-enabled when the AC progress is 100%

        # We assemble indices ot all existing regions :
        region_set_mask = self.region_set.mask
        not_rules_indexes_list = ~region_set_mask
        # We call the auto_cluster with remaing X and explained(X) :
        if get_widget(app_widget, "440211").v_model:
            cluster_num = "auto"
        else:
            cluster_num = get_widget(app_widget, "4402100").v_model - len(self.region_set)

        self.compute_auto_cluster(not_rules_indexes_list, cluster_num)

        # We re-enable the button
        get_widget(app_widget, "4402000").disabled = False
        self.select_tab(2)

    def compute_auto_cluster(self, not_rules_indexes_list, cluster_num='auto'):
        if len(not_rules_indexes_list) > config.MIN_POINTS_NUMBER:
            vs_proj_3d_df = self.vs_hde.get_current_X_proj(3, False, progress_callback=self.get_ac_progress_update(1))
            es_proj_3d_df = self.es_hde.get_current_X_proj(3, False, progress_callback=self.get_ac_progress_update(2))

            ac = AutoCluster(self.X, self.get_ac_progress_update(3))

            found_regions = ac.compute(
                vs_proj_3d_df.loc[not_rules_indexes_list],
                es_proj_3d_df.loc[not_rules_indexes_list],
                # We send 'auto' or we read the number of clusters from the Slider
                cluster_num,
            )  # type: ignore
            self.region_set.extend(found_regions)
        else:
            print('not enough points to cluster')

    def get_ac_progress_update(self, step):
        def update_ac_progress_bar(caller, progress: float, duration: float):
            """
            Called by the AutoCluster to update the progress bar
            """
            total_steps = 3
            progress = ((step - 1) * 100 + progress) / total_steps
            get_widget(app_widget, "440212").v_model = progress

        return update_ac_progress_bar

    def disable_buttons(self, region: ModelRegion):
        if region is None:
            # substitute
            get_widget(app_widget, "4401000").disabled = True
            # subdivide
            get_widget(app_widget, "440110").disabled = True
            # delete
            get_widget(app_widget, "440120").disabled = True
        else:
            # substitute
            get_widget(app_widget, "4401000").disabled = region is None
            # subdivide
            disable_sub = bool(region.num_points() <= config.MIN_POINTS_NUMBER)
            get_widget(app_widget, "440110").disabled = disable_sub
            # delete
            get_widget(app_widget, "440120").disabled = region is None

    def region_selected(self, data):
        if self.tab != 2:
            self.select_tab(2)
        is_selected = data["value"]
        if is_selected is None:
            self.selected_region_num = None
            self.disable_buttons(None)
        else:
            # We use this GUI attribute to store the selected region
            # TODO : read the selected region from the ColorTable
            self.selected_region_num = data["item"]["Region"] if is_selected else None

            # UI rules :
            # If selected, we enable the 'substitute' and 'delete' buttons and vice-versa
            self.disable_buttons(self.region_set.get(self.selected_region_num))

    def clear_selected_regions(self):
        self.selected_region_num = None
        self.disable_buttons(None)

    def subdivide_region_clicked(self, widget, event, data):
        """
        Called when the user clicks on the 'delete' (region) button
        """
        if self.tab != 2:
            self.select_tab(2)
        # we recover the region to sudivide
        region = self.region_set.get(self.selected_region_num)
        if region.num_points() > config.MIN_POINTS_NUMBER:
            # Then we delete the region in self.region_set
            self.region_set.remove(self.selected_region_num)
            # we compute the subregions and add them to the region set
            self.compute_auto_cluster(region.mask)
            # UI rules : if we deleted a region coming from the Skope rules, we re-enable the Skope button
            if self.selected_region_num == self.region_num_for_validated_rules:
                get_widget(app_widget, "43010").disabled = False

        self.select_tab(2)
        # There is no more selected region
        self.clear_selected_regions()

    def delete_region_clicked(self, widget, event, data):
        """
        Called when the user clicks on the 'delete' (region) button
        """
        if self.tab != 2:
            self.select_tab(2)
        # Then we delete the regions in self.region_set
        self.region_set.remove(self.selected_region_num)
        # UI rules : if we deleted a region comming from the Skope rules, we re-enable the Skope button
        if self.selected_region_num == self.region_num_for_validated_rules:
            get_widget(app_widget, "43010").disabled = False

        self.select_tab(2)
        # There is no more selected region
        self.clear_selected_regions()

    # ==================== TAB 3 ==================== #

    def substitute_clicked(self, widget, event, data):
        assert self.selected_region_num is not None
        region = self.region_set.get(self.selected_region_num)
        if region is not None:
            # We update the substitution table once to show the name of the region
            self.substitution_model_training = True
            self.select_tab(3)
            region.train_subtitution_models()

            self.substitution_model_training = False
            # We update the substitution table a second time to show the results
            self.update_substitution_table(region)

    def update_substitution_table(self, region: ModelRegion):
        """
        Called twice to update table
        """
        disable = region is None
        # Region v.HTML
        get_widget(app_widget, "450000").class_ = "mr-2 black--text" if not disable else "mr-2 grey--text"
        # v.Chip
        get_widget(app_widget, "450001").color = region.color if not disable else "grey"
        get_widget(app_widget, "450001").children = [str(region.num)] if not disable else ["-"]

        vHtml = get_widget(app_widget, "450002")
        prog_circular = get_widget(app_widget, "45011")

        if not disable:
            # We're enabled
            if self.substitution_model_training:
                # We tell to wait ...
                vHtml.class_ = "ml-2 grey--text italic "
                vHtml.tag = "h3"
                vHtml.children = [f"Sub-models are being evaluated ..."]
                prog_circular.disabled = False
                prog_circular.color = "blue"
                prog_circular.indeterminate = True
                # We clear items int the SubModelTable
                get_widget(app_widget, "45001").items = []
            elif len(region.perfs) == 0:
                # We have no results
                vHtml.class_ = "ml-2 red--text"
                vHtml.tag = "h3"
                vHtml.children = [" Region too small for substitution !"]
                get_widget(app_widget, "45001").items = []
                # We stop the progress bar
                prog_circular.disabled = True
                prog_circular.color = "grey"
                prog_circular.indeterminate = False
            else:
                # We have results
                vHtml.class_ = "ml-2 black--text"
                vHtml.tag = "h3"
                vHtml.children = [
                    f"{Rule.multi_rules_to_string(region.rules) if region.rules is not None else 'auto-cluster'}, "
                    f"{region.num_points()} points, {100 * region.dataset_cov():.1f}% of the dataset"
                ]

                # We stop the progress bar
                prog_circular.disabled = True
                prog_circular.color = "grey"
                prog_circular.indeterminate = False

                # TODO : format cells in SubModelTable
                def series_to_str(series: pd.Series) -> str:
                    return series.apply(lambda x: f"{x:.2f}")

                perfs = region.perfs.copy()
                for col in perfs.columns:
                    perfs[col] = series_to_str(perfs[col])
                perfs = perfs.reset_index().rename(columns={"index": "Sub-model"})
                get_widget(app_widget, "45001").items = perfs.to_dict("records")
        else:
            # We're disabled
            vHtml.class_ = "ml-2 grey--text italic "
            vHtml.tag = "h3"
            vHtml.children = [f"No region selected for substitution"]
            prog_circular.disabled = True
            prog_circular.indeterminate = False
            # We clear items int the SubModelTable
            get_widget(app_widget, "45001").items = []

    def sub_model_selected(self, data):
        is_selected = data["value"]

        # We use this GUI attribute to store the selected sub-model
        # TODO : read the selected sub-model from the SubModelTable
        self.validated_sub_model_dict = data["item"] if is_selected else None
        get_widget(app_widget, "450100").disabled = True if not is_selected else False

    def validate_sub_model(self, widget, event, data):
        # We get the sub-model data from the SubModelTable:
        # get_widget(app_widget,"45001").items[self.validated_sub_model]

        get_widget(app_widget, "450100").disabled = True

        # We udpate the region
        region = self.region_set.get(self.selected_region_num)
        region.select_model(self.validated_sub_model_dict["Sub-model"])
        region.validate()

        # Show tab 2
        self.select_tab(2)
