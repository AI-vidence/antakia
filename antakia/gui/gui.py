from __future__ import annotations
import pandas as pd

import ipyvuetify as v
from IPython.display import display

from antakia.data_handler.region import Region, RegionSet
from antakia.utils.long_task import LongTask
from antakia.compute.explanation.explanation_method import ExplanationMethod
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.explanation.explanations import compute_explanations
from antakia.compute.auto_cluster.auto_cluster import AutoCluster
from antakia.compute.skope_rule.skope_rule import skope_rules
from antakia.compute.model_subtitution.model_interface import InterpretableModels
import antakia.config as config
from antakia.data_handler.rules import Rule

from antakia.gui.widgets import get_widget, change_widget, splash_widget, app_widget
from antakia.gui.highdimexplorer import HighDimExplorer
from antakia.data_handler.projected_values import ProjectedValues
from antakia.gui.ruleswidget import RulesWidget

import os
import copy

import logging
from antakia.utils.logging import conf_logger
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
        self.X = X
        self.y = y
        self.model = model
        self.y_pred = pd.Series(model.predict(X), index=X.index)
        self.variables: DataVariables = variables
        self.score = score

        # We create our VS HDE
        self.vs_hde = HighDimExplorer(
            self.X,
            self.y,
            config.DEFAULT_VS_PROJECTION,
            config.DEFAULT_VS_DIMENSION,
            int(config.INIT_FIG_WIDTH / 2),
            self.selection_changed,
        )  # type: ignore
        self.vs_rules_wgt = self.es_rules_wgt = None

        # We create our ES HDE :

        self.es_hde = HighDimExplorer(
            self.X,
            self.y,
            config.DEFAULT_VS_PROJECTION,
            config.DEFAULT_VS_DIMENSION,  # We use the same dimension as the VS HDE for now
            int(config.INIT_FIG_WIDTH / 2),
            self.selection_changed,
            self.new_eplanation_values_required,
            X_exp if X_exp is not None else pd.DataFrame(),  # passing an empty df (vs None) tells it's an ES HDE
        )

        self.vs_rules_wgt = RulesWidget(self.X, self.y, self.variables, True, self.new_rules_defined)
        self.es_rules_wgt = RulesWidget(self.es_hde.current_X, self.y, self.variables, False, self.new_rules_defined)
        # We set empty rules for now :
        self.vs_rules_wgt.disable()
        self.es_rules_wgt.disable()

        self.region_set = RegionSet(self.X)
        self.region_num_for_validated_rules = None  # tab 1 : number of the region created when validating rules
        self.selected_region_num = None  # tab 2 :  num of the region selected for substitution
        self.validated_sub_model_dict = None  # tab 3 : num of the sub-model validated for the region
        self.selection_mask = pd.Series([False] * len(X), index=X.index)

        # UI rules :
        # We disable the selection datatable at startup (bottom of tab 1)
        get_widget(app_widget, "4320").disabled = True

    def show_splash_screen(self):
        """Displays the splash screen and updates it during the first computations."""
        get_widget(splash_widget, "110").color = "light blue"
        get_widget(splash_widget, "110").v_model = 0
        get_widget(splash_widget, "210").color = "light blue"
        get_widget(splash_widget, "210").v_model = 0
        display(splash_widget)

        # We trigger VS proj computation :
        get_widget(
            splash_widget, "220"
        ).v_model = f"{DimReducMethod.dimreduc_method_as_str(config.DEFAULT_VS_PROJECTION)} on {self.X.shape} x 4"
        self.vs_hde.compute_projs(False, self.update_splash_screen)

        # We trigger ES explain computation if needed :
        if self.es_hde.pv_dict['imported_explanations'] is None:  # No imported explanation values
            # We compute default explanations :
            explain_method = config.DEFAULT_EXPLANATION_METHOD
            get_widget(
                splash_widget, "120"
            ).v_model = (
                f"Computing {ExplanationMethod.explain_method_as_str(config.DEFAULT_EXPLANATION_METHOD)} on {self.X.shape}"
            )
            self.es_hde.current_pv = 'computed_shap' if explain_method == ExplanationMethod.SHAP else 'computed_lime'
            self.es_hde.pv_dict[self.es_hde.current_pv] = ProjectedValues(
                self.new_eplanation_values_required(explain_method, self.update_splash_screen)
            )
            self.es_hde.update_explanation_select()
            self.es_hde.update_compute_menu()
        else:
            get_widget(
                splash_widget, "120"
            ).v_model = (
                f"Imported explained values {self.X.shape}"
            )

            # THen we trigger ES proj computation :
        self.es_hde.compute_projs(False, self.update_splash_screen)

        splash_widget.close()

        self.show_app()

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
            number = 4  # (VS/ES) x (2D/3D)

        if progress_linear.color == "light blue":
            progress_linear.color = "blue"
            progress_linear.v_model = 0

        if isinstance(caller, ExplanationMethod):
            progress_linear.v_model = round(progress / number)
        else:
            progress_linear.v_model += round(progress / number)

        if progress_linear.v_model == 100:
            progress_linear.color = "light blue"

    def update_substitution_table(self, region: Region, perfs: pd.DataFrame, disable: bool = False):
        """
        Called twice to update table
        """

        # Region v.HTML
        get_widget(app_widget, "450000").class_ = "mr-2 black--text" if not disable else "mr-2 grey--text"
        # v.Chip
        get_widget(app_widget, "450001").color = region.color if not disable else "grey"
        get_widget(app_widget, "450001").children = [str(region.num)] if not disable else ["-"]

        vHtml = get_widget(app_widget, "450002")
        prog_circular = get_widget(app_widget, "45011")

        if not disable:
            # We're enabled
            if perfs is not None and perfs.shape[0] == 0:
                # We tell to wait ...
                vHtml.class_ = "ml-2 grey--text italic "
                vHtml.tag = "h3"
                vHtml.children = [f"Sub-models are being evaluated ..."]
                prog_circular.disabled = False
                prog_circular.color = "blue"
                prog_circular.indeterminate = True
                # We clear items int the SubModelTable
                get_widget(app_widget, "45001").items = []

            elif perfs is not None and perfs.shape[0] > 0:
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

                for col in perfs.columns:
                    perfs[col] = series_to_str(perfs[col])
                perfs = perfs.reset_index().rename(columns={"index": "Sub-model"})
                perfs["Sub-model"] = perfs["Sub-model"].str.replace("_", " ").str.capitalize()
                get_widget(app_widget, "45001").items = perfs.to_dict("records")
            else:
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
            # We're disabled
            vHtml.class_ = "ml-2 grey--text italic "
            vHtml.tag = "h3"
            vHtml.children = [f"No region selected for substitution"]
            prog_circular.disabled = True
            prog_circular.indeterminate = False
            # We clear items int the SubModelTable
            get_widget(app_widget, "45001").items = []

    def update_regions_table(self):
        # TODO : lenteurs
        """
        Called to empty / fill the RegionDataTable with our >WXC V
        """
        temp_items = self.region_set.to_dict()

        # We populate the ColorTable :
        get_widget(app_widget, "4400100").items = temp_items
        # We populate the regions DataTable :
        get_widget(app_widget, "4400110").items = temp_items

        region_stats = self.region_set.stats()
        get_widget(app_widget, "44002").children = [
            f"{region_stats['regions']} {'regions' if region_stats['regions'] > 1 else 'region'}, {region_stats['points']} points, {region_stats['coverage']}% of the dataset"
        ]

        # It seems HDEs need to display regions each time we udpate the table :
        self.vs_hde.display_regions(self.region_set)
        self.es_hde.display_regions(self.region_set)

        # UI rules :
        # If regions coverage > 80%, we disable the 'auto-cluster' button
        get_widget(app_widget, "4402000").disabled = region_stats["coverage"] > 80

    def new_eplanation_values_required(self, explain_method: int, callback: callable = None) -> pd.DataFrame:
        """
        Called either by :
        - the splash screen
        - the ES HighDimExplorer (HDE) when the user wants to compute new explain values
        callback is a HDE function to update the progress linear
        """
        return compute_explanations(self.X, self.model, explain_method, callback)

    def selection_changed(self, caller: HighDimExplorer, new_selection_mask: pd.Series):
        """Called when the selection of one HighDimExplorer changes"""

        # UI rules :
        # Selection (empty or not) we remove any rule or region trace from HDEs
        self.vs_hde.display_rules(None)
        self.es_hde.display_rules(None)
        self.region_set.pop_last()
        self.update_regions_table()

        # Selection (empty or not) we reset both RulesWidgets
        self.vs_rules_wgt.disable()
        self.es_rules_wgt.disable()

        self.selection_mask = new_selection_mask

        if not new_selection_mask.any():
            # UI rules :
            # We disable the Skope button
            get_widget(app_widget, "43010").disabled = True
            # We disable 'undo' and 'validate rules' buttons
            get_widget(app_widget, "4302").disabled = True
            get_widget(app_widget, "43030").disabled = True
            # We enable HDEs (proj select, explain select etc.)
            self.vs_hde.disable_widgets(False)
            # We display tab 1
            get_widget(app_widget, "4").v_model = 0
            self.es_hde.disable_widgets(False)
            # We disable the selection datatable :
            get_widget(app_widget, "4320").disabled = True

        else:
            # Selection is not empty anymore / changes
            # UI rules :
            # We enable the SkopeButton
            get_widget(app_widget, "43010").disabled = False
            # We disable HDEs
            self.vs_hde.disable_widgets(True)
            self.es_hde.disable_widgets(True)
            # We show and fill the selection datatable :
            get_widget(app_widget, "4320").disabled = False
            sel_df = copy.copy((self.X.loc[new_selection_mask]))
            sel_df = round(sel_df, 3)
            change_widget(
                app_widget,
                "432010",
                v.DataTable(
                    v_model=[],
                    show_select=False,
                    headers=[{"text": column, "sortable": True, "value": column} for column in self.X.columns],
                    items=sel_df.to_dict("records"),
                    hide_default_footer=False,
                    disable_sort=False,
                ),
            )
        # We store the new selection
        self.selection_mask = new_selection_mask
        # We synchronize selection between the two HighDimExplorers
        other_hde = self.es_hde if caller == self.vs_hde else self.vs_hde
        other_hde.set_selection(self.selection_mask)

        # UI rules :
        # We update the selection status :
        selection_status_str_1 = f"{new_selection_mask.sum()} point selected"
        selection_status_str_2 = f"{100 * new_selection_mask.mean():.2f}% of the  dataset"
        change_widget(app_widget, "4300000", selection_status_str_1)
        change_widget(app_widget, "430010", selection_status_str_2)

    def fig_size_changed(self, widget, event, data):
        """Called when the figureSizeSlider changed"""
        self.vs_hde.fig_size = self.es_hde.fig_size = round(widget.v_model / 2)
        self.vs_hde.redraw()
        self.es_hde.redraw()

    def new_rules_defined(self, rules_widget: RulesWidget, df_mask: pd.Series, skr: bool = False):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        # We make sure we're in 2D :
        get_widget(app_widget, "100").v_model == 2  # Switch button
        # TODO : pourquoi on passe en dim 2 ici ?
        self.vs_hde.set_dimension(2)
        self.es_hde.set_dimension(2)

        # We sent to the proper HDE the rules_indexes to render :
        self.vs_hde.display_rules(df_mask) if rules_widget.is_value_space else self.es_hde.display_rules(df_mask)

        # We disable the 'undo' button if RsW has less than 2 rules
        get_widget(app_widget, "4302").disabled = rules_widget.rules_num < 1
        # We disable the 'validate rules' button if RsW has less than 1 rule
        get_widget(app_widget, "43030").disabled = rules_widget.rules_num < 0

    def show_app(self):
        # AppBar

        # --------- Two HighDimExplorers ----------

        # We attach each HighDimExplorers component to the app_graph:
        change_widget(app_widget, "201", self.vs_hde.container),
        change_widget(app_widget, "14", self.vs_hde.get_projection_select())
        change_widget(app_widget, "16", self.vs_hde.get_projection_prog_circ())
        change_widget(app_widget, "15", self.vs_hde.get_proj_params_menu())
        change_widget(app_widget, "211", self.es_hde.container)
        change_widget(app_widget, "17", self.es_hde.get_projection_select())
        change_widget(app_widget, "19", self.es_hde.get_projection_prog_circ())
        change_widget(app_widget, "18", self.es_hde.get_proj_params_menu())
        change_widget(app_widget, "12", self.es_hde.get_explanation_select())
        change_widget(app_widget, "13", self.es_hde.get_compute_menu())

        # ------------------ figure size ---------------

        # We wire the input event on the figureSizeSlider (050100)
        get_widget(app_widget, "03000").on_event("input", self.fig_size_changed)
        # We set the init value to default :
        get_widget(app_widget, "03000").v_model = int(os.environ.get("INIT_FIG_WIDTH"))

        # --------- ColorChoiceBtnToggle ------------
        def change_color(widget, event, data):
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

            self.vs_hde.redraw(color)
            self.es_hde.redraw(color)

        # Set "change" event on the Button Toggle used to chose color
        get_widget(app_widget, "11").on_event("change", change_color)

        # ------- Dimension Switch ----------
        def switch_dimension(widget, event, data):
            """
            Called when the switch changes.
            We call the HighDimExplorer to update its figure and, enventually,
            compute its proj
            """
            self.vs_hde.set_dimension(3 if data else 2)
            self.es_hde.set_dimension(3 if data else 2)

        get_widget(app_widget, "100").v_model == config.DEFAULT_VS_DIMENSION
        get_widget(app_widget, "100").on_event("change", switch_dimension)

        # ------------- Tab 1 Selection ----------------

        # We add our 2 RulesWidgets to the GUI :
        change_widget(app_widget, "4310", self.vs_rules_wgt.root_widget)
        change_widget(app_widget, "4311", self.es_rules_wgt.root_widget)

        def compute_skope_rules(widget, event, data):
            # if clicked, selection can't be empty
            assert self.selection_mask.any()
            # Let's disable the Skope button. It will be re-enabled if a new selection occurs
            get_widget(app_widget, "43010").disabled = True

            hde = self.vs_hde if self.vs_hde._has_lasso else self.es_hde
            rsw = self.vs_rules_wgt if self.vs_hde._has_lasso else self.es_rules_wgt
            skr_rules_list, skr_score_dict = skope_rules(self.selection_mask, hde.current_X, self.variables)
            skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
            if len(skr_rules_list) > 0:  # SKR rules found
                # UI rules :
                # We enable the 'validate rule' button
                get_widget(app_widget, "43030").disabled = False
                # We enable RulesWidet and init it wit the rules
                rsw.enable()
                rsw.init_rules(skr_rules_list, skr_score_dict, self.selection_mask)
            else:
                # No skr found
                rsw.show_msg("No rules found", "red--text")

        # We wire the click event on the 'Skope-rules' button
        get_widget(app_widget, "43010").on_event("click", compute_skope_rules)
        # UI rules :
        # The Skope rules button is disabled at startup
        # It's only enabled when a selection occurs - when the selection is empty, it's disabled again
        # Startup status :
        get_widget(app_widget, "43010").disabled = True

        def undo(widget, event, data):
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

        # We wire the ckick event on the 'Undo' button
        get_widget(app_widget, "4302").on_event("click", undo)
        # At start the button is disabled
        get_widget(app_widget, "4302").disabled = True

        # Its enabled when rules graphs have been updated with rules

        def validate_rules(widget, event, data):
            # We get the rules to validate
            # TODO : changer la façon de récupérer les règles
            if self.vs_rules_wgt.rules_num >= 0:
                rules_widget = self.vs_rules_wgt
                hde = self.vs_hde
            else:
                rules_widget = self.es_rules_wgt
                hde = self.es_hde

            rules_list = rules_widget.get_current_rules_list()
            self.region_num_for_validated_rules = self.region_set.get_max_num() + 1
            # We add them to our region_set

            region = self.region_set.add_region(rules=rules_list)
            # UI rules: we disable HDEs selection of we have one or more regions
            if len(self.region_set) > 0:
                self.vs_hde.disable_selection(True)
                self.es_hde.disable_selection(True)

            region.validate()

            # And update the rules table (tab 2)
            self.update_regions_table()
            # UI rules :
            # We clear selection
            self.es_hde._deselection_event(None, None, False)
            self.vs_hde._deselection_event(None, None, False)
            # We force tab 2
            get_widget(app_widget, "4").v_model = 1

            # We clear the RulesWidget
            rules_widget.disable()
            rules_widget.init_rules(None, None, None)
            # We disable the 'undo' button
            get_widget(app_widget, "4302").disabled = True
            # We disable the 'validate rules' button
            get_widget(app_widget, "43030").disabled = True
            # We clear HDEs 'rule traces'
            hde.display_rules(None)

        # We wire the click event on the 'Valildate rules' button
        get_widget(app_widget, "43030").on_event("click", validate_rules)

        # UI rules :
        # The 'validate rules' button is disabled at startup
        get_widget(app_widget, "43030").disabled = True

        # It's enabled when a SKR rules has been found and is disabled when the selection gets empty
        # or when validated is pressed

        # ------------- Tab 2 : regions -----------

        def region_selected(data):
            is_selected = data["value"]

            # We use this GUI attribute to store the selected region
            # TODO : read the selected region from the ColorTable
            self.selected_region_num = data["item"]["Region"] if is_selected else None

            # UI rules :
            # If selected, we enable the 'substitute' and 'delete' buttons and vice-versa
            get_widget(app_widget, "4401000").disabled = not is_selected
            get_widget(app_widget, "440110").disabled = not is_selected
            # If no region selected, we empty the substitution table:
            if not is_selected:
                self.update_substitution_table(None, None, True)

        get_widget(app_widget, "4400100").set_callback(region_selected)

        def substitute_clicked(widget, event, data):
            assert self.selected_region_num is not None
            get_widget(app_widget, "4").v_model = 2

            region = self.region_set.get(self.selected_region_num)

            # We update the substitution table once to show the name of the region
            self.update_substitution_table(region, pd.DataFrame())

            perfs = InterpretableModels(self.score).get_models_performance(
                self.model, self.X.loc[region.mask], self.y.loc[region.mask]
            )

            # We update the substitution table a second time to show the results
            self.update_substitution_table(region, perfs)

        # We wire events on the 'substitute' button:
        get_widget(app_widget, "4401000").on_event("click", substitute_clicked)

        # UI rules :
        # The 'substitute' button is disabled at startup; it'es enabled when a region is selected
        # and disabled if no region is selected or when substitute is pressed
        get_widget(app_widget, "4401000").disabled = True

        def delete_region_clicked(widget, event, data):
            """
            Called when the user clicks on the 'delete' (region) button
            """
            # Then we delete the regions in self.region_set
            self.region_set.remove(self.selected_region_num)
            # UI rules : if we deleted a region comming from the Skope rules, we re-enable the Skope button
            if self.selected_region_num == self.region_num_for_validated_rules:
                get_widget(app_widget, "43010").disabled = False

            self.update_regions_table()
            # There is no more selected region
            self.selected_region_num = None
            get_widget(app_widget, "440110").disabled = True
            get_widget(app_widget, "4401000").disabled = True

            # UI rules: if region table is emptye, HDE are selectable again
            if len(self.region_set) == 0:
                self.vs_hde.disable_selection(False)
                self.es_hde.disable_selection(False)

        # We wire events on the 'delete' button:
        get_widget(app_widget, "440110").on_event("click", delete_region_clicked)

        # UI rules :
        # The 'delete' button is disabled at startup
        get_widget(app_widget, "440110").disabled = True

        def auto_cluster_clicked(widget, event, data):
            """
            Called when the user clicks on the 'auto-cluster' button
            """
            if self.region_set.stats()["coverage"] > 80:
                # UI rules :
                # region_set coverage is > 80% : we need to clear it to do another auto-cluster
                self.region_set.clear_unvalidated()

            # We disable the AC button. Il will be re-enabled when the AC progress is 100%
            get_widget(app_widget, "4402000").disabled = True

            # We assemble indices ot all existing regions :
            rules_mask_list = self.region_set.get_masks()

            # We call the auto_cluster with remaing X and explained(X) :
            not_rules_indexes_list = pd.Series([True] * len(self.X), index=self.X.index)
            for mask in rules_mask_list:
                not_rules_indexes_list &= ~mask

            vs_proj_3d_df = self.vs_hde.get_current_X_proj(3, False)
            es_proj_3d_df = self.es_hde.get_current_X_proj(3, False)

            ac = AutoCluster(vs_proj_3d_df.loc[not_rules_indexes_list], update_ac_progress_bar)
            found_clusters = ac.compute(
                es_proj_3d_df.loc[not_rules_indexes_list],
                # We send 'auto' or we read the number of clusters from the Slider
                "auto" if get_widget(app_widget, "440211").v_model else get_widget(app_widget, "4402100").v_model,
            )  # type: ignore

            for cluster_num in found_clusters.unique():
                self.region_set.add_region(mask=(found_clusters == cluster_num))

            # UI rules: we disable HDEs selection of we have one or more regions
            if len(self.region_set) > 0:
                self.vs_hde.disable_selection(True)
                self.es_hde.disable_selection(True)

            self.update_regions_table()

        def update_ac_progress_bar(caller, progress: float, duration: float):
            """
            Called by the AutoCluster to update the progress bar
            """
            # #TODO Hack because we do not reveive 100% at the end
            if progress >= 96:
                progress = 100

            get_widget(app_widget, "440212").v_model = progress
            # We re-enable the button
            get_widget(app_widget, "4402000").disabled = progress == 100

        # We wire events on the 'auto-cluster' button :
        get_widget(app_widget, "4402000").on_event("click", auto_cluster_clicked)

        # UI rules :
        # The 'auto-cluster' button is disabled at startup
        get_widget(app_widget, "4402000").disabled = True
        # Checkbox automatic number of cluster is set to True at startup
        get_widget(app_widget, "440211").v_model = True

        def checkbox_auto_cluster_clicked(widget, event, data):
            """
            Called when the user clicks on the 'auto-cluster' checkbox
            """
            # In any case, we enable the auto-cluster button
            get_widget(app_widget, "4402000").disabled = False

            # We reveive either True or {}
            if data != True:
                data = False

            # IF true, we disable the Slider
            get_widget(app_widget, "4402100").disabled = data

        # We wire select events on this checkbox :
        get_widget(app_widget, "440211").on_event("change", checkbox_auto_cluster_clicked)

        def num_cluster_changed(widget, event, data):
            """
            Called when the user changes the number of clusters
            """
            # We enable the 'auto-cluster' button
            get_widget(app_widget, "4402000").disabled = False

        # We wire events on the num cluster Slider
        get_widget(app_widget, "4402100").on_event("change", num_cluster_changed)

        # UI rules : at startup, the slider is is disabled and the chckbox is checked
        get_widget(app_widget, "4402100").disabled = True

        self.update_regions_table()

        # ------------- Tab 3 : substitution -----------

        # UI rules :
        # At startup the validate sub-model btn is disabled :
        get_widget(app_widget, "450100").disabled = True

        def sub_model_selected(data):
            is_selected = data["value"]

            # We use this GUI attribute to store the selected sub-model
            # TODO : read the selected sub-model from the SubModelTable
            self.validated_sub_model_dict = data["item"] if is_selected else None
            get_widget(app_widget, "450100").disabled = True if not is_selected else False

        # We wire a select event on the 'substitution table' :
        get_widget(app_widget, "45001").set_callback(sub_model_selected)

        def validate_sub_model(widget, event, data):
            # We get the sub-model data from the SubModelTable:
            # get_widget(app_widget,"45001").items[self.validated_sub_model]

            score_val = 0
            if isinstance(self.score, str):
                score_name = self.score.upper()
            elif callable(self.score):
                score_name = self.score.__name__.upper()
            if float(self.validated_sub_model_dict[score_name]) > score_val:
                # TODO changer ça, c'est pas adapté
                score_val = float(self.validated_sub_model_dict[score_name])

            get_widget(app_widget, "450100").disabled = True

            # We udpate the region
            region = self.region_set.get(self.selected_region_num)
            region.set_model(self.validated_sub_model_dict["Sub-model"], f"{score_name} : {score_val:.2f}")

            # update_table
            self.update_regions_table()
            # Force tab 2 / disable tab 1
            get_widget(app_widget, "4").v_model = 1

        # We wire a ckick event on the "validate sub-model" button :
        get_widget(app_widget, "450100").on_event("click", validate_sub_model)

        # We disable the Substitution table at startup :
        self.update_substitution_table(None, None, True)

        display(app_widget)
