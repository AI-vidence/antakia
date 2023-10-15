import logging
import time

# Warnings imports
import warnings
from copy import deepcopy
from importlib.resources import files

import ipyvuetify as v
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display

from ipywidgets import Layout, widgets

# Internal imports
from antakia.data import DimReducMethod, ExplanationMethod, Variable
import antakia.config as config
import antakia.antakia as antakia
from antakia.compute import (
    auto_cluster
)
from antakia.data import (  
    ExplanationMethod,
    ProjectedValues
)
from antakia.selection import Selection
from antakia.utils import confLogger, overlapHandler
from antakia.gui_utils import (
    HighDimExplorer,
    RuleVariableRefiner,
    WidgetGraph,
    get_app_graph,
    get_splash_graph
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()

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
    X_list
    X_method_list
    y
    variables : a list of Variable
    selection : a list of points. Immplemented via a "Selection" object
    last_skr : a Selection object allowing to reinit the skope rules
    opacity : a list of two Pandas Series storing the opacity for each observation (VS and ES)
        The `Selection` object containing the current selection.
    out : Output Widget
    app_graph : a WidgetGraph of nested Widgets 
    vs_hde : HighDimExplorer for the VS and ES space
    ref1, ref2, ref3 : RuleVariableRefiner
    color : Pandas Series : color for each Y point
    fig_size : int
    labels_list : the list of labels for the ES HDE

    """

    def __init__(self, X_list, X_method_list : list, y: pd.Series, model):
        self.X_list = X_list
        self.X_method_list = X_method_list
        self.y = y
        self.variables = Variable.guess_variables(X_list[0])

        # For now, we assume atk has one or two Xs (if explainations are provided)
        assert X_list is not None and len(X_list) <=2
        assert X_method_list is not None and X_method_list[0] == ExplanationMethod.NONE

        self.out = widgets.Output()  # Ipwidgets output widget
        self.app_graph = get_app_graph()

        # We create our VS HDE :
        self.vs_hde = HighDimExplorer(
            "Values space",
            [X_list[0]],
            [None], # ignored
            [None], # ignored
            y,
            config.DEFAULT_VS_PROJECTION,
            config.DEFAULT_VS_DIMENSION,
            self.selection_changed,
            self.new_values_wanted)
        
        # We create our ES HDE :
        labels_list = ["SHAP imported", "SHAP computed", "LIME imported", "LIME computed"]
        # We put atk.Xs in the right slots
        values_list = [
            X_list[1] if (X_method_list[1] == ExplanationMethod.SHAP) else None,
            None,
            X_list[1] if (X_method_list[1] == ExplanationMethod.LIME) else None,
            None]
        is_computable_list = [False, True, False, True]
        self.es_hde = HighDimExplorer(
            "Explanations space", 
            values_list,
            labels_list,
            is_computable_list,
            y,
            config.DEFAULT_VS_PROJECTION, 
            config.DEFAULT_VS_DIMENSION, 
            self.selection_changed, 
            self.new_values_wanted)

        self.ref1 = RuleVariableRefiner(self.variables[0], self.update_skope_rules)
        self.ref2 = RuleVariableRefiner(self.variables[1], self.update_skope_rules)
        self.ref3 = RuleVariableRefiner(self.variables[2], self.update_skope_rules)
        self.fig_size = 200
        self.color = []
        self.selection = None
        self.opacity = pd.Series()

    def update_skope_rules(self):
        return None
        

    def new_values_wanted(self, new_values_index:int)-> pd.DataFrame:
        # Here we know ["SHAP imported", "SHAP computed", "LIME imported", "LIME computed"] were transmitted :
        desired_explain_method = None
        if new_values_index == 1:
            desired_explain_method = ExplanationMethod.SHAP
        elif new_values_index == 3:
            desired_explain_method = ExplanationMethod.LIME
        else:
            raise ValueError("new_values_index can only be 1 or 3")

        if len(self.atk.X_method_list) > 1:
            if self.atk.X_method_list[1] == desired_explain_method:
                raise ValueError("This explain method was already provided")

        return compute.compute_explanations(
            self.atk.X_list[0],
            self.atk.model, 
            desired_explain_method
            )


    def selection_changed(self, new_selection: Selection, side: int):
        """ Called when the selection of one HighDimExplorer changes
        """ 
        self.selection = new_selection # just created, no need to copy

        # We set opacity to 10% for the selection
        new_opacity_serie = pd.Series()
        for i in range(self._ds.get_length()):
                if i in self.selection.get_indexes:
                    new_opacity_serie.append(1)
                else:
                    new_opacity_serie.append(0.1)
        self._opacity[0 if side == config.VS else 1] = new_opacity_serie

        self.update_selection_table()
        self.redraw_graph(side)
        
        # We update the info Card
        selection_txt = "Current selection : \n0 pont selected (0% of the overall data)"
        if not self.selection.is_empty():
            selection_txt = \
            str(self.selection.size()) + \
            " points selected (" + \
            str( \
                round( \
                    self.selection.size() \
                    / self._ds.get_length() \
                    * 100, \
                    2, \
                ) \
            )
        self.app_graph.get_widget_at_address("304001").children[selection_txt]

    def __repr__(self):
        self.show()
        return ""

    def show(self):
        splash = get_splash_graph()
        # We display the splash screen
        display(splash.get_widget)

        splash.get_widget_at_address("120").v_model = "Values space ... "
        splash.get_widget_at_address("22").v_model = (
            "Default dimension reduction : "
            + DimReducMethod.dimreduc_method_as_str(config.DEFAULT_VS_PROJECTION)
            + " in "
            + str(config.DEFAULT_VS_DIMENSION) 
            + " dimensions ..."
        )

        # We remove the Splahs screen
        self.out.clear_output(wait=True)
        splash.hide()
        splash.destroy()

        # ------------- Main app  -----------------

        # We hide the 2 Progress Circular widgets (used to tell when a computation is in progress)
        self.app_graph.get_widget_at_address("12020").hide()
        self.app_graph.get_widget_at_address("12120").hide()

        # --------- Two HighDimExplorers ----------
        # We add each HighDimExplorer component to the app_graph :
        self.app_graph.change_widget("2001", self.vs_hde.get_figure_widget())
        self.app_graph.change_widget("1200", self.vs_hde.get_projection_select())

        self.app_graph.change_widget("2011", self.es_hde.get_figure_widget())
        self.app_graph.change_widget("1210", self.es_hde.get_projection_select())
        # Because self.es_hde _is_explain_explorer :
        self.app_graph.change_widget("113", self.es_hde.get_values_select())
        self.es_hde.get_explain_compute_menu()
        # --------- Set colorChoiceBtnToggle ------------
        def change_color(*args):
            """
                Called with the user clicks on the colorChoiceBtnToggle
                Allows change the color of the dots
            """
            # TODO : read the choice from the event, not from the GUI
            choice = self.app_graph.get_widget_at_address("111").v_model

            self.color = None
            if choice == "y":
                self.color = self.atk.y
            elif choice == "y^":
                self.color = self.atk.y_pred
            elif choice == "current selection":
                self.color = ["grey"] * len(self.atk.y)
                for i in range(len(self.selection.get_indexes())):
                    self.color[self.selection.get_indexes()[i]] = "blue"
            elif choice == "residual":
                self.color = self.atk.y - self.atk.y_pred
                self.color = [abs(i) for i in self.color]
            elif choice == "regions":
                self.color = [0] * len(self.atk.y)
                for i in range(len(self.atk.y)):
                    for j in range(len(self.regions)):
                        if i in self.regions[j].get_indexes():
                            self.color[i] = j + 1
            elif choice == "not selected":
                self.color = ["red"] * len(self.atk.X_list[0])
                if len(self.regions) > 0:
                    for i in range(len(self.atk.X_list[0])):
                        for j in range(len(self.regions)):
                            if i in self.regions[j].get_indexes():
                                self.color[i] = "grey"
            elif choice == "auto":
                self.color = None # TODO


        # Set "change" event on the Button Toggle used to chose color
        self.app_graph.get_widget_at_address("111").on_event("change", change_color)

        # ------- Dimension Switch ----------

        def switch_dimension(widget, event, data,):
            """
            Called when the switch changes.
            We call the HighDimExplorer to update its figure and, enventually,
            compute its proj
            """
            self.vs_hde.set_dimension(widget.v_model)
            self.es_hde.set_dimension(widget.v_model)
        self.app_graph.get_widget_at_address("102").on_event("change", switch_dimension)

        # ---- Tab 2 : refinement ----

        # ----------- Refiners ------------
        # Let's create 3 RuleVariableRefiner  :
        self.app_graph.change_widget("305010", self.ref1.get_widget_graph().get_widget())
        self.app_graph.change_widget("305011", self.ref2.get_widget_graph().get_widget())
        self.app_graph.change_widget("305012", self.ref3.get_widget_graph().get_widget())

        # -----------  reinitSkopeBtn Btn ------------
        def reinitSkopeRules(*b):
            # Why the rules should be saved ?
            # self.selection.setVSRules(self._save_rules)

            # If called, means the selection is back to a selection :
            self.selection.setType(Selection.SELECTION)

            # We reset the refiners
            self.reinit_skope_rules(None)

            # We udpate the scores for the submodels
            self.update_submodels_scores(None)

        # We wire the click event on the reinitSkopeBtn (3050001)
        self.app_graph.get_widget_at_address("3050001").on_event("click", reinitSkopeRules)

        # ---------- cluster and regions  ------------

        # Called when validating a tile to add it to the set of Regions
        def new_region_validated(*args):
            if len(args) == 0:
                pass
            elif self.selection in self.regions:
                # TODO : wez should have an Alert system
                print("AntakIA WARNING: this region is already in the set of Regions")
            else:
                # TODO : selection is selection. We should create another Selection
                self.selection.set_type(Selection.REGION)
                if self._model_index is None:
                    model_name = None
                    model_scores = [1, 1, 1]
                else:
                    model_name = config.get_default_submodels[self._model_index].__class__.__name__
                    # model_scores = self._subModelsScores[self._model_index]
                if self.selection.get_vs_rules() is None:
                    return

                # self.selection.setIndexesWithRules() # TODO not sure we have to call that

                new_indexes = deepcopy(self.selection.get_indexes())
                self.selection.get_submodel()["name"], self.selection.get_submodel()["score"] = (
                    model_name,
                    model_scores,
                )
                # We check that all the points of the new region belong only to it: we will modify the existing tiles
                self.regions = overlapHandler(self.regions, new_indexes)
                self.selection.set_new_indexes(new_indexes)

            self._regionColor = [0] * self._ds.__len__()
            if self.regions is not None:
                for i in range(len(self.regions_colors)):
                    for j in range(len(self.regions)):
                        if i in self.regions[j].get_indexes():
                            self._regionColor[i] = j + 1
                            break

            toute_somme = 0
            temp = []
            score_tot = 0
            score_tot_glob = 0
            autre_toute_somme = 0
            for i in range(len(self.regions)):
                if self.regions[i].get_submodel()["score"] is None:
                    temp.append(
                        [
                            i + 1,
                            len(self.regions[i]),
                            np.round(
                                len(self.regions[i])
                                / len(self.atk.y)
                                * 100,
                                2,
                            ),
                            "/",
                            "/",
                            "/",
                            "/",
                        ]
                    )
                else:
                    temp.append(
                        [
                            i + 1,
                            len(self.regions[i]),
                            np.round(
                                len(self.regions[i])
                                / len(self._ds.get_full_values())
                                * 100,
                                2,
                            ),
                            self.regions[i].get_submodel()["model"],
                            self.regions[i].get_submodel()["score"][0],
                            self.regions[i].get_submodel()["score"][1],
                            str(self.regions[i].get_submodel()["score"][2]) + "%",
                        ]
                    )
                    score_tot += self.regions[i].get_submodel()["score"][0] * len(
                        self.regions[i]
                    )
                    score_tot_glob += self.regions[i].get_submodel()["score"][1] * len(
                        self.regions[i]
                    )
                    autre_toute_somme += len(self.regions[i])
                toute_somme += len(self.regions[i])
            if autre_toute_somme == 0:
                score_tot = "/"
                score_tot_glob = "/"
                percent = "/"
            else:
                score_tot = round(score_tot / autre_toute_somme, 3)
                score_tot_glob = round(score_tot_glob / autre_toute_somme, 3)
                percent = (
                    str(round(100 * (score_tot_glob - score_tot) / score_tot_glob, 1))
                    + "%"
                )
            temp.append(
                [
                    "Total",
                    toute_somme,
                    np.round(toute_somme / len(self._ds) * 100, 2),
                    "/",
                    score_tot,
                    score_tot_glob,
                    percent,
                ]
            )
            pd.DataFrame(
                temp,
                columns=[
                    "Region #",
                    "Number of points",
                    "% of the dataset",
                    "Model",
                    "Score of the sub-model (MSE)",
                    "Score of the global model (MSE)",
                    "Gain in MSE",
                ],
            )
            # TODO what shall we do with this dataframe ???

        # We wire a click event on validateRegionBtn(307000) : note we created it in app_graph
        self.app_graph.get_widget_at_address("307000").on_event("click", new_region_validated)
        def cluster_number_changed(*b):
            # TODO : read the slider from the event, not from the GUI
            # We set clustersSliderTxt to the current clustersSlider value

            self.app_graph.change_widget("304230", "Number of clusters " + str(self.app_graph.get_widget_at_address("30422").v_model))

        # We wire the input event on the clustersSlider (30422)
        self.app_graph.get_widget_at_address("30422").on_event("input", cluster_number_changed)

        def cluster_check_changed(*b):
            # TODO : read the slider from the event, not from the GUI
            # TODO : what is this clusterCheck ?
            # clusterSlider(30422) visibility is linked to clusterCheck(30421)
            self.app_graph.get_widget_at_address("30422").disabled = self.app_graph.get_widget_at_address("30421").v_model

        self.app_graph.get_widget_at_address("30421").on_event("change", cluster_check_changed)

        # Let's create an empty / dummy cluster_results_table :
        new_df = pd.DataFrame([], columns=["Region #", "Number of points"])
        columns = [{"text": c, "sortable": True, "value": c} for c in new_df.columns]

        # We insert the cluster_results_table
        self.app_graph.change_widget("304410", v.Row(
            children=[
                v.Layout(
                    class_="flex-grow-0 flex-shrink-0",
                    children=[v.Btn(class_="d-none", elevation=0, disabled=True)],
                ),
                v.Layout( 
                    class_="flex-grow-1 flex-shrink-0",
                    children=[v.DataTable( # cluster_results_table
                        class_="w-100",
                        style_="width : 100%",
                        v_model=[],
                        show_select=False,
                        headers=columns,
                        explanationsMenuDict=new_df.to_dict("records"),
                        item_value="Region #",
                        item_key="Region #",
                        hide_default_footer=True,
                    )],
                ),
            ],
        )
        )

        # ------ Magic Buton ------
        def magic_btn_clicked(*b) -> int:
            # TODO : why ??
            # We update our loadingClustersProgLinear (3043)
            self.app_graph.get_widget_at_address("3043").class_ = "d-flex"

            # Depending on clusterCheck (30421) :
            if self.app_graph.get_widget_at_address("30421").v_model:
                self._auto_cluster_regions = auto_cluster(
                    self._ds.get_full_values(Dataset.SCALED),
                    self._xds.get_full_values(self._explanation_es[0]),
                    3,
                    True,
                )
            else:
                self._auto_cluster_regions = auto_cluster(
                    self._ds.get_full_values(Dataset.SCALED),
                    self._xds.get_full_values(self._explanation_es[0]),
                    # We read clusterSlider (30423) value :
                    self.app_graph.get_widget_at_address("30423").v_model,
                    False,
                )

            # TODO we'll have to change this when Selection will be returned
            self.color = self._auto_cluster_regions[1]

            self.redraw_both_graphs()

            # We set our regions accordingly
            self.regions = self._auto_cluster_regions # just created, no need to copy
            # We update the GUI tab 1(304) clusterResults(3044) (ie a v.Row)
            self.app_graph.change_widget("304234", self.datatable_from_Selectiones(self.regions))


            # TODO : understand
            # tabOneSelectionColumn.children = tabOneSelectionColumn.children[:-1] + [
            #     clusterResults
            # ]

            # colorChoiceBtnToggle(111) must be changed :
            self.app_graph.get_widget_at_address("30423").v_model = "auto"

            # We wire a change event on a button above the cluster_results_table DataTable (30420)
            self.app_graph.get_widget_at_address("30420").on_event(
                "change", cluster_results_table_changed
            )

            # We udpate loadingClustersProgLinear (3043)
            self.app_graph.get_widget_at_address("30440").class_ = "d-none"

            # TODO : check if we really expect an int from this function
            return len(self._auto_cluster_regions)
        
        # We wire the click event on the "Magic" findClusterBtn (30451)
        self.app_graph.get_widget_at_address("30451").on_event("click", magic_btn_clicked)

        def cluster_results_table_changed(widget, event, data):  # 1803
            """
            Called when a new magic clustering ahs been computed
            """
            # TODO : maybe this event should not be called only for "auto regions"

            # TODO : this has to do with the color and the current 
            # auto_cluser() implementation. It should return Selectiones
            labels = self._auto_cluster_regions[1]

            # TODO : I guess tabOneSelectionColumn.children[-1].children[0].children[0] was refering to the DataTable
            index = self.app_graph.get_widget_at_address("304420").v_model
            liste = [i for i, d in enumerate(labels) if d == float(index)]

            self.app_graph.get_widget_at_address("30423").v_model = "auto"
            # We call change_color by hand 
            # TODO : register change_color on the cluster_results_table instead 
            change_color()


        
        def magic_checkbox_changed(widget, event, data):
            textField = self.app_graph.get_widget_at_address("30453")
            if widget.v_model:
                textField.disabled = False
            else:
                textField.disabled = True

        # We wire a change event on magicCheckBox (or "demonstration mode" chekcbox)
        self.app_graph.get_widget_at_address("30452").on_event("change", magic_checkbox_changed)

        # TODO : strange it appears here no ?
        # We wire the ckick event on validateSkopeBtn (3050000)
        self.app_graph.get_widget_at_address("3050000").on_event("click", self.update_skope_rules)


        # -------- figure size ------ 
        def fig_size_changed(widget, event, data):  # 2121
            """ Called when the figureSizeSlider changed"""

            self._fig_size = widget.v_model

            self.vs_hde.set_fig_size(self._fig_size/2)
            self.es_hde.set_fig_size(self._fig_size/2)
            self.vs_hde.redraw()
            self.es_hde.redraw()

        # We wire the input event on the figureSizeSlider (050100)
        self.app_graph.get_widget_at_address("050100").on_event("input", fig_size_changed)

        # We wire the click event on addSkopeBtn (305020)
        self.app_graph.get_widget_at_address("305020").on_event("click", self.update_skope_rules)

        # -- Opacity button --
        def reset_opacity(*args):
            # We reset the opacity values
            self._opacity = [pd.Series(), pd.Series()]
            
            self.redraw_both_graphs()

        # We wire the click event on opacityBtn (112)
        self.app_graph.get_widget_at_address("112").on_event("click", reset_opacity)

        # -- Show beeswarms Checkbox --

        def show_beeswarms_check_changed(widget, event, data):
            # TODO : read the beeSwarmCheck from the event, not from the GUI
    
            refiners_list = self.app_graph.get_widget_at_address("30501").children

            for refiner in refiners_list:
                refiner.hide_beeswarm(widget.v_model)
        
        # We wire the change event on beeSwarmCheck (3050003)
        self.app_graph.get_widget_at_address("3050003").on_event("change", show_beeswarms_check_changed)

        display(self.app_graph.get_widget())
