from copy import deepcopy
from importlib.resources import files


import numpy as np
import pandas as pd

import plotly.graph_objects as go
import ipyvuetify as v
from IPython.display import display
from ipywidgets import Layout, widgets


from antakia.data import DimReducMethod, ExplanationMethod, Variable, LongTask
import antakia.config as config
from antakia.compute import (
    auto_cluster, compute_explanations
)
from antakia.data import (  
    ExplanationMethod,
    ProjectedValues
)
import antakia.config as config
from antakia.rules import Rule
from antakia.utils import confLogger, overlapHandler
from antakia.gui_utils import (
    HighDimExplorer,
    RulesWidget,
    get_widget_at_address,
    change_widget,
    splash_widget,
    app_widget
)

import logging
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
    X_list : a list of one or more Pandas DataFrame
    X_method_list : a list of int, each int being an ExplanationMethod constant
    y : Pandas Series
    y_pred : Pandas Series, calculated at initialization with the model
    model : a model
    variables : a list of Variable
    selection_ids : a list of a pd.DataFrame indexes, corresponding to the current selected points
        IMPORTANT : a dataframe index may differ from the row number
    vs_hde, es_hde : HighDimExplorer for the VS and ES space
    vs_rules_wgt, es_rules_wgt : RulesWidget
    color : Pandas Series : color for each Y point

    """

    def __init__(self, X_list, X_method_list: list, y: pd.Series, model):
        self.X_list = X_list
        self.y_pred = model.predict(X_list[0])
        self.X_method_list = X_method_list
        self.y = y
        self.model = model

        self.variables = Variable.guess_variables(X_list[0])

        # We create our VS HDE
        self.vs_hde = HighDimExplorer(
            "Values space",
            [X_list[0]],
            [None], # ignored
            [None], # ignored
            y,
            config.DEFAULT_VS_PROJECTION,
            config.DEFAULT_VS_DIMENSION,
            config.INIT_FIG_WIDTH / 2,
            40, # border size
            self.selection_changed)
        
        self.vs_rules_wgt = self.es_rules_wgt = None
        
        # We create our ES HDE :
        self.labels_list = ["SHAP imported", "SHAP computed", "LIME imported", "LIME computed"]
        temp_is_computable_list = [False, True, False, True]
        # We put our dataframes in the right slots :
        # Remember : X_list[0] carries the original X values X_list[>=1] carry the explanations
        if len(X_list) == 1:

            temp_dataframes_list = [None, None, None, None]

        elif len(X_list) == 2:
            if X_method_list[1] == ExplanationMethod.SHAP:
                temp_dataframes_list = [X_list[1], None, None, None]
            elif X_method_list[1] == ExplanationMethod.LIME:
                temp_dataframes_list = [None, None, X_list[1], None]
            else:
                raise ValueError("X_method_list[1] must be SHAP or LIME")
        else:
            raise NotImplementedError(f"We only support initialization with 1 set of explanations (LIME or SHAP), not more")

        self.es_hde = HighDimExplorer(
            "Explanations space", 
            temp_dataframes_list,
            self.labels_list,
            temp_is_computable_list,
            y,
            config.DEFAULT_ES_PROJECTION, 
            config.DEFAULT_VS_DIMENSION, # We use the same dimension as the VS HDE for now
            config.INIT_FIG_WIDTH / 2,
            40, # border size
            self.selection_changed, 
            self.new_values_wanted)
        

        self.color = []
        self.selection_ids = []

        # Various inits 
        # No selection, so the datatable is not visible :
        get_widget_at_address(app_widget,"3041").disabled = True
        # No selection, so the tabs 2, 3 and 4 are not avaiable
        get_widget_at_address(app_widget,"301").disabled = True
        get_widget_at_address(app_widget,"302").disabled = True
        get_widget_at_address(app_widget,"303").disabled = True
        

    def __repr__(self) -> str:
        return "Hello i'm the GUI !"

    def show_splash_screen(self):
        """ Displays the splash screen and updates it during the first computations.
        """
        get_widget_at_address(splash_widget, "110").color = "light blue"
        get_widget_at_address(splash_widget, "110").v_model = 100
        get_widget_at_address(splash_widget, "210").color = "light blue"
        get_widget_at_address(splash_widget, "210").v_model = 100
        display(splash_widget)
        
        # We trigger VS proj computation :
        get_widget_at_address(splash_widget, "220").v_model = f"{DimReducMethod.dimreduc_method_as_str(config.DEFAULT_VS_PROJECTION)} on {self.X_list[0].shape} x 4"
        self.vs_hde.compute_projs(self.update_splash_screen)


        # We trigger ES explain computation if needed :
        if self.es_hde.pv_list[0] is None:
            # We compute default explanations :
            index = 1 if config.DEFAULT_EXPLANATION_METHOD == ExplanationMethod.SHAP else 3
            get_widget_at_address(splash_widget, "120").v_model = f"{ExplanationMethod.explain_method_as_str(config.DEFAULT_EXPLANATION_METHOD)} on {self.X_list[0].shape}"
            self.es_hde.pv_list[0] = ProjectedValues(self.new_values_wanted(index, self.update_splash_screen))
        else:
            get_widget_at_address(splash_widget, "120").v_model = f"{self.labels_list[self.es_hde.current_pv]}"

        # THen we trigger ES proj computation :
        self.es_hde.compute_projs(self.update_splash_screen)
        
        splash_widget.close()
        self.show_app()

    def update_splash_screen(self, caller: LongTask, progress: int, duration:float):
        """ Its role as a callback is to update the 2 progress bars of the splash screen"""
        
        if isinstance(caller, ExplanationMethod):
            # It's an explanation
            progress_linear = get_widget_at_address(splash_widget, "110")
            number = 1
        else:  # It's a projection
            progress_linear = get_widget_at_address(splash_widget, "210")
            number = 4 # (VS/ES) x (2D/3D)
        
        if progress_linear.color == "light blue":
            progress_linear.color = "blue"
            progress_linear.v_model = 0

        progress_linear.v_model = round(progress/number)

        if progress_linear.v_model == 100:
            progress_linear.color = "light blue"

    def new_values_wanted(self, new_values_index:int, callback:callable=None)-> pd.DataFrame:
        """
        Called either by :
        - the splash screen
        - the ES HighDimExplorer (HDE) when the user wants to compute new explain values
        callback is a HDE function to update the progress linear
        """
        
        desired_explain_method = None
        if new_values_index == 1:
            desired_explain_method = ExplanationMethod.SHAP
        elif new_values_index == 3:
            desired_explain_method = ExplanationMethod.LIME
        else:
            raise ValueError("new_values_index can only be 1 or 3 for now")
        
        if desired_explain_method in self.X_method_list:
            raise ValueError("This explain method was already provided")

        logger.debug(f"new_values_wanted: i will compute {'SHAP' if desired_explain_method==ExplanationMethod.SHAP else 'LIME'} ")

        return compute_explanations(
            self.X_list[0],
            self.model, 
            desired_explain_method,
            callback
            )

    
    def selection_changed(self, caller:HighDimExplorer, new_selection_indexes: list):
        """ Called when the selection of one HighDimExplorer changes
        """
        
        selection_status_str = ""

        if len(new_selection_indexes)==0:
            selection_status_str = f"No point selected. Use the 'lasso' tool to select points on one of the two graphs"
            # We disable the datatable :
            get_widget_at_address(app_widget,"3041").disabled = True
            # We disable the SkopeButton
            get_widget_at_address(app_widget,"3050000").disabled = True
            
        else: 
            selection_status_str = f"Current selection : {len(new_selection_indexes)} point selected {round(100*len(new_selection_indexes)/len(self.X_list[0]))}% of the  dataset"
            get_widget_at_address(app_widget,"3041").disabled = False
            # TODO : format the cells, remove digits
            change_widget(app_widget,"3041010000", v.DataTable(
                    v_model=[],
                    show_select=False,
                    headers=[{"text": column, "sortable": True, "value": column } for column in self.X_list[0].columns],
                    # IMPORTANT note : df.loc(index_ids) and df.iloc(row_ids)
                    items=self.X_list[0].loc[new_selection_indexes].to_dict("records"),
                    hide_default_footer=False,
                    disable_sort=False,
                )
            )
            # Navigation
            # We open tab2 "refinement"
            get_widget_at_address(app_widget,"301").disabled = False
            # We enable the SkopeButton
            get_widget_at_address(app_widget,"3050000").disabled = False

        change_widget(app_widget,"3040010", selection_status_str)
        
        # We syncrhonize selection between the two HighDimExplorers
        other_hde = self.es_hde if caller == self.vs_hde else self.vs_hde
        other_hde.set_selection(new_selection_indexes)

        # We store the new selection
        self.selection_ids = new_selection_indexes

    def show_app(self):
        # --------- Two HighDimExplorers ----------

        # We attach each HighDimExplorers component to the app_graph :
        change_widget(app_widget, "2001", self.vs_hde.container)
        change_widget(app_widget, "1200", self.vs_hde.get_projection_select())
        change_widget(app_widget, "12020", self.vs_hde.get_projection_prog_circ())
        change_widget(app_widget, "1201000", self.vs_hde.get_proj_params_menu())
        change_widget(app_widget, "2011", self.es_hde.container)
        change_widget(app_widget, "1210", self.es_hde.get_projection_select())
        change_widget(app_widget, "12120", self.es_hde.get_projection_prog_circ())
        change_widget(app_widget, "1211000", self.es_hde.get_proj_params_menu())
        change_widget(app_widget, "111", self.es_hde.get_values_select())
        change_widget(app_widget, "112", self.es_hde.get_compute_menu())

        
        # --------- ColorChoiceBtnToggle ------------
        def change_color(*args):
            """
                Called with the user clicks on the colorChoiceBtnToggle
                Allows change the color of the dots
            """
            # TODO : read the choice from the event, not from the GUI
            choice = get_widget_at_address(app_widget,"110").v_model

            self.color = None
            if choice == "y":
                self.color = self.y
            elif choice == "y^":
                self.color = self.y_pred
            elif choice == "current selection":
                logger.debug(f"change_color: we ignore 'current selection' choice with grey and blue")
                # self.color = ["grey"] * len(self.y)
                # for i in range(len(self.selection.indexes)):
                #     self.color[self.selection.indexes[i]] = "blue"
            elif choice == "residual":
                self.color = self.y - self.y_pred
                self.color = [abs(i) for i in self.color]
            elif choice == "regions":
                logger.debug(f"change_color: we ignore 'regions' choice")
                # self.color = [0] * len(self.y)
                # for i in range(len(self.y)):
                #     for j in range(len(self.regions)):
                #         if i in self.regions[j].get_indexes():
                #             self.color[i] = j + 1
            elif choice == "not selected":
                logger.debug(f"change_color: we ignore 'not selected' choice")
                # self.color = ["red"] * len(self.X_list[0])
                # if len(self.regions) > 0:
                #     for i in range(len(self.X_list[0])):
                #         for j in range(len(self.regions)):
                #             if i in self.regions[j].get_indexes():
                #                 self.color[i] = "grey"
            elif choice == "auto":
                logger.debug(f"change_color: we ignore 'auto' choice")
                # self.color = None # TODO

            self.vs_hde.redraw(self.color)
            self.es_hde.redraw(self.color)


        # Set "change" event on the Button Toggle used to chose color
        get_widget_at_address(app_widget, "110").on_event("change", change_color)

        # ------- Dimension Switch ----------

        def switch_dimension(widget, event, data,):
            """
            Called when the switch changes.
            We call the HighDimExplorer to update its figure and, enventually,
            compute its proj
            """
            self.vs_hde.set_dimension(3 if data else 2)
            self.es_hde.set_dimension(3 if data else 2)


        get_widget_at_address(app_widget, "102").v_model == config.DEFAULT_VS_DIMENSION
        get_widget_at_address(app_widget, "102").on_event("change", switch_dimension)

        # ---- Tab 2 : refinement ----

        # ---------- cluster and regions  ------------

        # Called when validating a tile to add it to the set of Regions
        def new_region_validated(*args):
            pass
            # if len(args) == 0:
            #     pass
            # elif self.selection in self.regions:
            #     # TODO : wez should have an Alert system
            #     print("AntakIA WARNING: this region is already in the set of Regions")
            # else:
            #     # TODO : selection is selection. We should create another Selection
            #     # self.selection.set_type(Selection.REGION)
            #     if self._model_index is None:
            #         model_name = None
            #         model_scores = [1, 1, 1]
            #     else:
            #         model_name = config.get_default_submodels[self._model_index].__class__.__name__
            #         # model_scores = self._subModelsScores[self._model_index]
            #     # if self.selection.get_vs_rules() is None:
            #         return

            #     # self.selection.setIndexesWithRules() # TODO not sure we have to call that

            #     # new_indexes = deepcopy(self.selection.get_indexes())
            #     self.selection.get_submodel()["name"], self.selection.get_submodel()["score"] = (
            #         model_name,
            #         model_scores,
            #     )
            #     # We check that all the points of the new region belong only to it: we will modify the existing tiles
            #     self.regions = overlapHandler(self.regions, new_indexes)
            #     self.selection.set_new_indexes(new_indexes)

            # self._regionColor = [0] * self._ds.__len__()
            # if self.regions is not None:
            #     for i in range(len(self.regions_colors)):
            #         for j in range(len(self.regions)):
            #             if i in self.regions[j].get_indexes():
            #                 self._regionColor[i] = j + 1
            #                 break

            # toute_somme = 0
            # temp = []
            # score_tot = 0
            # score_tot_glob = 0
            # autre_toute_somme = 0
            # for i in range(len(self.regions)):
            #     if self.regions[i].get_submodel()["score"] is None:
            #         temp.append(
            #             [
            #                 i + 1,
            #                 len(self.regions[i]),
            #                 np.round(
            #                     len(self.regions[i])
            #                     / len(self.atk.y)
            #                     * 100,
            #                     2,
            #                 ),
            #                 "/",
            #                 "/",
            #                 "/",
            #                 "/",
            #             ]
            #         )
            #     else:
            #         temp.append(
            #             [
            #                 i + 1,
            #                 len(self.regions[i]),
            #                 np.round(
            #                     len(self.regions[i])
            #                     / len(self._ds.get_full_values())
            #                     * 100,
            #                     2,
            #                 ),
            #                 self.regions[i].get_submodel()["model"],
            #                 self.regions[i].get_submodel()["score"][0],
            #                 self.regions[i].get_submodel()["score"][1],
            #                 str(self.regions[i].get_submodel()["score"][2]) + "%",
            #             ]
            #         )
            #         score_tot += self.regions[i].get_submodel()["score"][0] * len(
            #             self.regions[i]
            #         )
            #         score_tot_glob += self.regions[i].get_submodel()["score"][1] * len(
            #             self.regions[i]
            #         )
            #         autre_toute_somme += len(self.regions[i])
            #     toute_somme += len(self.regions[i])
            # if autre_toute_somme == 0:
            #     score_tot = "/"
            #     score_tot_glob = "/"
            #     percent = "/"
            # else:
            #     score_tot = round(score_tot / autre_toute_somme, 3)
            #     score_tot_glob = round(score_tot_glob / autre_toute_somme, 3)
            #     percent = (
            #         str(round(100 * (score_tot_glob - score_tot) / score_tot_glob, 1))
            #         + "%"
            #     )
            # temp.append(
            #     [
            #         "Total",
            #         toute_somme,
            #         np.round(toute_somme / len(self._ds) * 100, 2),
            #         "/",
            #         score_tot,
            #         score_tot_glob,
            #         percent,
            #     ]
            # )
            # pd.DataFrame(
            #     temp,
            #     columns=[
            #         "Region #",
            #         "Number of points",
            #         "% of the dataset",
            #         "Model",
            #         "Score of the sub-model (MSE)",
            #         "Score of the global model (MSE)",
            #         "Gain in MSE",
            #     ],
            # )
            # # TODO what shall we do with this dataframe ???

        # We wire a click event on validateRegionBtn(307000)
        get_widget_at_address(app_widget, "307000").on_event("click", new_region_validated)
        def cluster_number_changed(*b):
            # TODO : read the slider from the event, not from the GUI
            # We set clustersSliderTxt to the current clustersSlider value

            change_widget(app_widget, "304230", "Number of clusters " + str(get_widget_at_address(app_widget,"30422").v_model))

        # We wire the input event on the clustersSlider (30422)
        get_widget_at_address(app_widget,"30422").on_event("input", cluster_number_changed)

        def cluster_check_changed(*b):
            # TODO : read the slider from the event, not from the GUI
            # TODO : what is this clusterCheck ?
            # clusterSlider(30422) visibility is linked to clusterCheck(30421)
            get_widget_at_address(app_widget,"30422").disabled = get_widget_at_address(app_widget,"30421").v_model

        get_widget_at_address(app_widget,"30421").on_event("change", cluster_check_changed)

        # Let's create an empty / dummy cluster_results_table :
        new_df = pd.DataFrame([], columns=["Region #", "Number of points"])
        columns = [{"text": c, "sortable": True, "value": c} for c in new_df.columns]

        # We insert the cluster_results_table
        change_widget(app_widget, "304410", v.Row(
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
            get_widget_at_address(app_widget,"3043").class_ = "d-flex"

            # Depending on clusterCheck (30421) :
            if get_widget_at_address(app_widget,"30421").v_model:
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
                    get_widget_at_address(app_widget,"30423").v_model,
                    False,
                )

            # TODO we'll have to change this when Selection will be returned
            self.color = self._auto_cluster_regions[1]

            self.redraw_both_graphs()

            # We set our regions accordingly
            self.regions = self._auto_cluster_regions # just created, no need to copy
            # We update the GUI tab 1(304) clusterResults(3044) (ie a v.Row)
            # change_widget(app_widget, "304234", self.datatable_from_Selectiones(self.regions))


            # TODO : understand
            # tabOneSelectionColumn.children = tabOneSelectionColumn.children[:-1] + [
            #     clusterResults
            # ]

            # colorChoiceBtnToggle(111) must be changed :
            get_widget_at_address(app_widget,"30423").v_model = "auto"

            # We wire a change event on a button above the cluster_results_table DataTable (30420)
            get_widget_at_address(app_widget,"30420").on_event(
                "change", cluster_results_table_changed
            )

            # We udpate loadingClustersProgLinear (3043)
            get_widget_at_address(app_widget,"30440").class_ = "d-none"

            # TODO : check if we really expect an int from this function
            return len(self._auto_cluster_regions)
        
        # We wire the click event on the "Magic" findClusterBtn (30451)
        get_widget_at_address(app_widget,"30451").on_event("click", magic_btn_clicked)

        def cluster_results_table_changed(widget, event, data):  # 1803
            """
            Called when a new magic clustering ahs been computed
            """
            # TODO : maybe this event should not be called only for "auto regions"

            # TODO : this has to do with the color and the current 
            # auto_cluser() implementation. It should return Selectiones
            labels = self._auto_cluster_regions[1]

            # TODO : I guess tabOneSelectionColumn.children[-1].children[0].children[0] was refering to the DataTable
            index = get_widget_at_address(app_widget,"304420").v_model
            liste = [i for i, d in enumerate(labels) if d == float(index)]

            get_widget_at_address(app_widget,"30423").v_model = "auto"
            # We call change_color by hand 
            # TODO : register change_color on the cluster_results_table instead 
            change_color()


        
        def magic_checkbox_changed(widget, event, data):
            textField = get_widget_at_address(app_widget,"30453")
            if widget.v_model:
                textField.disabled = False
            else:
                textField.disabled = True

        # We wire a change event on magicCheckBox (or "demonstration mode" chekcbox)
        get_widget_at_address(app_widget,"30452").on_event("change", magic_checkbox_changed)

        # =============== Skope rules ===============

        def compute_rules(widget, event, data):
            # if clicked, selection can't be empty
            # Let's disable the button during computation:
            get_widget_at_address(app_widget,"3050000").disabled = True
            
            # We try fo find Skope Rules on "VS" side :
            vs_rules_list, vs_score_list = Rule.compute_rules(self.selection_ids, self.vs_hde.get_current_X(), True, self.variables)
            if len(vs_rules_list) > 0:
                logger.debug(f"compute_rules: VS rules found : {vs_rules_list}, {vs_score_list}")
                self.vs_rules_wgt = RulesWidget(self.vs_hde.get_current_X(), self.variables, True, vs_rules_list, vs_score_list, rules_updated)
                change_widget(app_widget,"305010", self.vs_rules_wgt.vbox_widget)
            else:
                logger.debug(f"compute_rules: no VS rules found")
            
            # We try fo find Skope Rules on "ES" side :
            es_rules_list, es_score_list = Rule.compute_rules(self.selection_ids, self.es_hde.get_current_X(), False, self.variables)
            if len(es_rules_list) > 0:
                logger.debug(f"compute_rules: ES rules found : {es_rules_list}, {es_score_list}")
                self.es_rules_wgt = RulesWidget(self.es_hde.get_current_X(), self.variables, False, es_rules_list, es_score_list, rules_updated)
                change_widget(app_widget,"305011", self.es_rules_wgt.vbox_widget)
            else:
                logger.debug(f"compute_rules: no ES rules found")

            get_widget_at_address(app_widget,"3050000").disabled = False

        # We wire the ckick event on the skope-rules buttonn (3050000)
        get_widget_at_address(app_widget,"3050000").on_event("click", compute_rules)


        def rules_updated(rules_widget: RulesWidget, df_indexes: list):
            """
            Called by a RulesWidget upond (skope) rule creation or when the user updates the rules via a RulesWidget
            The function asks the HDEs to display the rules result ()
            """
            # Navigation :
            # We make HDE figures non selectable :
            self.vs_hde.set_selection_disabled(True)
            self.es_hde.set_selection_disabled(True)
            # We disable tab 1 :
            get_widget_at_address(app_widget,"300").disabled = True


            logger.debug(f"rules_updated: received from {'VS RsW' if rules_widget.is_value_space else 'ES RsW'} : {rules_widget.get_current_rule_list()} rules to display")

            # We make sure we're in 2D :
            get_widget_at_address(app_widget, "102").v_model == 2 # Switch button
            self.vs_hde.set_dimension(2)
            self.es_hde.set_dimension(2)

            
            

            # We sent to the proper HDE the rules_indexes to render :
            self.vs_hde.display_rules(df_indexes) if rules_widget.is_value_space else self.es_hde.display_rules(df_indexes)


        def reinitSkopeRules(*b):
            pass
            # Why the rules should be saved ?
            # self.selection.setVSRules(self._save_rules)

            # If called, means the selection is back to a selection :
            # self.selection.setType(Selection.SELECTION)

            # We reset the refiners
            # self.reinit_skope_rules(None)

            # We udpate the scores for the submodels
            # self.update_submodels_scores(None)

        # We wire the click event on the reinitSkopeBtn (3050001)
        get_widget_at_address(app_widget, "3050001").on_event("click", reinitSkopeRules)


        # -------- figure size ------ 
        def fig_size_changed(widget, event, data):  # 2121
            """ Called when the figureSizeSlider changed"""
            self.vs_hde.fig_size = self.es_hde.fig_size = round(widget.v_model/2)
            self.vs_hde.redraw()
            self.es_hde.redraw()

        # We wire the input event on the figureSizeSlider (050100)
        get_widget_at_address(app_widget,"04000").on_event("input", fig_size_changed)
        # We set the init value to default :
        get_widget_at_address(app_widget,"04000").v_model=config.INIT_FIG_WIDTH

        # We wire the click event on addSkopeBtn (305020)
        get_widget_at_address(app_widget,"305020").on_event("click", None)

        # -- Show beeswarms Checkbox --

        def show_beeswarms_check_changed(widget, event, data):
            # TODO : read the beeSwarmCheck from the event, not from the GUI
    
            refiners_list = get_widget_at_address(app_widget,"30501").children

            for refiner in refiners_list:
                refiner.hide_beeswarm(widget.v_model)
        
        # We wire the change event on beeSwarmCheck (3050003)
        # get_widget_at_address(app_widget,"3050003").on_event("change", show_beeswarms_check_changed)

        display(app_widget)
