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
from antakia.utils import conf_logger
from antakia.gui.widgets import (
    get_widget,
    change_widget,
    splash_widget,
    app_widget
)
from antakia.gui.highdimexplorer import HighDimExplorer
from antakia.gui.ruleswidget import RulesWidget

import logging
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
    update_graphs_stack : list of tuples # history for undos

    """

    def __init__(self, X_list, X_method_list: list, y: pd.Series, model, variables: list=None):
        self.X_list = X_list
        self.y_pred = model.predict(X_list[0])
        self.X_method_list = X_method_list
        self.y = y
        self.model = model

        self.variables = variables

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

        self.vs_rules_wgt = RulesWidget(self.X_list[0], self.variables, True, self.new_rules_defined)
        self.es_rules_wgt = RulesWidget(self.X_list[0], self.variables, False, self.new_rules_defined)
        # We set empty rules for now :
        self.vs_rules_wgt.init_rules(None, None)
        self.es_rules_wgt.init_rules(None, None)


        self.color = []
        self.selection_ids = []

        # Various inits 
        # No selection, so the datatable is not visible :
        get_widget(app_widget,"3041").disabled = True
        # No selection, so the tabs 2, 3 and 4 are not avaiable
        get_widget(app_widget,"301").disabled = True
        get_widget(app_widget,"302").disabled = False
        get_widget(app_widget,"303").disabled = False

        self.update_graphs_stack = []
        

    def __repr__(self) -> str:
        return "AntakIA's GUI"

    def show_splash_screen(self):
        """ Displays the splash screen and updates it during the first computations.
        """
        get_widget(splash_widget, "110").color = "light blue"
        get_widget(splash_widget, "110").v_model = 100
        get_widget(splash_widget, "210").color = "light blue"
        get_widget(splash_widget, "210").v_model = 100
        display(splash_widget)
        
        # We trigger VS proj computation :
        get_widget(splash_widget, "220").v_model = f"{DimReducMethod.dimreduc_method_as_str(config.DEFAULT_VS_PROJECTION)} on {self.X_list[0].shape} x 4"
        self.vs_hde.compute_projs(self.update_splash_screen)


        # We trigger ES explain computation if needed :
        if self.es_hde.pv_list[0] is None:
            # We compute default explanations :
            index = 1 if config.DEFAULT_EXPLANATION_METHOD == ExplanationMethod.SHAP else 3
            get_widget(splash_widget, "120").v_model = f"{ExplanationMethod.explain_method_as_str(config.DEFAULT_EXPLANATION_METHOD)} on {self.X_list[0].shape}"
            self.es_hde.pv_list[0] = ProjectedValues(self.new_values_wanted(index, self.update_splash_screen))
        else:
            get_widget(splash_widget, "120").v_model = f"{self.labels_list[self.es_hde.current_pv]}"

        # THen we trigger ES proj computation :
        self.es_hde.compute_projs(self.update_splash_screen)
        
        splash_widget.close()

        self.show_app()

    def update_splash_screen(self, caller: LongTask, progress: int, duration:float):
        """ 
        Updates progress bars of the splash screen
        """
        
        if isinstance(caller, ExplanationMethod):
            # It's an explanation
            progress_linear = get_widget(splash_widget, "110")
            number = 1
        else:  # It's a projection
            progress_linear = get_widget(splash_widget, "210")
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
            get_widget(app_widget,"3041").disabled = True
            # We disable the SkopeButton
            get_widget(app_widget,"3050001").disabled = True
            
        else: 
            selection_status_str = f"Current selection : {len(new_selection_indexes)} point selected {round(100*len(new_selection_indexes)/len(self.X_list[0]))}% of the  dataset"
            
            # We show the datatable :
            get_widget(app_widget,"3041").disabled = False
            # TODO : format the cells, remove digits
            change_widget(app_widget,"3041010", v.DataTable(
                    v_model=[],
                    show_select=False,
                    headers=[{"text": column, "sortable": True, "value": column } for column in self.X_list[0].columns],
                    # IMPORTANT note : df.loc(index_ids) vs df.iloc(row_ids)
                    items=self.X_list[0].loc[new_selection_indexes].to_dict("records"),
                    hide_default_footer=False,
                    disable_sort=False,
                )
            )

            # Navigation
            # We open tab2 "refinement"
            get_widget(app_widget,"301").disabled = False
            # We enable the SkopeButton
            get_widget(app_widget,"3050001").disabled = False
            # We disable the 'back_to_selection' button:
            get_widget(app_widget, "3050000").disabled = True
            # We set the switch space to the  correct side :
            get_widget(app_widget,"3050003").v_model = caller == self.es_hde
            self.vs_rules_wgt.disable(caller == self.es_hde)
            self.es_rules_wgt.disable(caller == self.vs_hde)

        change_widget(app_widget,"304010", selection_status_str)
        
        # We syncrhonize selection between the two HighDimExplorers
        other_hde = self.es_hde if caller == self.vs_hde else self.vs_hde
        other_hde.set_selection(new_selection_indexes)

        # We store the new selection
        self.selection_ids = new_selection_indexes

    def new_rules_defined(self, rules_widget: RulesWidget, df_indexes: list, skr:bool=False):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        # Navigation :
        # We enable the 'Update graphs' buttonn, but not for the SKR init
        logger.debug(f"new_rules_defined")
        if not skr:
            # We enbale the 'Update graphs' button
            get_widget(app_widget, "3050005").disabled = False

        # We make HDE figures non selectable :
        self.vs_hde.set_selection_disabled(True)
        self.es_hde.set_selection_disabled(True)
        # We disable tab 1 :
        get_widget(app_widget,"300").disabled = True

        # We make sure we're in 2D :
        get_widget(app_widget, "102").v_model == 2 # Switch button
        self.vs_hde.set_dimension(2)
        self.es_hde.set_dimension(2)

        # We sent to the proper HDE the rules_indexes to render :
        self.vs_hde.display_rules(df_indexes) if rules_widget.is_value_space else self.es_hde.display_rules(df_indexes)

    def show_app(self):

        # --------- Two HighDimExplorers ----------

        # We attach each HighDimExplorers component to the app_graph:        
        change_widget(app_widget, "2001", self.vs_hde.container),
        change_widget(app_widget, "1200", self.vs_hde.get_projection_select())
        change_widget(app_widget, "12020", self.vs_hde.get_projection_prog_circ())
        change_widget(app_widget, "1201000", self.vs_hde.get_proj_params_menu())
        change_widget(app_widget, "2011", self.es_hde.container)
        change_widget(app_widget, "1210", self.es_hde.get_projection_select())
        change_widget(app_widget, "12120", self.es_hde.get_projection_prog_circ())
        change_widget(app_widget, "1211000", self.es_hde.get_proj_params_menu())
        change_widget(app_widget, "111", self.es_hde.get_values_select())
        change_widget(app_widget, "112", self.es_hde.get_compute_menu())

        # ------------------ figure size ---------------
        def fig_size_changed(widget, event, data):
            """ Called when the figureSizeSlider changed"""
            self.vs_hde.fig_size = self.es_hde.fig_size = round(widget.v_model/2)
            self.vs_hde.redraw()
            self.es_hde.redraw()

        # We wire the input event on the figureSizeSlider (050100)
        get_widget(app_widget,"04000").on_event("input", fig_size_changed)
        # We set the init value to default :
        get_widget(app_widget,"04000").v_model=config.INIT_FIG_WIDTH
        
        # --------- ColorChoiceBtnToggle ------------
        def change_color(*args):
            """
                Called with the user clicks on the colorChoiceBtnToggle
                Allows change the color of the dots
            """
            # TODO : read the choice from the event, not from the GUI
            choice = get_widget(app_widget,"110").v_model

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
        get_widget(app_widget, "110").on_event("change", change_color)

        # ------- Dimension Switch ----------

        def switch_dimension(widget, event, data,):
            """
            Called when the switch changes.
            We call the HighDimExplorer to update its figure and, enventually,
            compute its proj
            """
            self.vs_hde.set_dimension(3 if data else 2)
            self.es_hde.set_dimension(3 if data else 2)


        get_widget(app_widget, "102").v_model == config.DEFAULT_VS_DIMENSION
        get_widget(app_widget, "102").on_event("change", switch_dimension)

        # ------------- Tab 2 : refinement -----------

        # ------------- Skope rules ----------------

        # We add our 2 RulesWidgets to the GUI :
        change_widget(app_widget, "305010", self.vs_rules_wgt.root_widget)
        change_widget(app_widget, "305011", self.es_rules_wgt.root_widget)

        def compute_rules(widget, event, data):
            # if clicked, selection can't be empty
            # Let's disable the button during computation:
            get_widget(app_widget,"3050001").disabled = True

            rules_found=False
            # We compute SKR on the space indicated by the switch
            if get_widget(app_widget,"3050003").v_model :
                es_rules_list, es_score_dict = Rule.compute_rules(self.selection_ids, self.es_hde.get_current_X(), False, self.variables)
                self.es_rules_wgt.init_rules(es_rules_list, es_score_dict)
                rules_found = True
            else:
                vs_rules_list, vs_score_dict = Rule.compute_rules(self.selection_ids, self.vs_hde.get_current_X(), True, self.variables)
                self.vs_rules_wgt.init_rules(vs_rules_list, vs_score_dict)
                rules_found = True

            # Navigation :
            # If rules found, we enable 'back_to_selection' button:
            if rules_found: get_widget(app_widget, "3050000").disabled = False

        # We wire the click event on the 'Skope-rules' button
        get_widget(app_widget,"3050001").on_event("click", compute_rules)

        def switch_space(widget, event, data):
            logger.debug(f"switch_space: widget = {widget.__class__}, event = {event}, data = {data}")
            # TODO : understand why data is True or empty dict {}
            if data == True:  # ES side
                self.vs_rules_wgt.disable(True)
                self.es_rules_wgt.disable(False)
            else: # VS side
                self.vs_rules_wgt.disable(False)
                self.es_rules_wgt.disable(True)

        # We wire the change event on the VS/ES switch
        get_widget(app_widget,"3050003").on_event("change", switch_space)
        # At init, the switch is on VS s:
        get_widget(app_widget,"3050003").v_model = False
        get_widget(app_widget,"3050003").disabled = False
        self.vs_rules_wgt.disable(False)
        self.es_rules_wgt.disable(True)


        def back_to_selection(widget, event, data):
            # Navigation :
            # We enable tab 1 
            get_widget(app_widget,"300").disabled = False
            # We show tab 1
            get_widget(app_widget,"30").v_model=0
            # We remove HDE's rules_traces
            self.vs_hde.display_rules(None)
            self.es_hde.display_rules(None)
            # We disable the 'Update graphs' button
            get_widget(app_widget, "3050005").disabled = True
            # We empty our RulesWidgets
            self.vs_rules_wgt.init_rules(None, None)
            self.es_rules_wgt.init_rules(None, None)
            # We make HDE figures selectable
            self.vs_hde.set_selection_disabled(False)
            self.es_hde.set_selection_disabled(False)
            # We fon't disable tab 1 since there's an active selection

        # We wire the click event on the 'back to selection' btn
        get_widget(app_widget, "3050000").on_event("click", back_to_selection)

        # At start the button is disabled
        get_widget(app_widget, "3050000").disabled = True
        # Its enabled when a rule is found and disabled when a selection is made

        def update_graphs(widget, event, data):
            self.update_graphs_stack.append([self.vs_rules_wgt.current_index, self.es_rules_wgt.current_index])
            # We update the HDEs with the new rules
            self.vs_hde.display_rules(self.vs_rules_wgt.rules_indexes)
            self.es_hde.display_rules(self.es_rules_wgt.rules_indexes)
            # Navigation :
            # The 'Update graphs' button is disabled until new rules are defined
            get_widget(app_widget, "3050005").disabled = True
            # Now the 'Undo' button is enabled
            get_widget(app_widget, "3050006").disabled = False
            # TODO : implement when undo should be disabled

        # We wire the ckick event on the 'Update graphs' button
        get_widget(app_widget, "3050005").on_event("click", update_graphs)
        # At start the button is disabled
        get_widget(app_widget, "3050005").disabled = True
        # Its enabled when rules are modified in a RsW and disabled when we're back to selections

        def undo(widget, event, data):
            logger.debug(f"undo stack is {len(self.update_graphs_stack)}")
            last_refresh_indexes = self.update_graphs_stack[len(self.update_graphs_stack)-1]
            self.update_graphs_stack.pop(-1)
            self.vs_rules_wgt.current_index = last_refresh_indexes[0]
            self.es_rules_wgt.current_index = last_refresh_indexes[1]
            self.vs_hde.redraw()
            self.es_hde.redraw()
            update_graphs(None, None, None)
            if len(self.update_graphs_stack) == 0:
                get_widget(app_widget, "3050006").disabled = True
            

        # We wire the ckick event on the 'Undo' button
        get_widget(app_widget, "3050006").on_event("click", undo)
        # At start the button is disabled
        get_widget(app_widget, "3050006").disabled = True
        # Its enabled when rules graphs have been updated with rules



        # ------------- Tab 3 : sub-models -----------

        def skr_change_side(widget, event, data):
            self.vs_rules_wgt.display(data)
            self.es_rules_wgt.display(not data)



        # ------------- Tab 4 : regions -----------

        display(app_widget)
