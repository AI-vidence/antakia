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
    X : Pandas DataFrame, the orignal dataset 
    y : Pandas Series, target values
    model : a model
    X_exp : a Pandas DataFrame, containing imported explanations
    variables : a list of Variable
    selection_ids : a list of a pd.DataFrame indexes, corresponding to the current selected points
        IMPORTANT : a dataframe index may differ from the row number
    vs_hde, es_hde : HighDimExplorer for the VS and ES space
    vs_rules_wgt, es_rules_wgt : RulesWidget
    update_graphs_stack : list of tuples # history for undos

    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, model, variables: list=None, X_exp: pd.DataFrame=None):
        self.X = X
        self.y = y
        self.model = model
        self.y_pred = model.predict(X)
        self.variables = variables

        # We create our VS HDE
        self.vs_hde = HighDimExplorer(
            X,
            y,
            config.DEFAULT_VS_PROJECTION,
            config.DEFAULT_VS_DIMENSION,
            config.INIT_FIG_WIDTH / 2,
            40, # border size
            self.selection_changed)
        
        self.vs_rules_wgt = self.es_rules_wgt = None
        
        # We create our ES HDE :

        self.es_hde = HighDimExplorer(
            X,
            y,
            config.DEFAULT_ES_PROJECTION, 
            config.DEFAULT_VS_DIMENSION, # We use the same dimension as the VS HDE for now
            config.INIT_FIG_WIDTH / 2,
            40, # border size
            self.selection_changed, 
            self.new_eplanation_values_required,
            X_exp if X_exp is not None else pd.DataFrame(), # passing an empty df meebns it's an ES HDE
            )

        self.vs_rules_wgt = RulesWidget(self.X, self.variables, True, self.new_rules_defined)
        self.es_rules_wgt = RulesWidget(self.X, self.variables, False, self.new_rules_defined)
        # We set empty rules for now :
        self.vs_rules_wgt.is_disabled=True
        # self.vs_rules_wgt.update_state()
        self.es_rules_wgt.is_disabled=True
        # self.es_rules_wgt.update_state()
    

        self.color = []
        self.selection_ids = []

        # UI rules : 
        # We disable tabs 2 and 3 at startup
        get_widget(app_widget,"41").disabled = True
        get_widget(app_widget,"42").disabled = True
        # We disable the selection datatable at startup (bottom of tab 1)
        get_widget(app_widget,"4320").disabled = True

        # We init the 'undo stack'
        self.update_graphs_stack = []
        
    def show_splash_screen(self):
        """ Displays the splash screen and updates it during the first computations.
        """
        get_widget(splash_widget, "110").color = "light blue"
        get_widget(splash_widget, "110").v_model = 100
        get_widget(splash_widget, "210").color = "light blue"
        get_widget(splash_widget, "210").v_model = 100
        display(splash_widget)
        
        # We trigger VS proj computation :
        get_widget(splash_widget, "220").v_model = f"{DimReducMethod.dimreduc_method_as_str(config.DEFAULT_VS_PROJECTION)} on {self.X.shape} x 4"
        self.vs_hde.compute_projs(False, self.update_splash_screen)

        # We trigger ES explain computation if needed :
        if self.es_hde.pv_list[1] is None: # No imported explanation values
            # We compute default explanations :
            index = 1 if config.DEFAULT_EXPLANATION_METHOD == ExplanationMethod.SHAP else 3
            get_widget(splash_widget, "120").v_model = f"{ExplanationMethod.explain_method_as_str(config.DEFAULT_EXPLANATION_METHOD)} on {self.X.shape}"
            self.es_hde.pv_list[0] = ProjectedValues(self.new_eplanation_values_required(index, self.update_splash_screen))
        else:
            get_widget(splash_widget, "120").v_model = "Imported values"

        # THen we trigger ES proj computation :
        self.es_hde.compute_projs(False, self.update_splash_screen)
        
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

    def new_eplanation_values_required(self, explain_method:int, callback:callable=None)-> pd.DataFrame:
        """
        Called either by :
        - the splash screen
        - the ES HighDimExplorer (HDE) when the user wants to compute new explain values
        callback is a HDE function to update the progress linear
        """

        return compute_explanations(
            self.X,
            self.model, 
            explain_method,
            callback
        )
    
    def selection_changed(self, caller:HighDimExplorer, new_selection_indexes: list):
        """ Called when the selection of one HighDimExplorer changes
        """
        selection_status_str = ""

        if len(new_selection_indexes)==0:
            selection_status_str_1 = f"No point selected"
            selection_status_str_2 = f"0% of the dataset"
            # We disable the selection datatable :
            get_widget(app_widget,"4320").disabled = True
            # We disable the SkopeButton
            get_widget(app_widget,"4301").disabled = True
            if self.vs_rules_wgt.is_disabled == False:
                # ongoing rules on VS RsW : we disable it
                self.vs_rules_wgt.is_disabled = True
                self.vs_rules_wgt.update_state()
            self.selection_ids = []
            
        else: 
            self.selection_ids = new_selection_indexes
            selection_status_str_1 = f"{len(new_selection_indexes)} point selected"
            selection_status_str_2 = f"{round(100*len(new_selection_indexes)/len(self.X))}% of the  dataset"
            
            # We show the selection datatable :
            get_widget(app_widget,"4320").disabled = False
            # TODO : format the cells, remove digits
            change_widget(app_widget,"432010", v.DataTable(
                    v_model=[],
                    show_select=False,
                    headers=[{"text": column, "sortable": True, "value": column } for column in self.X.columns],
                    # IMPORTANT note : df.loc(index_ids) vs df.iloc(row_ids)
                    items=self.X.loc[new_selection_indexes].to_dict("records"),
                    hide_default_footer=False,
                    disable_sort=False,
                )
            )

            # UI rules :
            # We enable the SkopeButton
            get_widget(app_widget,"4301").disabled = False

        change_widget(app_widget,"4300000", selection_status_str_1)
        change_widget(app_widget,"430010", selection_status_str_2)
        
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
        # We make sure we're in 2D :
        get_widget(app_widget, "10").v_model == 2 # Switch button
        self.vs_hde.set_dimension(2)
        self.es_hde.set_dimension(2)

        # We enable the rules_widget who found the rules :
        rules_widget.is_disabled = False
        rules_widget.update_state()

        # We sent to the proper HDE the rules_indexes to render :
        self.vs_hde.display_rules(df_indexes) if rules_widget.is_value_space else self.es_hde.display_rules(df_indexes)

        # We enable 'undo' and 'validate rules' buttons
        get_widget(app_widget, "4302").disabled = False
        get_widget(app_widget, "4303").disabled = False


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
        def change_color(widget, event, data):
            """
                Called with the user clicks on the colorChoiceBtnToggle
                Allows change the color of the dots
            """
            self.color = None
            if data == "y":
                self.color = self.y
            elif data == "y^":
                self.color = self.y_pred
            elif data == "residual":
                self.color = self.y - self.y_pred
                self.color = [abs(i) for i in self.color]

            self.vs_hde.redraw(self.color)
            self.es_hde.redraw(self.color)

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

        get_widget(app_widget, "10").v_model == config.DEFAULT_VS_DIMENSION
        get_widget(app_widget, "10").on_event("change", switch_dimension)

        # ------------- Skope rules ----------------

        # We add our 2 RulesWidgets to the GUI :
        change_widget(app_widget, "4310", self.vs_rules_wgt.root_widget)
        change_widget(app_widget, "4311", self.es_rules_wgt.root_widget)

        def compute_rules(widget, event, data):
            # if clicked, selection can't be empty
            # Let's disable the button during computation:
            get_widget(app_widget,"4301").disabled = True

            if self.vs_hde._has_lasso:
                vs_rules_list, vs_score_dict = Rule.compute_rules(self.selection_ids, self.vs_hde.get_current_X(), True, self.variables)
                self.vs_rules_wgt.init_rules(vs_rules_list, vs_score_dict)
            elif self.es_hde._has_lasso:
                es_rules_list, es_score_dict = Rule.compute_rules(self.selection_ids, self.es_hde.get_current_X(), False, self.variables)
                self.es_rules_wgt.init_rules(es_rules_list, es_score_dict)
            else:
                raise ValueError("compute_rules: called with no lasso")


        # We wire the click event on the 'Skope-rules' button
        get_widget(app_widget,"4301").on_event("click", compute_rules)
        # We disable the 'Skope-rules' button at startup
        get_widget(app_widget,"4301").disabled = True

        def undo(widget, event, data):
            pass
            # logger.debug(f"undo stack is {len(self.update_graphs_stack)}")
            # last_refresh_indexes = self.update_graphs_stack[len(self.update_graphs_stack)-1]
            # self.update_graphs_stack.pop(-1)
            # self.vs_rules_wgt.current_index = last_refresh_indexes[0]
            # self.es_rules_wgt.current_index = last_refresh_indexes[1]
            # self.vs_hde.redraw()
            # self.es_hde.redraw()
            # # update_graphs(None, None, None)
            # # if len(self.update_graphs_stack) == 0:
            # #     get_widget(app_widget, "4302").disabled = True
            

        # We wire the ckick event on the 'Undo' button
        get_widget(app_widget, "4302").on_event("click", undo)
        # At start the button is disabled
        get_widget(app_widget, "4302").disabled = True
        # Its enabled when rules graphs have been updated with rules

        def validate_rules(widget, event, data):
            pass

        # We wire the ckick event on the 'Valildate ruyles' button
        get_widget(app_widget, "4303").on_event("click", validate_rules)
        # At start the button is disabled
        get_widget(app_widget, "4303").disabled = True

        # ------------- Tab 2 : sub-models -----------


        # ------------- Tab 3 : regions -----------

        display(app_widget)
