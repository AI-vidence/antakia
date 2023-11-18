import pandas as pd

import ipyvuetify as v
from IPython.display import display

from antakia.data import DimReducMethod, LongTask,ExplanationMethod
from antakia.compute.explanations import compute_explanations
from antakia.compute.legacy_auto_cluster import auto_cluster, skope_rules

import antakia.config as config
from antakia.rules import Rule
import antakia.utils as utils
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
utils.conf_logger(logger)

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
    region_list : a list of Region,
        a region is a dict : {'rules': list of rules, 'indexes', 'model': class}
        if the list of rules is None, the region has been defined with auto-cluster
    selected_region : int, the selected region number; starts at 1. This is a hack to avoid using the ColorTable selection

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
        self.es_rules_wgt = RulesWidget(self.es_hde.get_current_X(), self.variables, False, self.new_rules_defined)
        # We set empty rules for now :
        self.vs_rules_wgt.disable(True)
        self.es_rules_wgt.disable(True)

        self.region_list = []
        self.selected_region = None # Holds the selection in our ColorTable
        self.selection_ids = []

        # UI rules :
        # We disable the selection datatable at startup (bottom of tab 1)
        get_widget(app_widget,"4320").disabled = True

        
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

    def update_regions_table(self):
        """
        Called to empty / fill the RegionDataTable with our region_list
        """

        def region_to_table_item(num:int, region:dict)-> dict:
            """
            Transforms a region dict into a dict to be displayed by the DataTable
            """
            indexes = len(region["indexes"]) if region["indexes"] is not None else str(Rule.rules_to_indexes(region["rules"], self.X).shape[0])
            return {
                "Region": num,
                "Rules": Rule.multi_rules_to_string(region["rules"]) if region["rules"] is not None else "auto-cluster",
                "Points": indexes,
                "% dataset": str(round(100*int(indexes)/len(self.X)))+"%",
                "Sub-model": region["model"],
                "Score": ""
            }

        def regions_to_items(regions:list)-> list:
            """
            Transforms a list of region dict into a list of dict to be displayed by the CustomDataTable
            """
            return [region_to_table_item(i+1, region) for i, region in enumerate(regions)]


        temp_items = regions_to_items(self.region_list)

        # We populate the ColorTable :
        get_widget(app_widget,"4400100").items = temp_items
        # We populate the regions DataTable :
        get_widget(app_widget,"4400110").items = temp_items
        
        region_stats = self.regions_stats()
        get_widget(app_widget,"44002").children=[f"{region_stats['regions']} {'regions' if region_stats['regions']>1 else 'region'}, {region_stats['points']} points, {region_stats['coverage']}% of the dataset"]
        
        # It seems HDEs need to display regions each time we udpate the table :
        self.vs_hde.display_regions(self.region_list)
        self.es_hde.display_regions(self.region_list)

        # UI rules :
        # If regions coverage > 80%, we disable the 'auto-cluster' button
        get_widget(app_widget,"440120").disabled = region_stats['coverage'] > 80

    def regions_stats(self)->dict:
        """ Computes the number of distinct points in the regions and the coverage in %
        """
        def union(list1:list, list2:list)-> list:
            return list(set(list1) | set(list2))
        stats = {}
        all_indexes = []
        for region in self.region_list:
            all_indexes = union(all_indexes, region["indexes"])

        stats['regions']=len(self.region_list)
        stats['points']=len(all_indexes)
        stats['coverage']=round(100*len(all_indexes)/len(self.X))
        return stats



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

        # UI rules :
        # Selection (empty or not) we remove any rule or region trace from HDEs
        self.vs_hde.display_rules(None)
        self.es_hde.display_rules(None)
        self.region_list = []
        self.update_regions_table()
    
        # Selection (empty or not) we reset both RulesWidgets
        self.vs_rules_wgt.disable(True)
        self.es_rules_wgt.disable(True)

        if len(new_selection_indexes)==0:
            # Selection gets emty
            selection_status_str_1 = f"No point selected"
            selection_status_str_2 = f"0% of the dataset"

            # UI rules :
            # We disable the Skope button
            get_widget(app_widget,"4301").disabled = True
            # We disable 'undo' and 'validate rules' buttons
            get_widget(app_widget,"4302").disabled = True
            get_widget(app_widget,"4303").disabled = True
            # We enable HDEs (proj select, explain select etc.)
            self.vs_hde.disable_widgets(False)
            self.es_hde.disable_widgets(False)
            # We disable the selection datatable :
            get_widget(app_widget,"4320").disabled = True

            self.selection_ids = []
        else: 
            # Selection is not empty anymore / changes
            self.selection_ids = new_selection_indexes

            # UI rules : we update the selection status :
            selection_status_str_1 = f"{len(new_selection_indexes)} point selected"
            selection_status_str_2 = f"{round(100*len(new_selection_indexes)/len(self.X))}% of the  dataset"
            # We enable the SkopeButton
            get_widget(app_widget,"4301").disabled = False
            # We disable HDEs
            self.vs_hde.disable_widgets(True)
            self.es_hde.disable_widgets(True)
            # We show and fill the selection datatable :
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
        # We syncrhonize selection between the two HighDimExplorers
        other_hde = self.es_hde if caller == self.vs_hde else self.vs_hde
        other_hde.set_selection(new_selection_indexes)
        # We store the new selection
        self.selection_ids = new_selection_indexes

        # UI rules :
        # We update the selection status :
        change_widget(app_widget,"4300000", selection_status_str_1)
        change_widget(app_widget,"430010", selection_status_str_2)
        

    def new_rules_defined(self, rules_widget: RulesWidget, df_indexes: list, skr:bool=False):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        # We make sure we're in 2D :
        get_widget(app_widget, "10").v_model == 2 # Switch button
        self.vs_hde.set_dimension(2)
        self.es_hde.set_dimension(2)

        # We sent to the proper HDE the rules_indexes to render :
        self.vs_hde.display_rules(df_indexes) if rules_widget.is_value_space else self.es_hde.display_rules(df_indexes)

        # We disable the 'undo' button if RsW has less than 2 rules
        get_widget(app_widget, "4302").disabled = rules_widget.current_index <1
        # We disable the 'validate rules' button if RsW has less than 1 rule
        get_widget(app_widget, "4303").disabled = rules_widget.current_index < 0
        
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
        get_widget(app_widget,"03000").on_event("input", fig_size_changed)
        # We set the init value to default :
        get_widget(app_widget,"03000").v_model=config.INIT_FIG_WIDTH
        
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

        def compute_skope_rules(widget, event, data):
            # if clicked, selection can't be empty
            # Let's disable the Skope button. I will be re-enabled if a new selection occurs
            get_widget(app_widget,"4301").disabled = True

            hde = self.vs_hde if self.vs_hde._has_lasso else self.es_hde
            rsw = self.vs_rules_wgt if self.vs_hde._has_lasso else self.es_rules_wgt
            skr_rules_list, skr_score_dict = skope_rules(self.selection_ids, hde.get_current_X(), self.variables)
            if len(skr_rules_list) > 0: # SKR rules found
                # UI rules :
                # We enable the 'validate rule' button
                get_widget(app_widget,"4303").disabled = False
                # We enable RulesWidet and init it wit the rules
                rsw.disable(False)
                rsw.init_rules(skr_rules_list, skr_score_dict, self.selection_ids )
            else:
                # No skr found
                rsw.show_msg("No Skope rules found", "red--text")

        # We wire the click event on the 'Skope-rules' button
        get_widget(app_widget,"4301").on_event("click", compute_skope_rules)
        # UI rules :
        # The Skope rules button is disabled at startup
        # It's only enabled when a selection occurs - when the selection is empty, it's disabled again
        # Startup status :
        get_widget(app_widget,"4301").disabled = True

        def undo(widget, event, data):
            if self.vs_rules_wgt.current_index > 0:
                self.vs_rules_wgt.undo()
                if self.vs_rules_wgt.current_index == 0:
                    # We disable the 'undo' button
                    get_widget(app_widget,"4302").disabled = True
            else:
                self.es_rules_wgt.undo()
                if self.es_rules_wgt.current_index == 0:
                    # We disable the 'undo' button
                    get_widget(app_widget,"4302").disabled = True

        # We wire the ckick event on the 'Undo' button
        get_widget(app_widget, "4302").on_event("click", undo)
        # At start the button is disabled
        get_widget(app_widget, "4302").disabled = True
        # Its enabled when rules graphs have been updated with rules

        def validate_rules(widget, event, data):
            # We get the rules to validate
            if self.vs_rules_wgt.current_index>=0:
                rules_widget = self.vs_rules_wgt
                hde = self.vs_hde
            else:
                rules_widget = self.es_rules_wgt
                hde = self.es_hde

            rules_list = rules_widget.get_current_rules_list()
            # We add them to our region_list
            self.region_list.append({
                "rules": rules_list,
                "indexes": Rule.rules_to_indexes(rules_list, self.X),
                "model": None
                })
            # And update the rules table (tab 2)
            self.update_regions_table()
            # We force tab 2
            get_widget(app_widget,"4").v_model=1

            # UI rules :
            # We clear the RulesWidget
            rules_widget.disable(True)
            rules_widget.init_rules(None, None, None)
            # We disable the 'undo' button
            get_widget(app_widget,"4302").disabled = True
            # We disable the 'validate rules' button
            get_widget(app_widget,"4303").disabled = True
            # We clear HDE 'rule traces'
            hde.display_rules(None)

            # We enable the 'Skope-rules' button
            get_widget(app_widget,"4301").disabled = False

        # We wire the click event on the 'Valildate rules' button
        get_widget(app_widget, "4303").on_event("click", validate_rules)

        # UI rules :
        # The 'validate rules' button is disabled at startup
        get_widget(app_widget, "4303").disabled = True
        # It's enabled when a SKR rules has been found and is disabled when the selection gets empty
        # or when validated is pressed

        # ------------- Tab 2 : regions -----------

        def region_selected(data):
            is_selected = data["value"]

            # We store this GUI attribute to store the selected region
            # TODO : read the selected region from the ColorTable
            self.selected_region = data["item"]["Region"] if is_selected else None

            #UI rules :
            # Is selected, we enable the 'substitute' and 'delete' buttons and vice-versa
            get_widget(app_widget,"440100").disabled = not is_selected
            get_widget(app_widget,"440110").disabled = not is_selected

        get_widget(app_widget,"4400100").set_callback(region_selected)

        def substitute_clicked(widget, event, data):
            assert self.selected_region is not None
            # We enable and show the substiution tab
            get_widget(app_widget,"42").disabled=True
            get_widget(app_widget,"4").v_model=2

        # We wire events on the 'substitute' button:
        get_widget(app_widget,"440100").on_event("click", substitute_clicked)

        # UI rules :
        # The 'substitute' button is disabled at startup; it'es enabled when a region is selected 
        # and disabled if no region is selected or when substitute is pressed
        get_widget(app_widget, "440100").disabled = True

        def delete_region_clicked(widget, event, data):
            """
            Called when the user clicks on the 'delete' (region) button
            """
            # We get the region to delete :
            # See https://vuetifyjs.com/en/components/data-tables/data-and-display/#selectable-rows
            selected_region = get_widget(app_widget,"4400110").v_model
            # What about multiple selection ?

            # Then we delete the regions in self.region_list
            # We keep numbers
            # We call udpatetable
            self.update_regions_table() # HDEs are updated then
            # if zero region, we disable the button

        # We wire events on the 'delete' button:
        get_widget(app_widget,"440110").on_event("click", delete_region_clicked)

        # UI rules :
        # The 'delete' button is disabled at startup
        get_widget(app_widget, "440110").disabled = True


        def auto_cluster_clicked(widget, event, data):
            """
            Called when the user clicks on the 'auto-cluster' button
            """
            if self.regions_stats()["coverage"]> 80:
                # UI rules :
                # region_list coverage is > 80% : we need to clear it to do another auto-cluster
                self.region_list = []
    
            # We assemble indices ot all existing regions :
            rules_indexes_list=[Rule.rules_to_indexes(region["rules"], self.X) for region in self.region_list]

            # We call the auto_cluster with remaing X and explained(X) :
            not_rules_indexes_list = [index for index in self.X.index if index not in rules_indexes_list]

            found_clusters = auto_cluster(
                self.X.loc[not_rules_indexes_list],
                self.es_hde.get_current_X().loc[not_rules_indexes_list],
                # We read the number of clusters from the Slider
                get_widget(app_widget,"440130").v_model,
                True
                ) # type: ignore

            for cluster in found_clusters:
                logger.debug(f"raw cluster : {cluster}")
                if cluster is not None and isinstance(cluster, list) and len(cluster)>0:
                    # The auto_cluster  returns lists of row_ids. We need to convert them ot the Dataframe indexes
                    self.region_list.append({"rules": None, "indexes": utils.rows_to_indexes(self.X, cluster), "model": None})

            self.update_regions_table()

        # We wire events on the 'auto-cluster' button :
        get_widget(app_widget,"440120").on_event("click", auto_cluster_clicked)
        # UI rules :
        # The 'auto-cluster' button is disabled at startup
        get_widget(app_widget, "440120").disabled = True


        def num_cluster_changed(widget, event, data):
            """
            Called when the user changes the number of clusters
            """
            # We enable the 'auto-cluster' button
            get_widget(app_widget,"440120").disabled = False

        # We wire events on the 'auto-cluster' button :
        get_widget(app_widget,"440130").on_event("change", num_cluster_changed)
        
        self.update_regions_table()

        display(app_widget)
