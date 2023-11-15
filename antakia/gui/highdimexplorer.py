import pandas as pd

from plotly.graph_objects import FigureWidget, Scatter, Scatter3d
import ipyvuetify as v
from ipywidgets.widgets import Widget
from ipywidgets import Layout, widgets

from antakia.data import ProjectedValues, DimReducMethod, ExplanationMethod
import antakia.compute as compute
from antakia.rules import Rule
from antakia.gui.widgets import get_widget, app_widget
import logging as logging
from antakia.utils import conf_logger

logger = logging.getLogger(__name__)
conf_logger(logger)

class HighDimExplorer:
    """
    An HighDimExplorer displays one or several high dim Dataframes on a scatter plot.
    It uses several dimension reduction techniques, through the DimReduction class.
    It can display in or 2 dimensions.

    Implemntation details :
    It handes projections computation itself when needed.
    But, it asks GUI when another dataframe is asked for.
    It stores dataframes with the ProjectedValues class.
    It stored the current projection method (in widget) but not the dimension
    Attributes are mostly privates (underscorred) since they are not meant to be used outside of the class.

    Attributes :
    pv_list: list # a list of one or several ProjectedValues (PV)
        Ex = [X values, imported values, commputed SHAP, computed LIME]
    current_pv : int, # where do wet get current displayed values from our pv_list
    _y : pd.Series
    _proj_params : dictionnary containing the parameters for the PaCMAP projection
        nested keys are "previous" / "current", then "VS" / "ES", then "n_neighbors" / "MN_ratio" / "FP_ratio"
    _current_dim
    _current_selection : list of X indexes # Plotly hack
    _has_lasso : bool # Plotly hack
    selection_changed : callable (from GUI)
    new_eplanation_values_required : callable (from GUI)

    Widgets :
    figure_2D and figure_3D : FigureWidget
        Plotly scatter plot
    _selection_disabled : bool
    container : a thin v.Container wrapper around the current Figure. Allows us to swap between 2D and 3D figures alone (without GUI)
    _proj_params_cards : dict of VBox,  parameters for dimreduc methods
    fig_size : int

    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        init_proj: int,
        init_dim: int,
        fig_size: int,
        border_size: int,
        selection_changed: callable,
        new_eplanation_values_required: callable = None,
        X_exp:pd.DataFrame = None,
    ):
        """
        Instantiate a new HighDimExplorer.

        Selected parameters :
            X : orignal dataset. Will be stored in a ProjectedValues oject
            X_exp : imported explained dataset. Idem.
            init_proj, init_dim : int, int, used to initialize widgets
        """
        if init_dim not in [2, 3]:
            raise ValueError(f"HDE.init: dim must be 2 or 3, not {init_dim}")
        self._current_dim = init_dim

        self.selection_changed = selection_changed
        self.new_eplanation_values_required = new_eplanation_values_required

        self.pv_list = []
        # Our pv_list is made of 1 (VS HDE) or 4 (ES HDE) items :
        # [X, imported values, computed SHAP, computed LIME]
        # We don't store dataframes but ProjectedValues objects
        # Item # 1 :
        self.pv_list.append(ProjectedValues(X))
        if X_exp is not None:
            # We are a ES HDE
            # We set the imported PV:
            # Item #2 (imported)
            if len(X_exp) > 0:
                self.pv_list.append(ProjectedValues(X_exp))
                self.current_pv = 1
            else:
                self.pv_list.append(None)
                self.current_pv = -1 # We have nothing to display yet
            # We set SHAP and LIME computed PV (to None for now):
            # Items 3 and 4
            self.pv_list.append(None)
            self.pv_list.append(None)
        else:
            self.current_pv = 0

        self._y = y

        self.get_projection_select().on_event("change", self.projection_select_changed)

        # We initiate it in grey, not indeterminate :
        self.get_projection_prog_circ().color = "grey"
        self.get_projection_prog_circ().indeterminate = False
        self.get_projection_prog_circ().v_model=100

        # Since HDE is responsible for storing its current proj, we check init value :
        if init_proj not in DimReducMethod.dimreduc_methods_as_list():
            raise ValueError(
                f"HDE.init: {init_proj} is not a valid projection method code"
            )
        self.get_projection_select().v_model = DimReducMethod.dimreduc_method_as_str(
            init_proj
        )
        # For each projection method, we store the widget (Card) that contains its parameters UI :
        self._proj_params_cards = {} # A dict of dict : keys are DimReducMethod, 'VS' or 'ES', then a dict of params
        self._proj_params = {} # A dict of dict of dict, see below. Nested keys
        # are 'DimReducMethod' (int), then 'previous' / 'current', then 'VS' / 'ES', then 'n_neighbors' / 'MN_ratio' / 'FP_ratio'
        # app_widget holds the UI for the PaCMAP params:

        self._proj_params_cards[DimReducMethod.PaCMAP] = get_widget(app_widget, "150" if self.is_value_space() else "180")
        # We init PaCMAP params for both sides
        self._proj_params[DimReducMethod.PaCMAP] =  {
            "previous": {
                "VS": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
                "ES": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
            },
            "current": {
                "VS": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
                "ES": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
            },
        }
        # We wire events on PaCMAP sliders only (for now):
        if self.is_value_space():
            get_widget(app_widget, "15000").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "15001").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "15002").on_event("change", self._proj_params_changed)
        else:
            get_widget(app_widget, "18000").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "18001").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "18002").on_event("change", self._proj_params_changed)

        if not self.is_value_space():
            self.get_explanation_select().items = [
                {"text": "Imported", "disabled": self.pv_list[1] is None},
                {"text": "SHAP", "disabled": True},
                {"text": "LIME", "disabled": True},
            ]
                    
            self.get_explanation_select().on_event("change", self.explanation_select_changed)
            self.update_explanation_select()

            # SHAP compute button :
            get_widget(app_widget, "13000203").on_event(
                "click", self.compute_btn_clicked
            )
            # LIME compute button :
            get_widget(app_widget, "13000303").on_event(
                "click", self.compute_btn_clicked
            )

        #  Now we can init figures 2 and 3D
        self.fig_size = fig_size
        self._selection_disabled = False

        self.container = v.Container()

        self.create_figure(2)
        self.create_figure(3)

        self._current_selection = []
        self._has_lasso = False

    # ---- Methods ------

    def disable_widgets(self, is_disabled: bool):
        """
        Called by GUI to enable/disable proj changes
        """
        self.get_projection_select().disabled = is_disabled
        self.get_proj_params_menu().disabled = is_disabled
        if not self.is_value_space :
            self.get_explanation_select().disabled = is_disabled
            self.get_compute_menu().disabled = is_disabled

    def display_rules(self, df_ids_list: list):
        """
        Draws the plots in blue / grey if they comply with the rules
        If non : draws back the dots with their original color
        """
        self._display_rules_one_side(df_ids_list, 2)
        self._display_rules_one_side(df_ids_list, 3)

    def _display_rules_one_side(self, df_ids_list: list, dim: int):

        fig = self.figure_2D if dim == 2 else self.figure_3D

        # We add a second trace (Scatter) to the figure to display the rules
        if df_ids_list is None or len(df_ids_list) == 0:
            # We remove the 'rule rules_trace' if exists
            if len(fig.data) > 1:
                if fig.data[1] is not None:
                    # It seems impossible to remove a trace from a figure once created
                    # So we hide or update&show this 'rules_trace'
                    fig.data[1].visible = False
        else:
            # Let's create a color scale with rule postives in blue and the rest in grey
            colors = ["grey"] * self.pv_list[self.current_pv].get_length()
            # IMPORTANT : we convert df_ids to row_ids (dataframe may not be indexed by row !!)
            row_ids_list = Rule.indexes_to_rows(
                self.pv_list[self.current_pv].X, df_ids_list
            )
            for row_id in row_ids_list:
                colors[row_id] = "blue"

            if len(fig.data) == 1:
                # We need to add a 'rules_trace'
                if dim == 2:
                    fig.add_trace(
                        Scatter(
                            x=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 2
                            )[0],
                            y=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 2
                            )[1],
                            mode="markers",
                            marker=dict(
                                color=colors,
                            ),
                        )
                    )
                else:
                    fig.add_trace(
                        Scatter3d(
                            x=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 3
                            )[0],
                            y=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 3
                            )[1],
                            z=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 3
                            )[2],
                            mode="markers",
                            marker=dict(
                                color=colors,
                            ),
                        )
                    )
            elif len(fig.data) == 2:
                # We replace the existing 'rules_trace'
                with fig.batch_update():
                    fig.data[1].x = self.pv_list[
                        self.current_pv
                    ].get_proj_values(self._get_projection_method(), dim)[0]
                    fig.data[1].y = self.pv_list[
                        self.current_pv
                    ].get_proj_values(self._get_projection_method(), dim)[1]
                    if dim == 3:
                        fig.data[1].z = self.pv_list[
                            self.current_pv
                        ].get_proj_values(self._get_projection_method(), dim)[2]
                    fig.layout.width = self.fig_size
                    fig.data[1].marker.color = colors

                fig.data[1].visible = True  # in case it was hidden
            else:
                raise ValueError(
                    f"HDE.{'VS' if len(self.pv_list) == 1 else 'ES'}.display_rules : to be debugged : the figure has {len(self.figure_2D.data)} traces, not 1 or 2"
                )

    def compute_projs(self, params_changed:bool=False, callback: callable = None):
        """
        If check if our projs (2 and 3D), are computed.
        NOTE : we only computes the values for _pv_list[self.current_pv]
        If needed, we compute them and store them in the PV
        The callback function may by GUI.update_splash_screen or HDE.update_progress_circular
        depending of the context.
        """

        if self.pv_list[self.current_pv] is None:
            projected_dots_2D = projected_dots_3D = None
        else:
            projected_dots_2D = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), 2
            )
            projected_dots_3D = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), 3
            )

        if params_changed:
            kwargs = {
                "n_neighbors": self._proj_params[self._get_projection_method()]["current"][self.get_space_name()]["n_neighbors"],
                "MN_ratio": self._proj_params[self._get_projection_method()]["current"][self.get_space_name()]["MN_ratio"],
                "FP_ratio": self._proj_params[self._get_projection_method()]["current"][self.get_space_name()]["FP_ratio"],
            }
        else:
            kwargs = {}


        if projected_dots_2D is None or params_changed:
            self.pv_list[self.current_pv].set_proj_values(
                self._get_projection_method(),
                2,
                compute.compute_projection(
                    self.pv_list[self.current_pv].X,
                    self._get_projection_method(),
                    2,
                    callback,
                    **kwargs
                ),
            )

            self.redraw_figure(self.figure_2D)

        if projected_dots_3D is None or params_changed:
            self.pv_list[self.current_pv].set_proj_values(
                self._get_projection_method(),
                3,
                compute.compute_projection(
                    self.pv_list[self.current_pv].X,
                    self._get_projection_method(),
                    3,
                    callback,
                    **kwargs
                ),
            )
            self.redraw_figure(self.figure_3D)

    def _proj_params_changed(self, widget, event, data):
        """
        Called when params slider changed"""
        # We disable the prooj params menu :
        self.get_proj_params_menu().disabled = True

        # We determine which param changed :
        if widget ==  get_widget(app_widget, "15000" if self.is_value_space() else "18000"):
            changed_param = 'n_neighbors'
        elif widget ==  get_widget(app_widget, "15001" if self.is_value_space() else "18001"):
            changed_param = 'MN_ratio'
        else:
            changed_param = 'FP_ratio'

        # We store previous value ...
        self._proj_params[self._get_projection_method()]["previous"][self.get_space_name()][changed_param] = self._proj_params[self._get_projection_method()]["current"][self.get_space_name()][changed_param]
        # .. and new value :
        self._proj_params[self._get_projection_method()]["current"][self.get_space_name()][changed_param] = data

        # We compute the PaCMAP new projection :
        self.compute_projs(True, self.update_progress_circular)  # to ensure we got the values
        self.redraw()

        self.get_proj_params_menu().disabled = False


    def update_progress_circular(
        self, caller, progress: int, duration: float
    ):
        """
        Each proj computation consists in 2 (2D and 3D) tasks.
        So progress of each task in divided by 2 and summed together
        """
        prog_circular = get_widget(app_widget, "16") if self.is_value_space() else get_widget(app_widget, "19")

        if prog_circular.color == "grey":
            prog_circular.color = "blue"
            # Since we don't have fine-grained progress, we set it to 'indeterminate'
            prog_circular.indeterminate = True
            # But i still need to store total progress in v_model :
            prog_circular.v_model = 0
            # We lock it during computation :
            prog_circular.disabled = True

        # Strange sicen we're in 'indeterminate' mode, but i need it, cf supra
        prog_circular.v_model = prog_circular.v_model + round(
            progress / 2
        )

        if prog_circular.v_model == 100:
            prog_circular.indeterminate = False
            prog_circular.color = "grey"
            prog_circular.disabled = False

    def projection_select_changed(self, widget, event, data):
        """ "
        Called when the user changes the projection method
        If needed, we compute the new projection
        """
        self.get_projection_select().disabled = True
        # We disable proj params if  not PaCMAP:
        self.get_proj_params_menu().disabled = self._get_projection_method() != DimReducMethod.PaCMAP
        self.compute_projs(False, self.update_progress_circular)  # to ensure we got the values
        self.get_projection_select().disabled = False
        self.redraw()

    def explanation_select_changed(self, widget, event, data):
        """
        Called when the user choses another dataframe
        """
        # Remember : impossible items ine thee Select are disabled = we have the desired values
       
        if data == "Imported":
            chosen_pv_index = 1
        elif data == "SHAP":
            chosen_pv_index = 2
        else: # LIME
            chosen_pv_index = 3
        self.current_pv = chosen_pv_index
        self.redraw()

    def compute_btn_clicked(self, widget, event, data):
        """
        Called  when new explanation computed values are wanted
        """
        # This compute btn is no longer useful / clickable
        widget.disabled = True

        if widget == get_widget(app_widget, "13000203"):
            desired_explain_method = ExplanationMethod.SHAP
        else:
            desired_explain_method = ExplanationMethod.LIME

        self.pv_list[2 if desired_explain_method == ExplanationMethod.SHAP else 3] = ProjectedValues(self.new_eplanation_values_required(desired_explain_method, self.update_progress_linear))

        self.current_pv = 2 if desired_explain_method == ExplanationMethod.SHAP else 3
        # We compute proj for this new PV :
        self.compute_projs(False, self.update_progress_circular)
        self.update_explanation_select()
        self.redraw_figure(self.figure_3D)

    def update_progress_linear(self, method: ExplanationMethod, progress: int, duration: float):
        """
        Called by the computation process (SHAP or LUME) to udpate the progress linear
        """
        
        if method.explanation_method == ExplanationMethod.SHAP:
            progress_linear = get_widget(app_widget, "13000201")
            progress_linear.indeterminate = True
        else:
            progress_linear = get_widget(app_widget, "13000301")

        progress_linear.v_model = progress

        if progress == 100:
            tab = None
            if method.explanation_method == ExplanationMethod.SHAP:
                tab = get_widget(app_widget, "130000")
                progress_linear.indeterminate = False
            else:
                tab = get_widget(app_widget, "130001")
                progress_linear.v_model= progress
            tab.disabled = True

    def set_dimension(self, dim: int):
        # Dimension is stored in the instance variable _current_dim
        """
        At runtime, GUI calls this function and swap our 2 and 3D figures
        """
        self._current_dim = dim
        self.container.children = [self.figure_2D] if dim == 2 else [self.figure_3D]

    def _get_projection_method(self) -> int:
        # proj is stored in the proj Select widget
        """
        Returns the current projection method
        """
        return DimReducMethod.dimreduc_method_as_int(
            self.get_projection_select().v_model
            )
    
    def _selection_event(self, trace, points, selector, *args):
        """Called whenever the user selects dots on the scatter plot"""
        # We don't call GUI.selection_changed if 'selectedpoints' length is 0 : it's handled by -deselection_event
        
        if len(points.point_inds) > 0:
            # NOTE : Plotly doesn't allow to show selection on Scatter3d
            self._has_lasso = True
            
            # We tell the GUI
            # NOTE : here we convert row ids to dataframe indexes
            self.selection_changed(
                self,
                Rule.rows_to_indexes(self.get_current_X(), points.point_inds))

        self._current_selection = points.point_inds

    def _deselection_event(self, trace, points, append: bool = False):
        """Called on deselection"""
        # We tell the GUI
        self._current_selection = []
        self._has_lasso = False
        self.selection_changed(self, [])

    def set_selection(self, new_selection_indexes: list):
        """
        Called by tne UI when a new selection occured on the other HDE
        """

        if len(self._current_selection) == 0 and len(new_selection_indexes) == 0:
            logger.debug(f"HDE.set_sel : {self.get_space_name()} ignore another unselect.")
            return

        if len(self._current_selection) > 0 and len(new_selection_indexes) == 0:
            logger.debug(f"HDE.set_sel : {self.get_space_name()} forced to unselect")
            self._current_selection = []
            # We have to rebuild our figure:
            self.create_figure(2)
            return
        
        if self._has_lasso:
            logger.debug(f"HDE.set_sel : {self.get_space_name()} had the lasso but receives set_sel.")
            # We don't have lasso anymore
            self._has_lasso = False
            # We have to rebuild our figure:
            self.create_figure(2)
            self.figure_2D.data[0].selectedpoints=Rule.indexes_to_rows(self.get_current_X(), new_selection_indexes)
            self._current_selection = new_selection_indexes
            return

        # We set the new selection on our figures :
        self.figure_2D.update_traces(selectedpoints=Rule.indexes_to_rows(self.get_current_X(), new_selection_indexes))

        # We store the new selection :
        self._current_selection = new_selection_indexes

    def create_figure(self, dim: int):
        """
        Called by __init__ and by set_selection
        Builds the FigureWidget for the given dimension
        """

        x = y = z = None

        if self.pv_list[self.current_pv] is not None:            
            proj_values = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), dim
            )
            if proj_values is not None:
                x = self.pv_list[self.current_pv].get_proj_values(
                    self._get_projection_method(), dim
                )[0]
                y = self.pv_list[self.current_pv].get_proj_values(
                    self._get_projection_method(), dim
                )[1]
                if dim == 3:
                    z = self.pv_list[self.current_pv].get_proj_values(
                        self._get_projection_method(), dim
                    )[2]

        hde_marker = dict(color=self._y, colorscale="Viridis")

        if dim == 2:
            fig = FigureWidget(
                data=Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=hde_marker,
                    customdata=self._y,
                    hovertemplate="%{customdata:.3f}",
                )
            )
        else:
            fig = FigureWidget(
                data=Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=hde_marker,
                    customdata=self._y,
                    hovertemplate="%{customdata:.3f}",
                )
            )

        fig.update_layout(dragmode=False if self._selection_disabled else "lasso")
        fig.update_traces(
            selected={"marker": {"opacity": 1.0}},
            unselected={"marker": {"opacity": 0.1}},
            selector=dict(type="scatter"),
        )
        fig.update_layout(margin=dict(t=0), width=self.fig_size)
        fig._config = fig._config | {"displaylogo": False}
        fig._config = fig._config | {'displayModeBar': True}

        if dim == 2:
            self.figure_2D = fig
            self.figure_2D.data[0].on_selection(self._selection_event)
            self.figure_2D.data[0].on_deselect(self._deselection_event)
        else:
            self.figure_3D = fig
        
        self.container.children = [self.figure_2D if self._current_dim == 2 else self.figure_3D]


    def redraw(self, color: pd.Series = None, opacity_values: pd.Series = None):
        """
        Redraws the 2D and 3D figures. FigureWidgets are not recreated.
        """
        self.redraw_figure(self.figure_2D, color)
        self.redraw_figure(self.figure_3D, color)

    def redraw_figure(
        self,
        fig: FigureWidget,
        color: pd.Series = None,
        opacity_values: pd.Series = None
    ):

        dim = (
            2 if isinstance(fig.data[0],Scatter) else 3
        )  # dont' use self._current_dim: it may be 3D while we want to redraw figure_2D


        x = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), dim)[0]
        y = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), dim)[1]
        if dim == 3:
            z = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), dim)[2]

        with fig.batch_update():
            fig.data[0].x = x
            fig.data[0].y = y
            if dim == 3 :
                fig.data[0].z = z
            fig.layout.width = self.fig_size
            if color is not None:
                fig.data[0].marker.color = color
            if opacity_values is not None:
                fig.data[0].marker.opacity = opacity_values
            fig.data[0].customdata = color

    def get_projection_select(self):
        """
       Called at startup by the GUI
       """
        return get_widget(app_widget, "14") if self.is_value_space() else get_widget(app_widget, "17")

    def get_projection_prog_circ(self) -> v.ProgressCircular:
        """
       Called at startup by the GUI
       """
        return get_widget(app_widget, "16") if self.is_value_space() else get_widget(app_widget, "19")

    def get_compute_menu(self):
        """
       Called at startup by the GUI (only ES HDE)
       """
        return get_widget(app_widget, "13")

    def get_explanation_select(self):
        """
       Called at startup by the GUI (only ES HE)
       """
        return get_widget(app_widget, "12")
    
    def update_explanation_select(self):
        """
       Called at startup by the GUI (only ES HE)
       """
        self.get_explanation_select().items = [
                {"text": "Imported", "disabled": self.pv_list[1] is None},
                {"text": "SHAP", "disabled": self.pv_list[2] is None},
                {"text": "LIME", "disabled": self.pv_list[3] is None},
            ]

    def get_proj_params_menu(self):
        """
        Called at startup by the GUI
        """
        # We return
        proj_params_menu = get_widget(app_widget, "15") if self.is_value_space() else get_widget(app_widget, "18")
        # We neet to set a Card, depending on the projection method
        if self._get_projection_method() == DimReducMethod.PaCMAP:
            proj_params_menu.children=[self._proj_params_cards[DimReducMethod.PaCMAP]]
        proj_params_menu.disabled = self._get_projection_method() != DimReducMethod.PaCMAP

        return proj_params_menu
    
    def is_value_space(self) -> bool:
        return len(self.pv_list) == 1

    def get_space_name(self) -> str:
        """
        For debug purposes only. Not very reliable.
        """
        return "VS" if self.is_value_space() else "ES"

    def get_current_X(self) -> pd.DataFrame:
        return self.pv_list[self.current_pv].X
    