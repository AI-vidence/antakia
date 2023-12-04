from __future__ import annotations

import pandas as pd
import numpy as np
from plotly.graph_objects import FigureWidget, Scattergl, Scatter3d
import ipyvuetify as v
from sklearn.neighbors import KNeighborsClassifier

from antakia.compute.dim_reduction.dim_reduction import compute_projection
from antakia.compute.explanation.explanation_method import ExplanationMethod
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.data_handler.region import RegionSet

from antakia.gui.widgets import get_widget, app_widget

import antakia.utils.utils as utils
import antakia.config as config
from antakia.data_handler.projected_values import ProjectedValues

import logging as logging
from antakia.utils.logging import conf_logger

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
    pv_dict : dict(ProjectedValues) # a dict of one or several ProjectedValues (PV)
        Keys are : 'original_values', 'imported_explanations', 'computed_shap', 'computed_lime'
    current_pv : str in ['original_values', 'imported_explanations', 'computed_shap', 'computed_lime']
        tells wich PV is currently displayed
    is_value_space : bool
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
    container : a thin v.Container wrapper around the current Figure. Allows us to swap between 2D and 3D figures alone (without GUI)
    _proj_params_cards : dict of VBox,  parameters for dimreduc methods
    fig_size : int

    """

    def __init__(
            self,
            X: pd.DataFrame,  # The original_values
            y: pd.Series,
            init_proj: int,  # see config.py
            init_dim: int,
            fig_size: int,
            selection_changed: callable,
            new_eplanation_values_required: callable = None,  # (ES only)
            X_exp: pd.DataFrame = None,  # The imported_explanations (ES only)
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

        self._mask = None
        self.selection_changed = selection_changed
        self.new_eplanation_values_required = new_eplanation_values_required

        # IMPORTANT : if x_exp is not None : we know it's an ES HDE
        self.is_value_space = X_exp is None

        # pv_dict is a dict of ProjectedValues objects
        # Keys can be : 'original_values', 'imported_explanations', 'computed_shap', 'computed_lime'
        # VS HDE has as only on PV, pv_divt['original_values']
        # ES HDE has 3 extra PVs : 'imported_explanations', 'computed_shap', 'computed_lime'
        if not self.is_value_space:
            self.pv_dict = {
                'original_values': ProjectedValues(X),
                'imported_explanations': None,
                'computed_shap': None,
                'computed_lime': None
            }
            if len(X_exp) > 0:
                # We set the imported PV:
                self.pv_dict['imported_explanations'] = ProjectedValues(X_exp)
                self.current_pv = 'imported_explanations'
            else:
                self.pv_dict['imported_explanations'] = None
                self.current_pv = None  # We have nothing to display yet
            self.pv_dict['computed_shap'] = None
            self.pv_dict['computed_lime'] = None
        else:
            self.pv_dict = {
                'original_values': ProjectedValues(X),
            }
            # We are a VS HDE
            self.current_pv = 'original_values'

        self._y = y

        self.get_projection_select().on_event("change", self.projection_select_changed)

        # We initiate it in grey, not indeterminate :
        self.get_projection_prog_circ().color = "grey"
        self.get_projection_prog_circ().indeterminate = False
        self.get_projection_prog_circ().v_model = 100

        # Since HDE is responsible for storing its current proj, we check init value :
        if init_proj not in DimReducMethod.dimreduc_methods_as_list():
            raise ValueError(
                f"HDE.init: {init_proj} is not a valid projection method code"
            )
        self.get_projection_select().v_model = DimReducMethod.dimreduc_method_as_str(
            init_proj
        )
        # For each projection method, we store the widget (Card) that contains its parameters UI :
        self._proj_params_cards = {}  # A dict of dict : keys are DimReducMethod, 'VS' or 'ES', then a dict of params
        self._proj_params = {}  # A dict of dict of dict, see below. Nested keys
        # are 'DimReducMethod' (int), then 'previous' / 'current', then 'VS' / 'ES', then 'n_neighbors' / 'MN_ratio' / 'FP_ratio'
        # app_widget holds the UI for the PaCMAP params:

        self._proj_params_cards[DimReducMethod.dimreduc_method_as_int('PaCMAP')] = get_widget(app_widget,
                                                                                              "150" if self.is_value_space else "180")
        # We init PaCMAP params for both sides
        self._proj_params[DimReducMethod.dimreduc_method_as_int('PaCMAP')] = {
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
        if self.is_value_space:
            get_widget(app_widget, "15000").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "15001").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "15002").on_event("change", self._proj_params_changed)
        else:
            get_widget(app_widget, "18000").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "18001").on_event("change", self._proj_params_changed)
            get_widget(app_widget, "18002").on_event("change", self._proj_params_changed)

        if not self.is_value_space:
            self.update_explanation_select()
            self.get_explanation_select().on_event("change", self.explanation_select_changed)

            get_widget(app_widget, "13000203").on_event("click", self.compute_btn_clicked)
            get_widget(app_widget, "13000303").on_event("click", self.compute_btn_clicked)
            self.update_compute_menu()

        #  Now we can init figures 2 and 3D
        self.fig_size = fig_size
        self._selection_disabled = False

        self.container = v.Container()
        self.container.class_ = "flex-fill"

        self.create_figure(2)
        self.create_figure(3)

        self._current_selection = pd.Series([False] * len(X), index=X.index)
        self._has_lasso = False

    # ---- Methods ------

    def disable_selection(self, is_disabled: bool):
        self.figure_2D.update_layout(
            dragmode=False if is_disabled else "lasso"
        )

    def disable_widgets(self, is_disabled: bool):
        """
        Called by GUI to enable/disable proj changes
        """
        self.get_projection_select().disabled = is_disabled
        self.get_proj_params_menu().disabled = is_disabled
        if not self.is_value_space:
            self.get_explanation_select().disabled = is_disabled
            self.get_compute_menu().disabled = is_disabled

    def display_rules(self, mask: pd.Series | None, color='blue'):
        """"
        Displays the dots corresponding to our current rules in blue, the others in grey
        """
        rs = RegionSet(self.current_X)
        rs.add_region(mask=mask, color=color)
        self._display_zones(rs, 1)

    def display_regions(self, region_set: RegionSet):
        """"
        Displays each region in a different color
        """
        self._display_zones(region_set, 2)

    def _display_zones(self, region_set: RegionSet, trace_id):
        """
        Paint on our extra scatter one or more zones (list of pv_list[0].X indexes) using
        the passed colors. Zones may be regions or rules
        if index_list is None, we restore the original color
        Common method for display_rules and display_regions
        """
        # We use two extra traces in the figure : the rules and the regions traces

        # We detect the trace of the figure we'll paint on

        if len(region_set) == 0 or not region_set.get(1).mask.any():
            # We need to clean the trace - we just hide it
            self.figure_2D.data[trace_id].visible = False
            self.figure_3D.data[trace_id].visible = False
            # And we're done
            return

        def _display_zone_on_figure(fig: FigureWidget, trace_id: int, colors: pd.Series):
            """
            Draws one zone on one figure using the passed colors
            """
            dim = 2 if isinstance(fig.data[0], Scattergl) else 3

            values = self.get_current_X_proj(dim)
            colors = colors[self.mask]

            x = values[0]
            y = values[1]
            if dim == 3:
                z = values[2]
            else:
                z = None

            with fig.batch_update():
                fig.data[trace_id].x = x
                fig.data[trace_id].y = y
                if dim == 3:
                    fig.data[trace_id].z = z
                fig.layout.width = self.fig_size
                fig.data[trace_id].marker.color = colors
            fig.data[trace_id].visible = True  # in case it was hidden

        # List of color names, 1 per point. Initialized to grey
        colors = region_set.get_color_serie()

        _display_zone_on_figure(self.figure_2D, trace_id, colors)
        _display_zone_on_figure(self.figure_3D, trace_id, colors)

    def compute_projs(self, params_changed: bool = False, callback: callable = None):
        """
        If check if our projs (2 and 3D), are computed.
        NOTE : we only computes the values for _pv_list[self.current_pv]
        If needed, we compute them and store them in the PV
        The callback function may by GUI.update_splash_screen or HDE.update_progress_circular
        depending of the context.
        """

        if self.current_pv is None or self.pv_dict[self.current_pv] is None:
            projected_dots_2D = projected_dots_3D = None
        else:
            projected_dots_2D = self.get_current_X_proj(2)
            projected_dots_3D = self.get_current_X_proj(3)

        if params_changed:
            kwargs = self._proj_params[self._get_projection_method()]["current"][self.get_space_name()]
        else:
            kwargs = {}

        if projected_dots_2D is None or params_changed:
            self.pv_dict[self.current_pv].set_proj_values(
                self._get_projection_method(),
                2,
                compute_projection(
                    self.pv_dict[self.current_pv].X,
                    self._y,
                    self._get_projection_method(),
                    2,
                    callback,
                    **kwargs
                ),
            )

            self.redraw_figure(self.figure_2D)

        if projected_dots_3D is None or params_changed:
            self.pv_dict[self.current_pv].set_proj_values(
                self._get_projection_method(),
                3,
                compute_projection(
                    self.pv_dict[self.current_pv].X,
                    self._y,
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
        if widget == get_widget(app_widget, "15000" if self.is_value_space else "18000"):
            changed_param = 'n_neighbors'
        elif widget == get_widget(app_widget, "15001" if self.is_value_space else "18001"):
            changed_param = 'MN_ratio'
        else:
            changed_param = 'FP_ratio'

        # We store previous value ...
        self._proj_params[self._get_projection_method()]["previous"][self.get_space_name()][changed_param] = \
            self._proj_params[self._get_projection_method()]["current"][self.get_space_name()][changed_param]
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
        prog_circular = get_widget(app_widget, "16") if self.is_value_space else get_widget(app_widget, "19")

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
        self.get_proj_params_menu().disabled = self._get_projection_method() != DimReducMethod.dimreduc_method_as_int(
            'PaCMAP')
        self.compute_projs(False, self.update_progress_circular)  # to ensure we got the values
        self.get_projection_select().disabled = False
        self.redraw()

    def explanation_select_changed(self, widget, event, data):
        """
        Called when the user choses another dataframe
        """
        # Remember : impossible items ine thee Select are disabled = we have the desired values

        if data == "Imported":
            self.current_pv = 'imported_explanations'
        elif data == "SHAP":
            self.current_pv = 'computed_shap'
        else:  # LIME
            self.current_pv = 'computed_lime'
        self.redraw()

    def get_compute_menu(self):
        """
       Called at startup by the GUI (only ES HDE)
       """
        return get_widget(app_widget, "13")

    def update_compute_menu(self):
        we_have_computed_shap = self.pv_dict['computed_shap'] is not None
        get_widget(app_widget, "130000").disabled = we_have_computed_shap
        get_widget(app_widget, "13000203").disabled = we_have_computed_shap

        we_have_computed_lime = self.pv_dict['computed_lime'] is not None
        get_widget(app_widget, "130001").disabled = we_have_computed_lime
        get_widget(app_widget, "13000303").disabled = we_have_computed_lime

    def compute_btn_clicked(self, widget, event, data):
        """
        Called when new explanation computed values are wanted
        """
        # This compute btn is no longer useful / clickable
        widget.disabled = True

        if widget == get_widget(app_widget, "13000203"):
            desired_explain_method = ExplanationMethod.SHAP
        else:
            desired_explain_method = ExplanationMethod.LIME

        self.current_pv = 'computed_shap' if desired_explain_method == ExplanationMethod.SHAP else 'computed_lime'
        self.pv_dict[self.current_pv] = ProjectedValues(
            self.new_eplanation_values_required(desired_explain_method, self.update_progress_linear))

        # We compute proj for this new PV :
        self.compute_projs(False, self.update_progress_circular)
        self.update_explanation_select()
        self.update_compute_menu()
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
                progress_linear.v_model = progress
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

        # We convert selected rows in mask
        self._current_selection = utils.rows_to_mask(self.pv_dict['original_values'].X, points.point_inds)

        if self._current_selection.any():
            # NOTE : Plotly doesn't allow to show selection on Scatter3d
            self._has_lasso = True

            # We tell the GUI
            # NOTE : here we convert row ids to dataframe indexes
            self.selection_changed(self, self._current_selection)

    def _deselection_event(self, trace, points, append: bool = False):
        """Called on deselection"""
        # We tell the GUI
        self._current_selection = utils.rows_to_mask(self.pv_dict['original_values'].X, [])
        self._has_lasso = False
        self.selection_changed(self, self._current_selection)

    def set_selection(self, new_selection_mask: pd.Series):
        """
        Called by tne UI when a new selection occured on the other HDE
        """

        if not self._current_selection.any() and not new_selection_mask.any():
            # New selection is empty. We already have an empty selection : nothing to do
            return

        if not new_selection_mask.any():
            self._current_selection = new_selection_mask
            # We have to rebuild our figure:
            self.create_figure(2)
            return

        if self._has_lasso:
            # We don't have lasso anymore
            self._has_lasso = False
            # We have to rebuild our figure:
            self.create_figure(2)
            self.figure_2D.data[0].selectedpoints = utils.mask_to_rows(new_selection_mask)
            self._current_selection = new_selection_mask
            return

        # We set the new selection on our figures :
        self.figure_2D.update_traces(selectedpoints=utils.mask_to_rows(new_selection_mask))
        # We store the new selection
        self._current_selection = new_selection_mask

    def create_figure(self, dim: int):
        """
        Called by __init__ and by set_selection
        Builds the FigureWidget for the given dimension
        """
        x = y = z = None

        if self.current_X is not None:
            proj_values = self.get_current_X_proj(dim)
            if proj_values is not None:
                x = proj_values[0]
                y = proj_values[1]
                if dim == 3:
                    z = proj_values[2]

        hde_marker = None
        if dim == 2:
            if self.is_value_space:
                hde_marker = dict(
                    color=self._y,
                    colorscale="Viridis",
                    # colorbar=
                    # dict(
                    #     thickness=20
                    # )
                )
            else:
                hde_marker = dict(color=self._y, colorscale="Viridis")
        else:
            if self.is_value_space:
                hde_marker = dict(color=self._y,
                                  colorscale="Viridis",
                                  #   colorbar=dict(
                                  #       thickness=20),
                                  size=2)
            else:
                hde_marker = dict(color=self._y, colorscale="Viridis", size=2)

        if dim == 2:
            fig = FigureWidget(
                data=[
                    Scattergl(  # Trace 0 for dots
                        x=x,
                        y=y,
                        mode="markers",
                        marker=hde_marker,
                        customdata=self._y,
                        hovertemplate="%{customdata:.3f}",
                    )
                ]
            )
            fig.add_trace(
                Scattergl(  # Trace 1 for rules
                    x=x,
                    y=y,
                    mode="markers",
                    marker=hde_marker,
                    customdata=self._y,
                    hovertemplate="%{customdata:.3f}",
                )
            )
            fig.add_trace(
                Scattergl(  # Trace 2 for regions
                    x=x,
                    y=y,
                    mode="markers",
                    marker=hde_marker,
                    customdata=self._y,
                )
            )
        else:
            fig = FigureWidget(
                data=[
                    Scatter3d(  # Trace 0 for dots
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=hde_marker,
                        customdata=self._y,
                        hovertemplate="%{customdata:.3f}",
                    )
                ]
            )
            fig.add_trace(
                Scatter3d(  # Trace 1 for rules
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=hde_marker,
                    customdata=self._y,
                    hovertemplate="%{customdata:.3f}",
                )
            )
            fig.add_trace(
                Scatter3d(  # Trace 2 for regions
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
        fig.update_layout(
            margin=dict(
                t=0,
                b=0,
                l=0,
                r=0
            ),
            width=self.fig_size,
            height=round(self.fig_size / 2),
        )
        fig._config = fig._config | {"displaylogo": False}
        fig._config = fig._config | {'displayModeBar': True}

        if dim == 2:
            self.figure_2D = fig
            self.figure_2D.data[0].on_selection(self._selection_event)
            self.figure_2D.data[0].on_deselect(self._deselection_event)
        else:
            self.figure_3D = fig

        self.container.children = [self.figure_2D if self._current_dim == 2 else self.figure_3D]

    def redraw(self, color: pd.Series = None):
        """
        Redraws the 2D and 3D figures. FigureWidgets are not recreated.
        """
        self.redraw_figure(self.figure_2D, color)
        self.redraw_figure(self.figure_3D, color)

    def redraw_figure(
            self,
            fig: FigureWidget,
            color: pd.Series = None
    ):

        dim = (
            2 if isinstance(fig.data[0], Scattergl) else 3
        )  # dont' use self._current_dim: it may be 3D while we want to redraw figure_2D

        projection = self.get_current_X_proj(dim)
        if color is not None:
            color = color.loc[self.mask]
        x = projection[0]
        y = projection[1]
        if dim == 3:
            z = projection[2]

        with fig.batch_update():
            fig.data[0].x = x
            fig.data[0].y = y
            if dim == 3:
                fig.data[0].z = z
            fig.layout.width = self.fig_size
            if color is not None:
                fig.data[0].marker.color = color
            fig.data[0].customdata = color

    def get_projection_select(self):
        """
       Called at startup by the GUI
       """
        return get_widget(app_widget, "14") if self.is_value_space else get_widget(app_widget, "17")

    def get_projection_prog_circ(self) -> v.ProgressCircular:
        """
       Called at startup by the GUI
       """
        return get_widget(app_widget, "16") if self.is_value_space else get_widget(app_widget, "19")

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
            {"text": "Imported", "disabled": self.pv_dict['imported_explanations'] is None},
            {"text": "SHAP", "disabled": self.pv_dict['computed_shap'] is None},
            {"text": "LIME", "disabled": self.pv_dict['computed_lime'] is None},
        ]

    def get_proj_params_menu(self):
        """
        Called at startup by the GUI
        """
        # We return
        proj_params_menu = get_widget(app_widget, "15") if self.is_value_space else get_widget(app_widget, "18")
        # We neet to set a Card, depending on the projection method
        if self._get_projection_method() == DimReducMethod.dimreduc_method_as_int('PaCMAP'):
            proj_params_menu.children = [self._proj_params_cards[DimReducMethod.dimreduc_method_as_int('PaCMAP')]]
        proj_params_menu.disabled = self._get_projection_method() != DimReducMethod.dimreduc_method_as_int('PaCMAP')

        return proj_params_menu

    def get_space_name(self) -> str:
        """
        For debug purposes only. Not very reliable.
        """
        return "VS" if self.is_value_space else "ES"

    @property
    def current_X(self) -> pd.DataFrame | None:
        if self.current_pv is None:
            return None  # When we're an ES HDE and no explanation have been importer nor computed yet
        return self.pv_dict[self.current_pv].X

    @property
    def mask(self):
        if self._mask is None:
            X = self.current_X
            self._mask = pd.Series([False] * len(X), index=X.index)
            limit = config.MAX_DOTS
            if len(X) > limit:
                indices = np.random.choice(X.index, size=limit, replace=False)
                self._mask.loc[indices] = True
            else:
                self._mask.loc[:] = True
        return self._mask

    def get_current_X_proj(self, dim: int, masked: bool = True) -> pd.DataFrame | None:
        X = self.pv_dict[self.current_pv].get_proj_values(self._get_projection_method(), dim)
        if X is None:
            return
        if masked:
            return X.loc[self.mask]
        return self.pv_dict[self.current_pv].get_proj_values(self._get_projection_method(), dim)

    def selection_to_mask(self, row_numbers, dim):
        selection = utils.rows_to_mask(self.pv_dict['original_values'].X.loc[self.mask], row_numbers)
        X_train = self.get_current_X_proj(dim)
        knn = KNeighborsClassifier().fit(X_train, selection)
        X_predict = self.get_current_X_proj(dim, masked=False)
        guessed_selection = pd.Series(knn.predict(X_predict), index=X_predict.index)
        print('mean', selection.mean(), guessed_selection.mean())
        print('len', len(selection), len(guessed_selection))
        # KNN extrapolation
        return guessed_selection.astype(bool)
