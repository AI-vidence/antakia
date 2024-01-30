from __future__ import annotations

import pandas as pd
from ipywidgets import widgets
import numpy as np
from plotly.graph_objects import FigureWidget, Scattergl, Scatter3d
import ipyvuetify as v
from sklearn.neighbors import KNeighborsClassifier

from antakia.compute.dim_reduction.dim_reduction import dim_reduc_factory
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.data_handler.region import Region, RegionSet

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
    But, it asks GUI when another dataframe is asked for (eg. compute SHAP or LIME)
    It stores dataframes with the ProjectedValues class.
    It stored the current projection method (in widget) but not the dimension (see _current_dim)
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

    # Trace indexes : 0 for values, 1 for rules, 2 for regions
    NUM_TRACES = 4
    VALUES_TRACE = 0
    RULES_TRACE = 1
    REGIONSET_TRACE = 2
    REGION_TRACE = 3

    @staticmethod
    def trace_name(trace_id: int) -> str:
        if trace_id == HighDimExplorer.VALUES_TRACE:
            return 'values trace'
        elif trace_id == HighDimExplorer.RULES_TRACE:
            return 'rules trace'
        elif trace_id == HighDimExplorer.REGIONSET_TRACE:
            return 'regionset trace'
        elif trace_id == HighDimExplorer.REGION_TRACE:
            return 'region trace'
        else:
            return "unknox trace"

    def __init__(
            self,
            pv: ProjectedValues,
            init_dim: int,
            fig_size: int,
            selection_changed: callable,
            space_type: str
    ):
        """
        Instantiate a new HighDimExplorer.

        Selected parameters :
            X : orignal dataset. Will be stored in a ProjectedValues oject
            X_exp : imported explained dataset. Idem.
            init_proj, init_dim : int, int, used to initialize widgets
        """
        if space_type not in ['ES', 'VS']:
            raise ValueError(f"HDE.init: space_type must be 'ES' or 'VS', not {space_type}")
        self.is_value_space = space_type == 'VS'
        if init_dim not in [2, 3]:
            raise ValueError(f"HDE.init: dim must be 2 or 3, not {init_dim}")
        self._current_dim = init_dim
        self.active_tab = 0

        self._mask = None
        self.selection_changed = selection_changed
        self.pv = pv

        # current projected values to display

        # For each projection method, we store the widget (Card) that contains its parameters UI :
        self._proj_params_cards = {}
        self.build_all_proj_param_w()
        self.update_proj_params_menu()

        #  Now we can init figure
        self.figure_container = v.Container()
        self.figure_container.class_ = "flex-fill"

        self.fig_width = fig_size
        self.fig_height = fig_size / 2

        self._selection_disabled = False
        if pv is not None:
            self._current_selection = utils.boolean_mask(pv.X, True)
        else:
            self._current_selection = None
        self.first_selection = False

        # traces to show
        self._visible = [True, False, False, False]
        # trace_colors
        self._colors: list[pd.Series | None] = [None, None, None, None]

        self.figure_2D = self.figure_3D = None

    def initialize(self, progress_callback):
        self.update_pv(self.pv, progress_callback)
        self.close_progress_circular()
        self.get_projection_select().on_event("change", self.projection_select_changed)

    ############
    ## widget ##
    ############
    def disable_widgets(self, is_disabled: bool):
        """
        Called by GUI to enable/disable proj changes and explaination computation or change
        """
        self.get_projection_select().disabled = is_disabled
        self.enable_proj_param_menu(not is_disabled)

    ################
    ## properties ##
    ################
    @property
    def current_dim(self):
        return self._current_dim

    @current_dim.setter
    def current_dim(self, dim):
        self._current_dim = dim
        # we recreate figure on dim change
        self.create_figure()

    @property
    def figure(self):
        if self.current_dim == 2:
            return self.figure_2D
        return self.figure_3D

    @figure.setter
    def figure(self, fig):
        if self.current_dim == 2:
            self.figure_2D = fig
        else:
            self.figure_3D = fig

    @property
    def current_projected_values(self):
        return self.pv

    # --- select projection method ---

    def projection_select_changed(self, widget, event, data):
        """ "
        Called when the user changes the projection method
        If needed, we compute the new projection
        """
        self.get_projection_select().disabled = True
        # We disable proj params if  not PaCMAP:

        self.update_proj_params_menu()
        self.redraw()
        self.get_projection_select().disabled = False

    def get_projection_prog_circ(self) -> v.ProgressCircular:
        """
       Called at startup by the GUI
       """
        return get_widget(app_widget, "16" if self.is_value_space else "19")

    def update_progress_circular(
            self, caller, progress: int, duration: float
    ):
        """
        Each proj computation consists in 2 (2D and 3D) tasks.
        So progress of each task in divided by 2 and summed together
        """
        prog_circular = self.get_projection_prog_circ()
        if prog_circular.color == "grey":
            prog_circular.color = "blue"
            # Since we don't have fine-grained progress, we set it to 'indeterminate'
            prog_circular.indeterminate = True
            # But i still need to store total progress in v_model :
            prog_circular.v_model = 0
            # We lock it during computation :
            prog_circular.disabled = True

        # Strange sicen we're in 'indeterminate' mode, but i need it, cf supra
        prog_circular.v_model = float(prog_circular.v_model) + round(progress)

        if prog_circular.v_model == 100:
            self.close_progress_circular()

    def close_progress_circular(self):
        prog_circular = self.get_projection_prog_circ()
        prog_circular.indeterminate = False
        prog_circular.color = "grey"
        prog_circular.disabled = False

    # ---- Projection parameters ---

    @property
    def proj_param_widget(self):
        if self.is_value_space:
            return '1500'
        return '1800'

    def projection_kwargs(self, dim_reduc_method):
        kwargs = self.pv.get_paramerters(dim_reduc_method, self.current_dim)['current']
        return kwargs

    def build_proj_param_widget(self, dim_reduc) -> list[v.Slider]:
        parameters = dim_reduc_factory[dim_reduc].parameters()
        sliders = []
        for param, info in parameters.items():
            min_, max_, step = utils.compute_step(info['min'], info['max'])
            if self.pv is not None:
                current_value = self.projection_kwargs(dim_reduc)[param]
            else:
                current_value = info['default']
            if info['type'] == int:
                step = max(round(step), 1)

            slider = v.Slider(  # 15000
                class_="ma-8 pa-2",
                v_model=current_value,
                min=float(min_),
                max=float(max_),
                step=step,
                label=param,
                thumb_label="always",
                thumb_size=25,
            )
            slider.on_event("change", self._proj_params_changed)
            sliders.append(slider)
        return sliders

    def build_all_proj_param_w(self):
        for dim_reduc in DimReducMethod.dimreduc_methods_as_list():
            self._proj_params_cards[dim_reduc] = self.build_proj_param_widget(dim_reduc)

    def _proj_params_changed(self, widget, event, data):
        """
        Called when params slider changed"""
        # We disable the prooj params menu :
        self.enable_proj_param_menu(False)

        changed_param = widget.label
        # We compute the PaCMAP new projection :
        self.pv.set_parameters(self._get_projection_method(), self.current_dim,
                               {changed_param: data})
        self.redraw()

        self.enable_proj_param_menu(True)

    def update_proj_params_menu(self):
        """
        Called at startup by the GUI
        """
        # We return
        proj_params_menu = get_widget(app_widget, "150" if self.is_value_space else "180")
        params = self._proj_params_cards[self._get_projection_method()]
        if len(params) == 0:
            proj_params_menu.disabled = True
        # We neet to set a Card, depending on the projection method
        proj_params_menu.children = [widgets.VBox(params)]
        # proj_params_menu.disabled = self._get_projection_method() != DimReducMethod.dimreduc_method_as_int('PaCMAP')

        return proj_params_menu

    def enable_proj_param_menu(self, enable):
        proj_params_menu = get_widget(app_widget, "15" if self.is_value_space else "18")
        proj_params_menu.disabled = not enable

        return proj_params_menu

    # ---- display Methods ------

    def disable_selection(self, is_disabled: bool):
        self._selection_disabled = is_disabled
        if self.figure_2D is not None:
            self.figure_2D.update_layout(
                dragmode=False if is_disabled else "lasso"
            )

    def show_trace(self, trace_id: int, show: bool):
        self._visible[trace_id] = show
        self.figure.data[trace_id].visible = show

    def display_rules(self, mask: pd.Series | None = None, color='blue'):
        """"
        Displays the dots corresponding to our current rules in blue, the others in grey
        """
        rs = RegionSet(self.current_X)
        if mask is None:
            self._colors[self.RULES_TRACE] = None
        else:
            rs.add_region(mask=mask, color=color)
            self._colors[self.RULES_TRACE] = rs.get_color_serie()

        self._display_zones(self.RULES_TRACE)

    def display_regionset(self, region_set: RegionSet):
        """"
        Displays each region in a different color
        """
        self._colors[self.REGIONSET_TRACE] = region_set.get_color_serie()
        self._display_zones(self.REGIONSET_TRACE)

    def display_region(self, region: Region):
        rs = RegionSet(self.current_X)
        rs.add(region)
        self._colors[self.REGION_TRACE] = rs.get_color_serie()
        self._display_zones(self.REGION_TRACE)

    def _display_zones(self, trace=None):
        """
        Paint on our extra scatter one or more zones using
        the passed colors. Zones may be region(s) or rules
        if index_list is None, we restore the original color
        Common method for display_rules and display_regions
        trace_id : (0 for default scatter plot) 1 for 'rules in progress' and 2 for 'regions' and 3 for 'region'
        """

        # We use three extra traces in the figure : the rules, the regions and the region traces (1, 2 and 3)

        # pd Series of color names, 1 per point.
        if trace is None:
            for trace_id in range(self.NUM_TRACES):
                self.display_color(trace_id=trace_id)
        else:
            self.display_color(trace_id=trace)

    def set_color(self, color, trace_id):
        self._colors[trace_id] = color
        self.display_color(trace_id)

    def display_color(self, trace_id: int):
        """
        Draws one zone on one figure using the passed colors
        """
        if self.figure is None:
            self.create_figure()
        else:
            colors = self._colors[trace_id]
            if colors is None:
                colors = self.pv.y
            colors = colors[self.mask]
            with self.figure.batch_update():
                self.figure.data[trace_id].marker.color = colors

    def update_fig_size(self):
        self.figure.layout.width = self.fig_width
        self.figure.layout.height = self.fig_height

    def update_pv(self, pv: ProjectedValues, progress_callback=None):
        if self.pv == pv:
            return
        if progress_callback is None:
            progress_callback = self.update_progress_circular
        self.pv = pv
        if self._current_selection is None:
            self._current_selection = utils.boolean_mask(self.pv.X, True)
        self.get_current_X_proj(progress_callback=progress_callback)
        self.get_projection_select().v_model = DimReducMethod.dimreduc_method_as_str(
            pv.current_proj[0]
        )
        if self.figure is None:
            self.create_figure()
        else:
            self.redraw()

    def set_dimension(self, dim: int):
        # Dimension is stored in the instance variable _current_dim
        """
        At runtime, GUI calls this function and swap our 2 and 3D figures
        """
        self._current_dim = dim
        self.create_figure()
        self.figure_container.children = [self.figure]

    def _get_projection_method(self) -> int:
        # proj is stored in the proj Select widget
        """
        Returns the current projection method
        """
        if self.get_projection_select().v_model == '!!disabled!!':
            self.get_projection_select().v_model = DimReducMethod.default_projection_as_str()
            return DimReducMethod.default_projection_as_int()
        return DimReducMethod.dimreduc_method_as_int(
            self.get_projection_select().v_model
        )

    def selection_to_mask(self, row_numbers):
        """
        to call between selection and setter
        """
        selection = utils.rows_to_mask(self.pv.X[self.mask], row_numbers)
        if selection.mean() == 0:
            return utils.boolean_mask(self.get_current_X_proj(masked=False), False)
        X_train = self.get_current_X_proj()
        knn = KNeighborsClassifier().fit(X_train, selection)
        X_predict = self.get_current_X_proj(masked=False)
        guessed_selection = pd.Series(knn.predict(X_predict), index=X_predict.index)
        # KNN extrapolation
        return guessed_selection.astype(bool)

    def _selection_event(self, trace, points, *args):
        self.first_selection |= self._current_selection.all()
        self._current_selection &= self.selection_to_mask(points.point_inds)
        if self._current_selection.any():
            self.create_figure()
            self.selection_changed(self, self._current_selection)
        else:
            self._deselection_event(rebuild=True)

    def _deselection_event(self, *args, rebuild=False):
        """Called on deselection"""
        # We tell the GUI
        self.first_selection = False
        self._current_selection = utils.boolean_mask(self.pv.X, True)
        self.display_rules()
        if rebuild:
            self.create_figure()
        else:
            self.update_selection()
        self.selection_changed(self, self._current_selection)

    def set_selection(self, new_selection_mask: pd.Series):
        """
        Called by tne UI when a new selection occured on the other HDE
        """

        if self._current_selection.all() and new_selection_mask.all():
            # New selection is empty. We already have an empty selection : nothing to do
            return

        # selection event
        self._current_selection = new_selection_mask
        self.update_selection()
        return

    def update_selection(self):
        if self.current_dim == 2:
            for fig in self.figure.data:
                fig.update(selectedpoints=utils.mask_to_rows(self._current_selection[self.mask]))
                fig.selectedpoints = utils.mask_to_rows(self._current_selection[self.mask])

    @property
    def mask(self):
        """
        mask should be applied on each display (x,y,z,color, selection)
        """
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

    def create_figure(self):
        """
        Called by __init__ and by set_selection
        Builds the FigureWidget for the given dimension
        """
        dim = self._current_dim
        x = y = z = None

        if self.current_X is not None:
            proj_values = self.get_current_X_proj()[self.mask]
            if proj_values is not None:
                x = proj_values[0]
                y = proj_values[1]
                if dim == 3:
                    z = proj_values[2]

        hde_marker = {'color': self.pv.y, 'colorscale': "Viridis"}
        if dim == 3:
            hde_marker['size'] = 2

        fig_args = {
            'x': x,
            'y': y,
            'mode': "markers",
            'marker': hde_marker,
            'customdata': self.pv.y[self.mask],
            'hovertemplate': "%{customdata:.3f}",
        }
        if dim == 3:
            fig_args['z'] = z
            fig_builder = Scatter3d
        else:
            fig_builder = Scattergl

        self.figure = FigureWidget(data=[fig_builder(**fig_args)])  # Trace 0 for dots
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 1 for rules
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 2 for region set
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 3 for region

        self.figure.update_layout(dragmode=False if self._selection_disabled else "lasso")
        self.figure.update_traces(
            selected={"marker": {"opacity": 1.0}},
            unselected={"marker": {"opacity": 0.1}},
            selector={'type': "scatter"}
        )
        self.figure.update_layout(
            margin={
                't': 0,
                'b': 0,
                'l': 0,
                'r': 0
            },
        )
        self.update_fig_size()
        self.figure._config = self.figure._config | {"displaylogo": False}
        self.figure._config = self.figure._config | {'displayModeBar': True}
        # We don't want the name of the trace to appear :
        for trace_id in range(len(self.figure.data)):
            self.figure.data[trace_id].showlegend = False
            self.show_trace(trace_id, self._visible[trace_id])
            self.display_color(trace_id)
        self.update_selection()

        if dim == 2:
            # selection only on trace 0
            self.figure.data[0].on_selection(self._selection_event)
            self.figure.data[0].on_deselect(self._deselection_event)

        self.figure_container.children = [self.figure]

    def redraw(self):
        projection = self.get_current_X_proj()
        x = projection[0]
        y = projection[1]
        if self.current_dim == 3:
            z = projection[2]

        with self.figure.batch_update():
            for trace_id in range(len(self.figure.data)):
                self.figure.data[trace_id].x = x
                self.figure.data[trace_id].y = y
                if self.current_dim == 3:
                    self.figure.data[trace_id].z = z
                self.figure.data[trace_id].showlegend = False
                self.show_trace(trace_id, self._visible[trace_id])
                self.display_color(trace_id)
            self.update_fig_size()

    def get_projection_select(self):
        """
       Called at startup by the GUI
       """
        return get_widget(app_widget, "14" if self.is_value_space else "17")

    def get_space_name(self) -> str:
        """
        For debug purposes only. Not very reliable.
        """
        return "VS" if self.is_value_space else "ES"

    @property
    def current_X(self) -> pd.DataFrame | None:
        if self.pv is None:
            return None  # When we're an ES HDE and no explanation have been imported nor computed yet
        return self.pv.X

    def get_current_X_proj(self, dim=None, masked: bool = True, progress_callback=None) -> pd.DataFrame | None:
        if dim is None:
            dim = self.current_dim
        if progress_callback is None:
            progress_callback = self.update_progress_circular
        X = self.pv.get_projection(self._get_projection_method(), dim, progress_callback)
        if X is None:
            return
        if masked:
            return X.loc[self.mask]
        return X

    def proj_should_be_computed(self, dim=None):
        if dim is None:
            dim = self.current_dim
        self.pv.is_present(self._get_projection_method(), dim)

    def set_tab(self, tab):
        self.disable_selection(tab > 1)
        self.show_trace(self.VALUES_TRACE, True)
        self.show_trace(self.RULES_TRACE, tab == 1)
        self.show_trace(self.REGIONSET_TRACE, tab == 2)
        self.show_trace(self.REGION_TRACE, tab == 3)
        # and it's the only place where selection is allowed
        self.active_tab = tab
