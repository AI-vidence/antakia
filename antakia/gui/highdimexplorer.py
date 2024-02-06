from __future__ import annotations

import pandas as pd
import numpy as np
from plotly.graph_objects import FigureWidget, Scattergl, Scatter3d
import ipyvuetify as v
from sklearn.neighbors import KNeighborsClassifier

from antakia.data_handler.region import Region, RegionSet
from antakia.gui.projected_value_selector import ProjectedValueSelector

import antakia.utils.utils as utils
import antakia.config as config
from antakia.data_handler.projected_values import ProjectedValues

import logging as logging
from antakia.utils.logging import conf_logger

logger = logging.getLogger(__name__)
conf_logger(logger)


class HighDimExplorer:
    """
    An HighDimExplorer displays one high dim Dataframes on a scatter plot.
    This class is only responsible for displaying the provided projection
    dimensionality reduction is handled by ProjectedValueSelector

    It can display in 3 or 2 dimensions.

    Implementation details :
    It handles projections computation and switch through ProjectedValueSelector
    it can update its high dim dataframe through the update_pv method

    Attributes are mostly privates (underscorred) since they are not meant to be used outside of the class.

    Attributes :

    """

    # Trace indexes : 0 for values, 1 for rules, 2 for regions
    NUM_TRACES = 4
    VALUES_TRACE = 0
    RULES_TRACE = 1
    REGIONSET_TRACE = 2
    REGION_TRACE = 3

    @staticmethod
    def trace_name(trace_id: int) -> str:
        '''
        get trace name from id
        Parameters
        ----------
        trace_id : int

        Returns
        -------

        '''
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
            pv: ProjectedValues | None,
            init_dim: int,
            fig_size: int,
            selection_changed: callable,
            space_type: str
    ):
        """

        Parameters
        ----------
        pv: projected Value to display
        init_dim : starting dimension
        fig_size : starting figure size
        selection_changed : callable called when a selection changed
        space_type : VS or ES
        """
        if space_type not in ['ES', 'VS']:
            raise ValueError(f"HDE.init: space_type must be 'ES' or 'VS', not {space_type}")
        self.is_value_space = space_type == 'VS'
        if init_dim not in [2, 3]:
            raise ValueError(f"HDE.init: dim must be 2 or 3, not {init_dim}")
        self._current_dim = init_dim
        # current active tab
        self.active_tab = 0
        # mask of value to display to limit points on graph
        self._mask = None
        # callback to notify gui that the selection has changed
        self.selection_changed = selection_changed
        # projected values handler & widget
        self.projected_value_selector = ProjectedValueSelector(
            self.is_value_space,
            init_dim,
            self.redraw,
            pv
        )

        # Now we can init figure
        self.figure_container = v.Container()
        self.figure_container.class_ = "flex-fill"

        # display parameters
        self.fig_width = fig_size
        self.fig_height = fig_size / 2

        # is graph selectable
        self._selection_disabled = False
        # current selection
        if pv is not None:
            self._current_selection = utils.boolean_mask(pv.X, True)
        else:
            self._current_selection = None
        # is this selection first since last deselection ?
        self.first_selection = False

        # traces to show
        self._visible = [True, False, False, False]
        # trace_colors
        self._colors: list[pd.Series | None] = [None, None, None, None]

        # figures
        self.figure_2D = self.figure_3D = None
        # is the class fully initialized
        self.initialized = False

    @property
    def current_selection(self):
        if self._current_selection is None:
            self._current_selection = utils.boolean_mask(self.current_X, True)
        return self._current_selection

    @current_selection.setter
    def current_selection(self, value):
        self._current_selection = value

    def initialize(self, progress_callback, pv=None):
        """
        inital computation (called at startup, after init to compute required values
        Parameters
        ----------
        progress_callback : callable to notify progress

        Returns
        -------

        """
        if pv is not None:
            # for ES space we need to manually provide the pv value in other to not trigger all auto updates
            self.projected_value_selector.projected_value = pv
        self.get_current_X_proj(progress_callback=progress_callback)
        self.create_figure()
        self.initialized = True

    def disable_widgets(self, is_disabled: bool):
        """
        disable dropdown select
        Parameters
        ----------
        is_disabled: disable value

        Returns
        -------

        """
        self.projected_value_selector.disable(is_disabled)

    @property
    def dim(self):
        """
        get current dimension
        Returns
        -------

        """
        return self._current_dim

    @dim.setter
    def dim(self, dim):
        """
        change current dim value
        Parameters
        ----------
        dim : new dim value

        Returns
        -------

        """
        self._current_dim = dim
        self.projected_value_selector.update_dim(dim)
        # we recreate figure on dim change
        self.create_figure()

    @property
    def figure(self):
        """
        get current active figure to display/edit
        Returns
        -------

        """
        if self.dim == 2:
            return self.figure_2D
        return self.figure_3D

    @figure.setter
    def figure(self, fig):
        """
        change current figure
        Parameters
        ----------
        fig : new figure value

        Returns
        -------

        """
        if self.dim == 2:
            self.figure_2D = fig
        else:
            self.figure_3D = fig

    # ---- display Methods ------

    def disable_selection(self, is_disabled: bool):
        """
        enable/disable selection on graph
        Parameters
        ----------
        is_disabled

        Returns
        -------

        """
        self._selection_disabled = is_disabled
        if self.figure_2D is not None:
            self.figure_2D.update_layout(
                dragmode=False if is_disabled else "lasso"
            )

    def _show_trace(self, trace_id: int, show: bool):
        """
        show/hide trace
        Parameters
        ----------
        trace_id : trace to show
        show : show/hide

        Returns
        -------

        """
        self._visible[trace_id] = show
        self.figure.data[trace_id].visible = show

    def display_rules(self, selection_mask: pd.Series, rules_mask: pd.Series):
        """
        display a rule vs a selection
        Parameters
        ----------
        selection_mask: boolean series of selected points
        rules_mask: boolean series of rule validating points

        Returns
        -------

        """
        color, _ = utils.get_mask_comparison_color(rules_mask, selection_mask)

        self._colors[self.RULES_TRACE] = color
        self._display_zones(self.RULES_TRACE)

    def display_regionset(self, region_set: RegionSet):
        """
        display a region set, each region in its color
        Parameters
        ----------
        region_set

        Returns
        -------

        """
        self._colors[self.REGIONSET_TRACE] = region_set.get_color_serie()
        self._display_zones(self.REGIONSET_TRACE)

    def display_region(self, region: Region):
        """
        display a single region
        Parameters
        ----------
        region

        Returns
        -------

        """
        rs = RegionSet(self.current_X)
        rs.add(region)
        self._colors[self.REGION_TRACE] = rs.get_color_serie()
        self._display_zones(self.REGION_TRACE)

    def _display_zones(self, trace=None):
        """
        refresh provided trace or all trace if None
        do not alter visibility
        Parameters
        ----------
        trace

        Returns
        -------

        """

        if trace is None:
            for trace_id in range(self.NUM_TRACES):
                self.refresh_trace(trace_id=trace_id)
        else:
            self.refresh_trace(trace_id=trace)

    def set_color(self, color: pd.Series, trace_id: int):
        """
        set the provided color as the scatter point color on the provided trace id
        do not alter show/hide trace
        Parameters
        ----------
        color
        trace_id

        Returns
        -------

        """
        self._colors[trace_id] = color
        self.refresh_trace(trace_id)

    def refresh_trace(self, trace_id: int):
        """
        refresh the provided trace id
        do not alter show/hide trace

        Parameters
        ----------
        trace_id

        Returns
        -------

        """
        if self.figure is None:
            self.create_figure()
        else:
            colors = self._colors[trace_id]
            if colors is None:
                colors = self.y
            colors = colors[self.mask]
            with self.figure.batch_update():
                self.figure.data[trace_id].marker.color = colors

    def update_fig_size(self):
        """
        update figure to match fig size attributes
        Returns
        -------

        """
        self.figure.layout.width = self.fig_width
        self.figure.layout.height = self.fig_height

    def update_pv(self, pv: ProjectedValues, progress_callback=None):
        """
        changes the undelying projected value instance - update the data used in display
        Parameters
        ----------
        pv
        progress_callback

        Returns
        -------

        """
        # update value in selector
        self.projected_value_selector.update_projected_value(pv)
        # compute value if needed
        self.get_current_X_proj(progress_callback=progress_callback)
        self.redraw()

    def selection_to_mask(self, row_numbers):
        """

        extrapolate selected row_numbers to full dataframe and return the selection mask on all the dataframe

        Parameters
        ----------
        row_numbers

        Returns
        -------

        """
        selection = utils.rows_to_mask(self.current_X[self.mask], row_numbers)

        if not selection.any() or selection.all():
            return utils.boolean_mask(self.get_current_X_proj(masked=False), selection.mean())
        if self.mask.all():
            return selection
        X_train = self.get_current_X_proj()
        knn = KNeighborsClassifier().fit(X_train, selection)
        X_predict = self.get_current_X_proj(masked=False)
        guessed_selection = pd.Series(knn.predict(X_predict), index=X_predict.index)
        # KNN extrapolation
        return guessed_selection.astype(bool)

    def _selection_event(self, trace, points, *args):
        """
        callback triggered by selection on graph
        selects points and update display on both hde (calls selection changed)
        deselects if no points selected

        update selection taking intersection
        Parameters
        ----------
        trace
        points
        args

        Returns
        -------

        """
        self.first_selection |= self.current_selection.all()
        self.current_selection &= self.selection_to_mask(points.point_inds)
        if self.current_selection.any():
            self.create_figure()
            self.selection_changed(self, self.current_selection)
        else:
            self._deselection_event(rebuild=True)

    def _deselection_event(self, *args, rebuild=False):
        """
        clear selection -- called by deselection on graph
        synchronize hdes

        Parameters
        ----------
        args
        rebuild

        Returns
        -------

        """
        # We tell the GUI
        self.first_selection = False
        self.current_selection = utils.boolean_mask(self.current_X, True)
        self.selection_changed(self, self.current_selection)
        self.display_rules(~self.current_selection, ~self.current_selection)
        if rebuild:
            self.create_figure()
        else:
            self.display_selection()

    def set_selection(self, new_selection_mask: pd.Series):
        """
        update selection from mask
        no update_callback
        Called by tne UI when a new selection occured on the other HDE
        Parameters
        ----------
        new_selection_mask

        Returns
        -------

        """

        if self.current_selection.all() and new_selection_mask.all():
            # New selection is empty. We already have an empty selection : nothing to do
            return

        # selection event
        self.current_selection = new_selection_mask
        self.display_rules(~self.current_selection, ~self.current_selection)
        self.display_selection()
        return

    def display_selection(self):
        """
        display selection on figure
        Returns
        -------

        """
        if self.dim == 2:
            fig = self.figure.data[0]
            fig.update(selectedpoints=utils.mask_to_rows(self.current_selection[self.mask]))
            fig.selectedpoints = utils.mask_to_rows(self.current_selection[self.mask])

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

        hde_marker = {'color': self.y, 'colorscale': "Viridis"}
        if dim == 3:
            hde_marker['size'] = 2

        fig_args = {
            'x': x,
            'y': y,
            'mode': "markers",
            'marker': hde_marker,
            'customdata': self.y[self.mask],
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
            self._show_trace(trace_id, self._visible[trace_id])
            self.refresh_trace(trace_id)
        self.display_selection()

        if dim == 2:
            # selection only on trace 0
            self.figure.data[0].on_selection(self._selection_event)
            self.figure.data[0].on_deselect(self._deselection_event)

        self.figure_container.children = [self.figure]

    def redraw(self):
        """
        redraw all traces, without recreating figure
        Returns
        -------

        """
        if self.figure is None:
            self.create_figure()
        projection = self.get_current_X_proj()
        x = projection[0]
        y = projection[1]
        if self.dim == 3:
            z = projection[2]

        with self.figure.batch_update():
            for trace_id in range(len(self.figure.data)):
                self.figure.data[trace_id].x = x
                self.figure.data[trace_id].y = y
                if self.dim == 3:
                    self.figure.data[trace_id].z = z
                self.figure.data[trace_id].showlegend = False
                self._show_trace(trace_id, self._visible[trace_id])
                self.refresh_trace(trace_id)
            self.update_fig_size()

    def get_space_name(self) -> str:
        """
        For debug purposes only. Not very reliable.
        """
        return "VS" if self.is_value_space else "ES"

    @property
    def current_X(self) -> pd.DataFrame | None:
        """
        return hde current X value (not projected)

        Returns
        -------

        """
        if self.projected_value_selector is None:
            return None  # When we're an ES HDE and no explanation have been imported nor computed yet
        return self.projected_value_selector.projected_value.X

    @property
    def y(self):
        """
        return y value
        Returns
        -------

        """
        return self.projected_value_selector.projected_value.y

    def get_current_X_proj(self, dim=None, masked: bool = True, progress_callback=None) -> pd.DataFrame | None:
        """

        return current projection value
        its computes it if necessary - progress is published in the callback

        Parameters
        ----------
        dim
        masked
        progress_callback

        Returns
        -------

        """
        X_proj = self.projected_value_selector.get_current_X_proj(dim, progress_callback)
        if masked and X_proj is not None:
            return X_proj.loc[self.mask]
        return X_proj

    def set_tab(self, tab):
        """
        show/hide trace depending on tab
        Parameters
        ----------
        tab

        Returns
        -------

        """
        self.disable_selection(tab > 1)
        self._show_trace(self.VALUES_TRACE, tab == 0)
        self._show_trace(self.RULES_TRACE, tab == 1)
        self._show_trace(self.REGIONSET_TRACE, tab == 2)
        self._show_trace(self.REGION_TRACE, tab == 3)
        # and it's the only place where selection is allowed
        self.active_tab = tab
