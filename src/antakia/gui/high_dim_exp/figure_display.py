from __future__ import annotations

import pandas as pd
import numpy as np
from plotly.graph_objects import FigureWidget, Scattergl, Scatter3d
import ipyvuetify as v
from sklearn.neighbors import KNeighborsClassifier

from antakia_core.data_handler.region import Region, RegionSet

import antakia_core.utils.utils as utils
import antakia.config as config

import logging as logging
from antakia.utils.logging_utils import conf_logger
from antakia.utils.stats import log_errors, stats_logger

logger = logging.getLogger(__name__)
conf_logger(logger)


class FigureDisplay:
    """
    A FigureDisplay objet manages all operation on a scatter plot 
    This class is only responsible for displaying the provided data

    It can display in 3 or 2 dimensions.

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
        """
        get trace name from id
        Parameters
        ----------
        trace_id : int

        Returns
        -------

        """
        if trace_id == FigureDisplay.VALUES_TRACE:
            return 'values trace'
        elif trace_id == FigureDisplay.RULES_TRACE:
            return 'rules trace'
        elif trace_id == FigureDisplay.REGIONSET_TRACE:
            return 'regionset trace'
        elif trace_id == FigureDisplay.REGION_TRACE:
            return 'region trace'
        else:
            return "unknown trace"

    def __init__(
        self,
        X: pd.DataFrame | None,
        y: pd.Series,
        selection_changed: callable,
        space: str
    ):
        """

        Parameters
        ----------
        X: data to display, should be 2 or 3D 
        y: target value (default color)
        selection_changed : callable called when a selection changed
        """
        # current active trace
        self.active_trace = 0
        # mask of value to display to limit points on graph
        self._mask = None
        # callback to notify gui that the selection has changed
        self.selection_changed = selection_changed
        self.X = X
        self.y = y

        self.space = space

        # Now we can init figure
        self.widget = v.Container()
        self.widget.class_ = "flex-fill"

        # is graph selectable
        self._selection_mode = 'lasso'
        # current selection
        if X is not None:
            self._current_selection = utils.boolean_mask(self.X, True)
        else:
            self._current_selection = None
        # is this selection first since last deselection ?
        self.first_selection = True

        # traces to show
        self._visible = [True, False, False, False]
        # trace_colors
        self._colors: list[pd.Series | None] = [None, None, None, None]

        # figures
        self.figure_2D = self.figure_3D = None
        # is the class fully initialized
        self.initialized = False

    @property
    def figure(self):
        if self.dim == 2:
            return self.figure_2D
        else:
            return self.figure_3D

    @figure.setter
    def figure(self, value):
        if self.dim == 2:
            self.figure_2D = value
        else:
            self.figure_3D = value

    @property
    def dim(self):
        if self.X is None:
            return config.ATK_DEFAULT_DIMENSION
        return self.X.shape[1]

    @property
    def current_selection(self):
        if self._current_selection is None:
            self._current_selection = utils.boolean_mask(self.X, True)
        return self._current_selection

    @current_selection.setter
    def current_selection(self, value):
        self._current_selection = value

    def initialize(self, X: pd.DataFrame = None):
        """
        inital computation called at startup, after init to compute required values
        Parameters
        ----------
        X : data to display, if NOne use the one provided during init

        Returns
        -------

        """
        if X is not None:
            self.X = X
            # compute X if needed
            self.get_X(masked=True)
        self.initialized = True

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

        self._selection_mode = False if is_disabled else "lasso"
        if self.dim == 2 and self.figure is not None:
            self.figure.update_layout(
                dragmode=self._selection_mode
            )

    def _show_trace(self, trace_id: int, show: bool):
        """
        show/hide trace
        Parameters
        ----------
        trace_id : trace to change
        show : show/hide

        Returns
        -------

        """
        self._visible[trace_id] = show
        self.figure.data[trace_id].visible = show

    def display_rules(self, selection_mask: pd.Series, rules_mask: pd.Series = None):
        """
        display a rule vs a selection
        Parameters
        ----------
        selection_mask: boolean series of selected points
        rules_mask: boolean series of rule validating points

        Returns
        -------

        """
        if selection_mask.all():
            selection_mask = ~selection_mask
        if rules_mask is None:
            rules_mask = selection_mask
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
        rs = RegionSet(self.X)
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

    def update_X(self, X: pd.DataFrame):
        """
        changes the underlying data - update the data used in display and dimension is necessary
        Parameters
        ----------
        X : data to display

        Returns
        -------

        """
        self.X = X
        self.redraw()

    def selection_to_mask(self, row_numbers: list[int]):
        """

        extrapolate selected row_numbers to full dataframe and return the selection mask on all the dataframe

        Parameters
        ----------
        row_numbers

        Returns
        -------

        """
        selection = utils.rows_to_mask(self.X[self.mask], row_numbers)
        if not selection.any() or selection.all():
            return utils.boolean_mask(self.get_X(masked=False), selection.mean())
        if self.mask.all():
            return selection
        X_train = self.get_X(masked=True)
        knn = KNeighborsClassifier().fit(X_train, selection)
        X_predict = self.get_X(masked=False)
        guessed_selection = pd.Series(knn.predict(X_predict), index=X_predict.index)
        # KNN extrapolation
        return guessed_selection.astype(bool)

    @log_errors
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
        stats_logger.log('hde_selection', {'first_selection': self.first_selection, 'space': self.space})
        self.current_selection &= self.selection_to_mask(points.point_inds)
        self.display_rules(self.current_selection)
        if self.current_selection.any():
            self.create_figure()
            self.selection_changed(self, self.current_selection)
        else:
            self._deselection_event(rebuild=True)

    @log_errors
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
        stats_logger.log('hde_deselection', {'first_selection': self.first_selection, 'space': self.space})
        # We tell the GUI
        self.first_selection = False
        self.current_selection = utils.boolean_mask(self.X, True)
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
        Called by tne UI when a new selection occurred on the other HDE
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
        self.display_rules(self.current_selection, self.current_selection)
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
    def mask(self) -> pd.Series:
        """
        mask should be applied on each display (x,y,z,color, selection)
        """
        if self._mask is None:
            self._mask = pd.Series([False] * len(self.X), index=self.X.index)
            limit = config.ATK_MAX_DOTS
            if len(self.X) > limit:
                indices = np.random.choice(self.X.index, size=limit, replace=False)
                self._mask.loc[indices] = True
            else:
                self._mask.loc[:] = True
        return self._mask

    def create_figure(self):
        """
        Builds the FigureWidget for the given dimension
        """
        x = y = z = None

        if self.X is not None:
            proj_values = self.get_X(masked=True)

        hde_marker = {'color': self.y, 'colorscale': "Viridis"}
        if self.dim == 3:
            hde_marker['size'] = 2

        fig_args = {
            'x': proj_values[0],
            'y': proj_values[1],
            'mode': "markers",
            'marker': hde_marker,
            'customdata': self.y[self.mask],
            'hovertemplate': "%{customdata:.3f}",
        }
        if self.dim == 3:
            fig_args['z'] = proj_values[2]
            fig_builder = Scatter3d
        else:
            fig_builder = Scattergl

        self.figure = FigureWidget(data=[fig_builder(**fig_args)])  # Trace 0 for dots
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 1 for rules
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 2 for region set
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 3 for region

        self.figure.update_layout(dragmode=self._selection_mode)
        self.figure.update_traces(
            selected={"marker": {"opacity": 1.0}},
            unselected={"marker": {"opacity": 0.1}},
            selector={'type': "scatter"}
        )
        self.figure.update_layout(
            autosize=True,
            margin={
                't': 0,
                'b': 0,
                'l': 0,
                'r': 0
            },
        )
        self.figure._config = self.figure._config | {"displaylogo": False}
        self.figure._config = self.figure._config | {'displayModeBar': True}
        # We don't want the name of the trace to appear :
        for trace_id in range(len(self.figure.data)):
            self.figure.data[trace_id].showlegend = False
            self._show_trace(trace_id, self._visible[trace_id])
            self.refresh_trace(trace_id)
        self.display_selection()

        if self.dim == 2:
            # selection only on trace 0
            self.figure.data[0].on_selection(self._selection_event)
            self.figure.data[0].on_deselect(self._deselection_event)
        self.widget.children = [self.figure]


    def redraw(self):
        """
        redraw all traces, without recreating figure
        Returns
        -------

        """
        if self.figure is None:
            self.create_figure()
        projection = self.get_X(masked=True)

        with self.figure.batch_update():
            for trace_id in range(len(self.figure.data)):
                self.figure.data[trace_id].x = projection[0]
                self.figure.data[trace_id].y = projection[1]
                if self.dim == 3:
                    self.figure.data[trace_id].z = projection[2]
                self.figure.data[trace_id].showlegend = False
                self._show_trace(trace_id, self._visible[trace_id])
                self.refresh_trace(trace_id)
        self.widget.children = [self.figure]

    def get_X(self, masked: bool) -> pd.DataFrame | None:
        """

        return current projection value
        its computes it if necessary - progress is published in the callback

        Parameters
        ----------
        masked

        Returns
        -------

        """
        if masked and self.X is not None:
            return self.X.loc[self.mask]
        return self.X

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
        self.active_trace = tab
