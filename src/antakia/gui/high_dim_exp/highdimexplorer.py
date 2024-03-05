from __future__ import annotations

import pandas as pd
from antakia_core.data_handler.projected_values import ProjectedValues

from antakia.gui.high_dim_exp.figure_display import FigureDisplay
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank
from antakia.gui.high_dim_exp.projected_values_selector import ProjectedValuesSelector

import logging as logging
from antakia.utils.logging_utils import conf_logger

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

    def __init__(
        self,
        pv_bank: ProjectedValueBank,
        selection_changed: callable,
        space: str
    ):
        """

        Parameters
        ----------
        pv_bank: projected values storage
        selection_changed : callable called when a selection changed
        """
        self.pv_bank = pv_bank

        # projected values handler & widget
        self.projected_value_selector = ProjectedValuesSelector(
            pv_bank,
            self.refresh,
            space
        )

        self.figure = FigureDisplay(
            None,
            pv_bank.y,
            selection_changed,
            space
        )

        self.initialized = False

    def get_current_X_proj(self, dim=None, progress_callback=None) -> pd.DataFrame | None:
        return self.projected_value_selector.get_current_X_proj(dim, progress_callback)

    def initialize(self, progress_callback, X: pd.DataFrame):
        """
        inital computation (called at startup, after init to compute required values
        Parameters
        ----------
        progress_callback : callable to notify progress

        Returns
        -------

        """
        self.projected_value_selector.initialize(progress_callback, X)
        self.figure.initialize(self.get_current_X_proj())
        self.initialized = True

    def disable(self, disable_figure: bool, disable_projection: bool):
        """
        disable dropdown select
        Parameters
        ----------
        is_disabled: disable value

        Returns
        -------

        """
        self.figure.disable_selection(disable_figure)
        self.projected_value_selector.disable(disable_projection)

    def refresh(self, progress_callback=None):
        self.disable(True, True)
        self.figure.update_X(self.get_current_X_proj(progress_callback=progress_callback))
        self.disable(False, False)

    def update_X(self, X: pd.DataFrame, progress_callback=None):
        """
        changes the undelying projected value instance - update the data used in display
        Parameters
        ----------
        pv
        progress_callback

        Returns
        -------

        """
        self.projected_value_selector.update_X(X)
        self.refresh(progress_callback)

    @property
    def current_pv(self) -> ProjectedValues:
        return self.projected_value_selector.projected_value

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

    def set_tab(self, *args, **kwargs):
        return self.figure.set_tab(*args, **kwargs)

    def set_selection(self, *args, **kwargs):
        return self.figure.set_selection(*args, **kwargs)

    def set_dim(self, dim: int):
        self.projected_value_selector.update_dim(dim)
