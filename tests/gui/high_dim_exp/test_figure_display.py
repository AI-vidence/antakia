from unittest import TestCase

import pandas as pd
import ipyvuetify as v
import numpy as np

import antakia.config as config
import antakia_core.utils.utils as utils

from antakia.gui.high_dim_exp.figure_display import FigureDisplay
from antakia.utils.dummy_datasets import generate_corner_dataset
from tests.utils_fct import dummy_mask

trace_name = FigureDisplay.trace_name


class TestFigureDisplay(TestCase):

    def setUp(self):
        np.random.seed(1234)
        X = pd.DataFrame(np.random.randn(500, 3), columns=['var1', 'var2', 'var3'])
        y = np.sum(X, axis=1)
        self.X_2D = X.iloc[:, [0, 1]]
        self.X_3D = X.iloc[:, [0, 1, 2]]
        self.y = pd.Series(y)

        self.fd = FigureDisplay(self.X_2D, self.y, lambda x: x + 1, 'VS')

    def test_trace_name(self):
        assert trace_name(FigureDisplay.VALUES_TRACE) == 'values trace'
        assert trace_name(FigureDisplay.RULES_TRACE) == 'rules trace'
        assert trace_name(FigureDisplay.REGIONSET_TRACE) == 'regionset trace'
        assert trace_name(FigureDisplay.REGION_TRACE) == 'region trace'
        assert trace_name(8) == 'unknown trace'

    def test_init(self):
        fd = self.fd
        assert fd.active_trace == 0
        assert fd._mask is None
        # assert fd.selection_changed
        # assert fd.widget == v.Container()
        assert fd.widget.class_ == "flex-fill"
        assert fd._selection_mode == 'lasso'
        assert fd._current_selection.equals(utils.boolean_mask(fd.X, True))
        assert fd.first_selection
        assert fd._visible == [True, False, False, False]
        assert fd._colors == [None, None, None, None]
        assert fd.figure_2D is None
        assert fd.figure_3D is None
        assert not fd.initialized

        fd = FigureDisplay(None, self.y, lambda x: x + 1, 'VS')
        assert fd._current_selection is None
        #test dim property when X is None
        assert fd.dim == config.ATK_DEFAULT_DIMENSION

    def test_set_get_figure(self):
        # check for dim=2
        fd = self.fd
        fd.figure = 5
        assert fd.figure == 5

        # check for dim=3
        fd = FigureDisplay(self.X_3D, self.y, lambda x: x + 1, 'VS')
        fd.figure = 5
        assert fd.figure == 5

    def test_set_get_current_selection(self):
        fd = FigureDisplay(None, self.y, lambda x: x + 1, 'VS')
        assert fd.current_selection.all()
        fd = self.fd
        mask = dummy_mask(fd.X)
        fd.current_selection = mask
        assert fd._current_selection.equals(mask)

    def test_initialize_create_figure(self):
        pass

    def test_get_X(self):
        fd = self.fd
        fd.get_X(True)

    def test_display_region(self):
        fd = self.fd
