from unittest import TestCase

import pandas as pd

import antakia_core.utils as utils

from antakia.gui.high_dim_exp.figure_display import FigureDisplay
from antakia.utils.dummy_datasets import generate_corner_dataset

trace_name = FigureDisplay.trace_name


class TestFigureDisplay(TestCase):

    def setUp(self):
        self.X, self.y = generate_corner_dataset(10)
        self.X = pd.DataFrame(self.X)
        self.y = pd.DataFrame(self.y)
        self.fd = FigureDisplay(self.X, self.y, lambda x: x + 1, 'VS')

    def test_trace_name(self):
        assert trace_name(0) == 'values trace'
        assert trace_name(1) == 'rules trace'
        assert trace_name(2) == 'regionset trace'
        assert trace_name(3) == 'region trace'
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

    def test_set_get_figure(self):
        fd = self.fd
        fd.figure = 5

        fd1 = FigureDisplay(None, self.y, lambda x: x + 1, 'VS')
        fd1.figure = 5

    def test_initialize_create_figure(self):
        pass

    def test_get_X(self):
        fd = self.fd
        fd.get_X(True)

    def test_display_region(self):
        fd = self.fd
