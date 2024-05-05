from unittest import TestCase

import pandas as pd

import antakia_core.utils as utils

from antakia.gui.high_dim_exp.figure_display import FigureDisplay
from antakia.utils.dummy_datasets import generate_corner_dataset
from tests.antakia_test_case import AntakiaTestCase

# trace_name = FigureDisplay.trace_name


class TestFigureDisplay(AntakiaTestCase):

    def setUp(self):
        super().setUp()
        self.fd = FigureDisplay(self.data_store, lambda x: x + 1, 'VS')
        self.X_proj = pd.DataFrame(self.data_store.X.values, index=self.data_store.X.index)

    # def test_trace_name(self):
    #     assert trace_name(0) == 'values trace'
    #     assert trace_name(1) == 'rules trace'
    #     assert trace_name(2) == 'regionset trace'
    #     assert trace_name(3) == 'region trace'
    #     assert trace_name(8) == 'unknown trace'

    def test_init(self):
        fd = self.fd
        assert fd.active_trace == 0
        assert fd._display_mask is None
        # assert fd.selection_changed
        # assert fd.widget == v.Container()
        assert fd.widget.class_ == "flex-fill"
        assert fd._selection_mode == 'lasso'
        assert fd.data_store is self.data_store
        assert fd.first_selection
        assert fd._visible == [True, False, False, False]
        assert fd._colors == [None, None, None, None]
        assert fd.figure_2D is None
        assert fd.figure_3D is None
        assert not fd.initialized

        self.fd.initialize(self.X_proj)
        assert fd.data_store.selection_mask.equals(utils.boolean_mask(fd.figure_data, True))

    def test_set_get_figure(self):
        self.fd.initialize(self.X_proj)
        fd = self.fd
        fd1 = FigureDisplay(self.data_store, lambda x: x + 1, 'VS')

    def test_initialize_create_figure(self):
        self.fd.initialize(self.X_proj)
        pass

    def test_get_X(self):
        self.fd.initialize(self.X_proj)
        fd = self.fd

    def test_display_region(self):
        self.fd.initialize(self.X_proj)
        fd = self.fd
