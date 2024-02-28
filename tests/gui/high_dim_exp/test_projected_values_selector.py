from unittest import TestCase

from ipywidgets import Widget
from mock import mock

from antakia.gui.high_dim_exp.projected_values_selector import ProjectedValuesSelector
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank
from antakia.gui.progress_bar import ProgressBar
from antakia.utils.dummy_datasets import generate_corner_dataset
from antakia_core.data_handler.projected_values import Proj
from tests.utils_fct import dummy_callable


class TestProjectedValuesSelector(TestCase):
    def setUp(self):
        self.X, self.y = generate_corner_dataset(10)
        self.pv_bank = ProjectedValueBank(self.y)
        self.pv_bank.get_projected_values(self.X)
        self.callable = dummy_callable()
        self.space = 'VS'
        self.k=0

    def dummy_callable(self):
        self.k+=1

    def test_init__build_widget(self):
        pvs = ProjectedValuesSelector(self.pv_bank, self.callable, self.space)
        assert isinstance(pvs.widget, Widget)
        assert isinstance(pvs.progress_bar, ProgressBar)
        assert pvs.projected_value is None
        assert pvs.X is None
        assert pvs.update_callback is self.callable
        assert pvs.space == self.space
        assert pvs.pv_bank is self.pv_bank
        assert isinstance(pvs.current_proj, Proj)
        assert pvs.progress_bar.progress == 100
