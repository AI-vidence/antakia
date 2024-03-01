import pandas as pd
from unittest import TestCase

from mock import mock

from antakia.gui.high_dim_exp.highdimexplorer import HighDimExplorer
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank
from antakia.utils.dummy_datasets import generate_corner_dataset
from tests.utils_fct import is_mask_of_X, dummy_callable


class TestHighDimExplorer(TestCase):
    def setUp(self):
        self.X, self.y = generate_corner_dataset(10)
        self.X, self.y = pd.DataFrame(self.X), pd.Series(self.y)
        self.pv_bank = ProjectedValueBank(self.y)
        self.pv_bank.get_projected_values(self.X)
        self.selection_changed_called = 0
        self.callable = dummy_callable()

    def test_init(self):
        hde = HighDimExplorer(self.pv_bank, dummy_callable(), 'VS')  # REMPLACER DUMMY CALLABLE
        assert hde.pv_bank == self.pv_bank
        # assert hde.projected_value_selector ==
        # assert hde.
        # assert hde.

    def selection_changed(self, caller, mask):
        assert is_mask_of_X(mask, self.X)
        if caller is not None:
            assert isinstance(caller, HighDimExplorer)
        self.selection_changed_called += 1

    @mock.patch('antakia_core.data_handler.projected_values.ProjectedValues.get_projection')
    def test_update_X(self, pv_cpt):
        pv_cpt.return_value, _ = generate_corner_dataset(10)
        hde = HighDimExplorer(self.pv_bank, dummy_callable(), 'VS')
        hde.update_X(pd.DataFrame(self.X))
