import unittest

import pandas as pd

from antakia.gui.high_dim_exp.highdimexplorer import HighDimExplorer
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank
from tests.utils_fct import is_mask_of_X


class TestHighDimExplorer(unittest.TestCase):
    def setUp(self):
        self.selection_changed_called = 0
        df = pd.read_csv('../data/california_housing.csv').drop(['Unnamed: 0'], axis=1)
        df = df.sample(len(df))
        self.X = df.iloc[:, :8]  # the dataset
        self.y = df.iloc[:, 9]  # the target variable

        self.pv_bank = ProjectedValueBank(self.y)
        self.pv_bank.get_projected_values(self.X)


    def selection_changed(self, caller, mask):
        assert is_mask_of_X(mask, self.X)
        if caller is not None:
            assert isinstance(caller, HighDimExplorer)
        self.selection_changed_called += 1

    def test_init():
        hde = HighDimExplorer(pv_bank, )
