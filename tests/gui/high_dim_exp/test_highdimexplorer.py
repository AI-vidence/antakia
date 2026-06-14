
from mock import mock

from antakia.gui.high_dim_exp.highdimexplorer import HighDimExplorer
from antakia.utils.dummy_datasets import generate_corner_dataset
from tests.antakia_test_case import AntakiaTestCase
from tests.utils_fct import dummy_callable, is_mask_of_X


class TestHighDimExplorer(AntakiaTestCase):
    def setUp(self):
        super().setUp()
        self.data_store.pv_bank.get_projected_values(self.data_store.X)
        self.selection_changed_called = 0
        self.callable = dummy_callable

    def test_init(self):
        hde = HighDimExplorer(self.data_store, dummy_callable, "VS")  # REMPLACER DUMMY CALLABLE
        assert hde.data_store == self.data_store
        # assert hde.projected_value_selector ==
        # assert hde.
        # assert hde.

    def selection_changed(self, caller, mask):
        assert is_mask_of_X(mask, self.data_store.X)
        if caller is not None:
            assert isinstance(caller, HighDimExplorer)
        self.selection_changed_called += 1

    @mock.patch("antakia_core.data_handler.projected_values.ProjectedValues.get_projection")
    def test_update_X(self, pv_cpt):
        pv_cpt.return_value, _ = generate_corner_dataset(10)
        hde = HighDimExplorer(self.data_store, dummy_callable, "VS")


#        hde.update_X(pd.DataFrame(self.X))
