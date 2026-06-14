from unittest import TestCase

import pandas as pd

from antakia import AntakIA
from antakia.utils.dummy_datasets import generate_corner_dataset
from tests.utils_fct import DummyModel


class AntakiaTestCase(TestCase):
    def setUp(self):
        X, y = generate_corner_dataset(10)
        model = DummyModel()
        self.data_store = AntakIA._get_data_store(X, y, model)
        X_exp = pd.DataFrame(generate_corner_dataset(10)[0])
        self.data_store_w_exp = AntakIA._get_data_store(X, y, model, X_exp=X_exp)
        self.X_exp = pd.DataFrame(generate_corner_dataset(10)[0])
