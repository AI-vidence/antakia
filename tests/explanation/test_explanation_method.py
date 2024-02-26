from unittest import TestCase

import numpy as np
import pytest

from antakia.utils.dummy_datasets import generate_corner_dataset
from tests.utils_fct import EMPTYExplanation, DummyModel, dummy_callable


class TestExplanationMethod(TestCase):
    def setUp(self):
        self.X, self.y = generate_corner_dataset(10)
        self.model = DummyModel
        self.callable = dummy_callable()

    def test_init(self):
        exp_meth = EMPTYExplanation(self.X, self.y, self.model, self.callable)
        assert exp_meth.explanation_method == 1
        np.testing.assert_array_equal(exp_meth.X, self.X)

    def test_is_valid_explanation_method(self):
        exp_meth = EMPTYExplanation(self.X, self.y, self.model, self.callable)
        assert (exp_meth.is_valid_explanation_method(1))

    def test_explanation_methods_as_list(self):
        exp_meth = EMPTYExplanation(self.X, self.y, self.model, self.callable)
        assert exp_meth.explanation_methods_as_list() == [1, 2]

    def test_explain_method_as_str(self):
        exp_meth = EMPTYExplanation(self.X, self.y, self.model, self.callable)
        assert exp_meth.explain_method_as_str(1) == 'SHAP'
        assert exp_meth.explain_method_as_str(2) == 'LIME'
        with pytest.raises(ValueError):
            exp_meth.explain_method_as_str(3)

    def test_explain_method_as_int(self):
        exp_meth = EMPTYExplanation(self.X, self.y, self.model, self.callable)
        assert exp_meth.explain_method_as_int('shap') == 1
        assert exp_meth.explain_method_as_int('Lime') == 2
        with pytest.raises(ValueError):
            exp_meth.explain_method_as_int('shapp')
