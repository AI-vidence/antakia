from unittest import TestCase

import numpy as np
import pytest
import pandas as pd

from antakia.explanation.explanation_method import ExplanationMethod
from antakia.utils.dummy_datasets import generate_corner_dataset
from antakia_core.utils.utils import ProblemCategory
from tests.utils_fct import DummyModel, dummy_callable


class TestExplanationMethod(TestCase):
    def setUp(self):
        self.X, self.y = generate_corner_dataset(10)
        self.model = DummyModel
        self.callable = dummy_callable()

    def test_init(self):
        class DummyExplanation(ExplanationMethod):
            def compute(self):
                self.publish_progress(100)
                return 0

        exp_meth = DummyExplanation(1, self.X, self.y, self.model, ProblemCategory.regression)
        assert exp_meth.explanation_method == 1
        np.testing.assert_array_equal(exp_meth.X, self.X)

        class DummyExplanation1(ExplanationMethod):
            def compute(self):
                self.publish_progress(100)
                return 0
        with pytest.raises(ValueError):
            exp_meth = DummyExplanation1(6, self.X, self.model, ProblemCategory.regression)


def test_is_valid_explanation_method():
    assert (ExplanationMethod.is_valid_explanation_method(1))


def test_explanation_methods_as_list():
    assert ExplanationMethod.explanation_methods_as_list() == [1, 2]


def test_explain_method_as_str():
    assert ExplanationMethod.explain_method_as_str(1) == 'SHAP'
    assert ExplanationMethod.explain_method_as_str(2) == 'LIME'
    with pytest.raises(ValueError):
        ExplanationMethod.explain_method_as_str(3)


def test_explain_method_as_int():
    assert ExplanationMethod.explain_method_as_int('shap') == 1
    assert ExplanationMethod.explain_method_as_int('Lime') == 2
    with pytest.raises(ValueError):
        ExplanationMethod.explain_method_as_int('shapp')
