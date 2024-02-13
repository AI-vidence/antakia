import numpy as np
import pytest

from antakia import config
from tests.utils_fct import generate_ExplanationValues, EMPTYExplanation


def test_init():
    X, y, X_exp, exp_val, _ = generate_ExplanationValues()
    exp_meth = EMPTYExplanation(X, y, 'DT', lambda *args: None)
    assert exp_meth.explanation_method == 1
    np.testing.assert_array_equal(exp_meth.X, X)


def test_is_valid_explanation_method():
    X, y, X_exp, exp_val, _ = generate_ExplanationValues()
    exp_meth = EMPTYExplanation(X, y, X_exp, exp_val)
    assert (exp_meth.is_valid_explanation_method(1))


def test_explanation_methods_as_list():
    X, y, X_exp, exp_val,_ = generate_ExplanationValues()
    exp_meth = EMPTYExplanation(X, y, X_exp, exp_val)
    assert exp_meth.explanation_methods_as_list() == [1, 2]


def test_explain_method_as_str():
    X, y, X_exp, exp_val, _ = generate_ExplanationValues()
    exp_meth = EMPTYExplanation(X, y, X_exp, exp_val)
    assert exp_meth.explain_method_as_str(1) == 'SHAP'
    assert exp_meth.explain_method_as_str(2) == 'LIME'
    with pytest.raises(ValueError):
        exp_meth.explain_method_as_str(3)


def test_explain_method_as_int():
    X, y, X_exp, exp_val, _ = generate_ExplanationValues()
    exp_meth = EMPTYExplanation(X, y, X_exp, exp_val)
    assert exp_meth.explain_method_as_int('shap') == 1
    assert exp_meth.explain_method_as_int('Lime') == 2
    with pytest.raises(ValueError):
        exp_meth.explain_method_as_int('shapp')
