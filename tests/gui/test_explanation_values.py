import pandas as pd
import numpy as np
import os
import mock

from antakia.gui.explanation_values import ExplanationValues
from antakia.data_handler.projected_values import ProjectedValues
from antakia.compute.explanation.explanations import compute_explanations, ExplanationMethod
from antakia.utils.utils import debug
from tests.utils_fct import generate_ExplanationValues


def test_init():
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 'SHAP')

    exp = ExplanationValues(X_df, Y_ser, 'DT', function)

    assert exp.X.equals(X_df)
    assert exp.y.equals(Y_ser)
    assert exp.model == 'DT'
    assert exp.explanations == {'Imported': None, 'SHAP': None, 'LIME': None}
    assert exp.current_exp == 'SHAP'
    assert exp.on_change_callback == function

    exp1 = ExplanationValues(X_df, Y_ser, 'DT', function, X_df)
    assert exp1.current_exp == 'Imported'
    assert exp1.explanations['Imported'].X.equals(X_df)
    assert exp1.explanations['SHAP'] is None
    assert exp1.explanations['LIME'] is None


@mock.patch('antakia.gui.explanation_values.compute_explanations')
def test_initialize(cpt_exp):
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT')
    cpt_exp.return_value = X_df

    exp.initialize(lambda x: x + 1)

    # assert exp.get_explanation_select().v_model ==


def test_current_pv():
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 'SHAP')
    print(exp.explanations)
    print(exp.current_exp)
    print(type(exp.current_pv))


def test_has_user_exp():
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 'SHAP')
    assert exp.has_user_exp

    X_df, Y_ser, function, exp = generate_ExplanationValues('DT')
    assert not exp.has_user_exp


def test_update_explanation_select():
    pass


def test_get_compute_menu():
    pass


@mock.patch('antakia.gui.explanation_values.compute_explanations')
def test_compute_explanation(cpt_exp):
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 1)
    cpt_exp.return_value = X_df
    assert exp.current_exp == exp.available_exp[0]

    assert exp.explanations['Imported'].X == 1
    assert exp.explanations['SHAP'] is None
    assert exp.explanations['LIME'] is None

    exp.compute_explanation(1, None)
    # print(exp.current_exp)
    # X_exp =
    # assert exp.explanation[exp.current_exp] == ProjectedValues(X_exp,exp.y)


def test_update_compute_menu():
    pass


def test_compute_btn_clicked():
    pass


def test_disable_selection():
    X_df, Y_ser, function, exp = generate_ExplanationValues()
    exp0 = ExplanationValues(X_df, Y_ser, 'DT', function)
    exp0.disable_selection(True)
    assert exp0.get_compute_menu().disabled and exp0.get_explanation_select().disabled
    exp2 = ExplanationValues(X_df, Y_ser, 'DT1', function)
    exp2.disable_selection(False)
    assert not (exp2.get_compute_menu().disabled and exp2.get_explanation_select().disabled)


def test_update_progress_linear():
    pass


def test_get_explanation_select():
    X_df = pd.DataFrame([[4, 7, 10],
                         [5, 8, 11],
                         [6, 9, 12]],
                        index=[1, 2, 3],
                        columns=['a', 'b', 'c'])
    Y_ser = pd.Series([1, 2, 3])

    def function():
        pass

    exp = ExplanationValues(X_df, Y_ser, 'DT', function)
    print(exp.get_explanation_select())
    print(type(exp.get_explanation_select()))
    # assert exp.get_explanation_select()


def test_explanation_select_changed():
    data = 'data'
    X_df = pd.DataFrame([[4, 7, 10],
                         [5, 8, 11],
                         [6, 9, 12]],
                        index=[1, 2, 3],
                        columns=['a', 'b', 'c'])
    Y_ser = pd.Series([1, 2, 3])

    def function(*args, **kwargs):
        pass

    exp = ExplanationValues(X_df, Y_ser, 'DT', function)

    exp.explanation_select_changed(None, None, data)
    assert exp.current_exp == data
    exp.explanation_select_changed(None, None, None)
    assert exp.current_exp == None

# test_init()  # OK
# test_initialize()  # not ok
# test_current_pv()  # commencé not OK
# test_has_user_exp()  # OK
# test_update_explanation_select()  # not ok à faire ou pas ?
# test_get_compute_menu()  # not OK
# test_get_explanation_select()  # celle du widget revenir plus tard
# test_compute_explanation()  # not ok
# test_update_progress_linear()  # not ok
# test_compute_btn_clicked()  # not ok
# test_disable_selection()  # ok
# test_explanation_select_changed()  # ok
