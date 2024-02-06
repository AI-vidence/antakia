import mock
import pandas as pd
import pytest

from antakia.gui.explanation_values import ExplanationValues
from antakia.gui.widgets import get_widget, app_widget
from tests.utils_fct import generate_ExplanationValues, generate_df_series_callable, test_progress_bar


def test_init():  # ajouter test click
    X, y, function = generate_df_series_callable()
    X_exp = pd.DataFrame([[1, 7, 10],
                          [5, 8, 11],
                          [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])

    X, y, _, exp_val, function = generate_ExplanationValues('DT')
    assert exp_val.X.equals(X)
    assert exp_val.y.equals(y)
    assert exp_val.model == 'DT'
    assert exp_val.explanations == {'Imported': None, 'SHAP': None, 'LIME': None}
    assert exp_val.current_exp == 'SHAP'
    assert exp_val.on_change_callback == function

    X, y, X_exp, exp_val, _ = generate_ExplanationValues('DT', X_exp)
    assert exp_val.current_exp == 'Imported'
    assert exp_val.explanations['Imported'].X.equals(X_exp)
    assert exp_val.explanations['SHAP'] is None
    assert exp_val.explanations['LIME'] is None

    # get_widget(app_widget.widget, "13000203").click()
    # get_widget(app_widget.widget, "13000303").click()


@mock.patch('antakia.gui.explanation_values.compute_explanations')
def test_initialize(cpt_exp):
    X, y, function = generate_df_series_callable()
    X_exp = pd.DataFrame([[1, 7, 10],
                          [5, 8, 11],
                          [6, 9, 12]],
                         index=[1, 2, 3],
                         columns=['a', 'b', 'c'])
    cpt_exp.return_value = X_exp

    exp_val = generate_ExplanationValues('DT', X_exp)[3]
    exp_val.initialize(function)
    assert get_widget(app_widget.widget, '130000').disabled == (
                exp_val.explanations[exp_val.available_exp[1]] is not None)
    assert get_widget(app_widget.widget, '13000203').disabled == (
                exp_val.explanations[exp_val.available_exp[1]] is not None)
    assert get_widget(app_widget.widget, '130001').disabled == (
                exp_val.explanations[exp_val.available_exp[2]] is not None)
    assert get_widget(app_widget.widget, '13000303').disabled == (
                exp_val.explanations[exp_val.available_exp[2]] is not None)

    assert exp_val.current_exp == exp_val.available_exp[0]
    assert exp_val.get_explanation_select().v_model == 'Imported'

    exp_val1 = generate_ExplanationValues('DT')[3]
    exp_val1.initialize(test_progress_bar.update)
    assert exp_val1.get_explanation_select().v_model == 'SHAP'
    assert test_progress_bar.progress == 100


def test_current_pv():
    X_df, Y_ser, function, exp, _ = generate_ExplanationValues('DT')
    assert exp.current_pv == exp.explanations[exp.current_exp]


def test_has_user_exp():
    X_df, Y_ser, function, exp, _ = generate_ExplanationValues('DT', 'SHAP')
    assert exp.has_user_exp

    X_df, Y_ser, function, exp, _ = generate_ExplanationValues('DT')
    assert not exp.has_user_exp


@mock.patch('antakia.gui.explanation_values.compute_explanations')
def test_update_explanation_select(cpt_exp):
    X_df, Y_ser, function, exp, _ = generate_ExplanationValues('DT', 1)
    cpt_exp.return_value = X_df
    assert exp.get_explanation_select().items == [{"text": 'Imported', 'disabled': False},
                                                  {"text": 'SHAP', 'disabled': True},
                                                  {"text": 'LIME', 'disabled': True}]


@mock.patch('antakia.gui.explanation_values.compute_explanations')
def test_compute_explanation(cpt_exp):
    X_df, Y_ser, function, exp, _ = generate_ExplanationValues('DT', 1)
    cpt_exp.return_value = X_df
    assert exp.current_exp == exp.available_exp[0]

    assert exp.explanations['Imported'].X == 1
    assert exp.explanations['SHAP'] is None
    assert exp.explanations['LIME'] is None

    exp.compute_explanation(1, test_progress_bar.update)

    assert test_progress_bar.progress == 100
    assert exp.get_explanation_select().items == [{"text": 'Imported', 'disabled': False},
                                                  {"text": 'SHAP', 'disabled': False},
                                                  {"text": 'LIME', 'disabled': True}]

    assert get_widget(app_widget.widget, '130000').disabled == (exp.explanations[exp.available_exp[1]] is not None)
    assert get_widget(app_widget.widget, '13000203').disabled == (exp.explanations[exp.available_exp[1]] is not None)
    assert get_widget(app_widget.widget, '130001').disabled == (exp.explanations[exp.available_exp[2]] is not None)
    assert get_widget(app_widget.widget, '13000303').disabled == (exp.explanations[exp.available_exp[2]] is not None)


def test_update_compute_menu():
    X_df, Y_ser, function, exp, _ = generate_ExplanationValues('DT', 'SHAP')
    exp.update_compute_menu()
    assert get_widget(app_widget.widget, '130000').disabled == (exp.explanations[exp.available_exp[1]] is not None)
    assert get_widget(app_widget.widget, '13000203').disabled == (exp.explanations[exp.available_exp[1]] is not None)
    assert get_widget(app_widget.widget, '130001').disabled == (exp.explanations[exp.available_exp[2]] is not None)
    assert get_widget(app_widget.widget, '13000303').disabled == (exp.explanations[exp.available_exp[2]] is not None)


def test_compute_btn_clicked():  # à compléter
    X, y, X_exp, exp_val, _ = generate_ExplanationValues()
    # exp_val.compute_btn_clicked(get_widget(app_widget.widget, "130000"), None, None)
    # exp_val.compute_btn_clicked(None, None, None)
    # assert get_widget(app_widget.widget, "13000203").disabled
    # assert exp_val.current_exp == 1


def test_disable_selection():
    X_df, Y_ser, function, exp, _ = generate_ExplanationValues()
    exp0 = ExplanationValues(X_df, Y_ser, 'DT', function)
    exp0.disable_selection(True)
    assert exp0.get_compute_menu().disabled and exp0.get_explanation_select().disabled
    exp2 = ExplanationValues(X_df, Y_ser, 'DT1', function)
    exp2.disable_selection(False)
    assert not (exp2.get_compute_menu().disabled and exp2.get_explanation_select().disabled)


def test_explanation_select_changed():
    data = 'SHAP'
    X, y, X_exp, exp_val, _ = generate_ExplanationValues()
    exp_val.explanation_select_changed(None, None, data)
    assert exp_val.current_exp == data
    with pytest.raises(KeyError):
        exp_val.explanation_select_changed(None, None, None)
    exp_val.explanation_select_changed(None, None, 'LIME')
    assert exp_val.current_exp == 'LIME'
    # tester l'appel de on_change_callback
