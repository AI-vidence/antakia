import mock

from antakia.gui.explanation_values import ExplanationValues
from antakia.gui.widgets import get_widget, app_widget
from tests.utils_fct import generate_ExplanationValues, EMPTYExplanation


def test_init():
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 'SHAP')

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
    X_df, Y_ser, X_exp, exp_val = generate_ExplanationValues('DT')
    function = lambda *args: None
    exp_val.initialize(function)
    # cpt_exp.return_value = X_df
    a = exp_val.on_change_callback
    a1 = function
    # tester que compute_explanation est bien appelé
    # assert exp_val.current_exp == exp_val.available_exp[0]
    # tester avec SHAP en current exp
    # assert exp_val.get_explanation_select().v_model == exp_val.current_exp
    z = 1

    # assert exp.get_explanation_select().v_model ==


def test_current_pv():
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT')
    assert exp.current_pv == exp.explanations[exp.current_exp]


def test_has_user_exp():
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 'SHAP')
    assert exp.has_user_exp

    X_df, Y_ser, function, exp = generate_ExplanationValues('DT')
    assert not exp.has_user_exp


@mock.patch('antakia.gui.explanation_values.compute_explanations')
def test_update_explanation_select(cpt_exp):
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 1)
    cpt_exp.return_value = X_df
    assert exp.get_explanation_select().items == [{"text": 'Imported', 'disabled': False},
                                                  {"text": 'SHAP', 'disabled': True},
                                                  {"text": 'LIME', 'disabled': True}]


@mock.patch('antakia.gui.explanation_values.compute_explanations')
def test_compute_explanation(cpt_exp):
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 1)
    cpt_exp.return_value = X_df
    assert exp.current_exp == exp.available_exp[0]

    assert exp.explanations['Imported'].X == 1
    assert exp.explanations['SHAP'] is None
    assert exp.explanations['LIME'] is None

    exp.compute_explanation(1, None)
    assert exp.get_explanation_select().items == [{"text": 'Imported', 'disabled': False},
                                                  {"text": 'SHAP', 'disabled': False},
                                                  {"text": 'LIME', 'disabled': True}]

    # assert que update_compute_menu est bien fait


def test_update_compute_menu():
    X_df, Y_ser, function, exp = generate_ExplanationValues('DT', 'SHAP')
    exp.update_compute_menu()
    assert get_widget(app_widget, '130000').disabled == (exp.explanations[exp.available_exp[1]] is not None)
    assert get_widget(app_widget, '13000203').disabled == (exp.explanations[exp.available_exp[1]] is not None)
    assert get_widget(app_widget, '130001').disabled == (exp.explanations[exp.available_exp[2]] is not None)
    assert get_widget(app_widget, '13000303').disabled == (exp.explanations[exp.available_exp[2]] is not None)


def test_compute_btn_clicked():
    X, y, X_exp, exp_val = generate_ExplanationValues()
    # exp_val.compute_btn_clicked(get_widget(app_widget, "130000"), None, None)
    # exp_val.compute_btn_clicked(None, None, None)
    # assert get_widget(app_widget, "13000203").disabled
    a = 1
    # assert exp_val.current_exp == 1


def test_disable_selection():
    X_df, Y_ser, function, exp = generate_ExplanationValues()
    exp0 = ExplanationValues(X_df, Y_ser, 'DT', function)
    exp0.disable_selection(True)
    assert exp0.get_compute_menu().disabled and exp0.get_explanation_select().disabled
    exp2 = ExplanationValues(X_df, Y_ser, 'DT1', function)
    exp2.disable_selection(False)
    assert not (exp2.get_compute_menu().disabled and exp2.get_explanation_select().disabled)


def test_update_progress_linear():
    b = get_widget(app_widget, "13000201")
    X, y, X_exp, exp_val = generate_ExplanationValues()
    exp_meth_shap = EMPTYExplanation(X, y, X_exp, exp_val)
    exp_meth_lime = EMPTYExplanation(X, y, X_exp, exp_val)
    exp_meth_lime.explanation_method = 2

    # Test SHAP incomplet
    exp_val.update_progress_linear(exp_meth_shap, 2)
    assert get_widget(app_widget, "13000201").indeterminate
    assert get_widget(app_widget, "13000201").v_model == 2

    # test LIME incomplet
    exp_val.update_progress_linear(exp_meth_lime, 2)
    assert get_widget(app_widget, "13000301").v_model == 2

    # Test SHAP complet
    exp_val.update_progress_linear(exp_meth_shap, 100)
    assert not get_widget(app_widget, "13000201").indeterminate
    assert get_widget(app_widget, "130000").disabled

    # test LIME complet

    exp_val.update_progress_linear(exp_meth_lime, 100)
    assert get_widget(app_widget, "13000301").v_model == 100
    assert get_widget(app_widget, "130001").disabled


def test_explanation_select_changed():
    data = 'data'
    X, y, X_exp, exp_val = generate_ExplanationValues()
    exp_val.explanation_select_changed(None, None, data)
    assert exp_val.current_exp == data
    exp_val.explanation_select_changed(None, None, None)
    assert exp_val.current_exp == None
    # tester l'appel de on_change_callback

# test_init()  # OK sauf click bouton
# test_initialize()  # not ok

# test_compute_btn_clicked()  # not ok
