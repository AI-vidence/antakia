import mock
import numpy as np
import pandas as pd
import pytest

from antakia.antakia import AntakIA
from antakia.gui.widgets import splash_widget, app_widget
from antakia.utils.dummy_datasets import load_dataset
from tests.interactions import *
from tests.status_checks import check_all
from sklearn.tree import DecisionTreeRegressor

X, y = load_dataset('Corner', 1000, random_seed=42)
X = pd.DataFrame(X, columns=['X1', 'X2'])
X['X3'] = np.random.random(len(X))
y = pd.Series(y)


class DummyModel:
    def predict(self, X):
        return ((X.iloc[:, 0] > 0.5) & (X.iloc[:, 1] > 0.5)).astype(int)

    def score(self, *args):
        return 1


model_DT = DecisionTreeRegressor().fit(X, y)
model_DT_np = DecisionTreeRegressor().fit(X.values, y.values)
model_any = DummyModel()
x_exp = pd.concat([(X.iloc[:, 0] > 0.5) * 0.5, (X.iloc[:, 1] > 0.5) * 0.5, (X.iloc[:, 2] > 2) * 1], axis=1)


def test_1():
    # vanilla run
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X, y, model_DT)
    run_antakia(atk, True)


def test_2():
    # shape issue
    splash_widget.reset()
    app_widget.reset()
    with pytest.raises(AssertionError):
        atk = AntakIA(X, y.iloc[:10], model_DT, X_exp=x_exp)
        run_antakia(atk, False)


def test_3():
    # run with explanations
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X, y, model_DT, X_exp=x_exp)
    run_antakia(atk, True)


def test_4():
    # run with non tree model
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X, y, model_any, X_exp=x_exp)
    run_antakia(atk, False)


def test_5():
    # run with non tree model and no x_exp
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X, y, model_any)
    run_antakia(atk, False)


def test_6():
    # run with np array
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X.values, y.values, model_DT_np)
    run_antakia(atk, False)


def test_7():
    # run with partial np array
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X, y.values, model_DT)
    run_antakia(atk, False)


def test_8():
    # run with partial np array
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X.values, y, model_DT_np)
    run_antakia(atk, False)


def test_9():
    # run with np array and x_exp
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X.values, y.values, model_DT_np, X_exp=x_exp.values)
    run_antakia(atk, False)


def test_10():
    # run y as df
    splash_widget.reset()
    app_widget.reset()
    atk = AntakIA(X, y.to_frame(), model_DT)
    run_antakia(atk, False)


def test_11():
    # run with numerical cols
    splash_widget.reset()
    app_widget.reset()
    X2 = X
    X2.columns = range(len(X.T))
    model_DT = DecisionTreeRegressor().fit(X2, y)

    atk = AntakIA(X2, y, model_DT)
    run_antakia(atk, False)


def dummy_projection(_X, y, method, dim, callback, *args, **kwargs):
    callback(100, 0)
    return pd.DataFrame(_X.values[:, :dim], index=_X.index)


def dummy_exp(_X, model, method, callback, *args, **kwargs):
    callback(100, 0)
    return pd.DataFrame(_X.values, index=_X.index, columns=_X.columns)


@mock.patch('antakia.gui.explanation_values.compute_explanations', wraps=dummy_exp)
@mock.patch('antakia.data_handler.projected_values.compute_projection', wraps=dummy_projection)
def run_antakia(atk: AntakIA, check, compute_proj, compute_exp):
    atk.start_gui()
    # assert both progress bar are full after start up
    assert get_widget(splash_widget.widget, '110').v_model == 100
    assert get_widget(splash_widget.widget, '210').v_model == 100

    gui = atk.gui
    check_all(gui, check)
    for color in range(3):
        set_color(gui, color)
        check_all(gui, check)
    if atk.X_exp is None:
        exp_range = [1, 2]
        compute = [2]
    else:
        exp_range = [0, 1, 2]
        compute = [1, 2]
    check_all(gui, check)

    for exp in compute:
        compute_exp_method(gui, exp)
        check_all(gui, check)
    for exp in exp_range:
        set_exp_method(gui, exp)
        check_all(gui, check)

    for proj in range(3):
        set_proj_method(gui, True, proj)
        check_all(gui, check)
        edit_parameter(gui, True)
        check_all(gui, check)
        set_proj_method(gui, False, proj)
        check_all(gui, check)

    for tab in range(3):
        change_tab(gui, tab)
        check_all(gui, check)

    change_tab(gui, 0)
    check_all(gui, check)
    select_points(gui, True)
    check_all(gui, check)
    select_points(gui, False)
    check_all(gui, check)
    unselect(gui, False)
    check_all(gui, check)

    select_points(gui, False)
    find_rules(gui)
    check_all(gui, check)
    validate_rules(gui)

    auto_cluster(gui)
    check_all(gui, check)
    toggle_select_region(gui, 1)
    check_all(gui, check)
    subdivide(gui)
    check_all(gui, check)
    toggle_select_region(gui, 1)
    check_all(gui, check)
    substitute(gui)
    check_all(gui, check)
    select_model(gui, 'Decision Tree')
    check_all(gui, check)
    validate_model(gui)
    check_all(gui, check)
