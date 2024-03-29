import random
from unittest import TestCase

import mock
import numpy as np
import pandas as pd
import pytest

from antakia.antakia import AntakIA
from antakia.utils.dummy_datasets import load_dataset
from tests.interactions import *
from tests.status_checks import check_all
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from antakia import config
from tests.utils_fct import DummyModel


class TestAntakia(TestCase):

    @classmethod
    def setUpClass(cls):
        config.ATK_MIN_POINTS_NUMBER = 10
        config.ATK_MAX_DOTS = 100

        X, y = load_dataset('Corner', 1000, random_seed=42)
        cls.X = pd.DataFrame(X, columns=['X1', 'X2'])
        cls.X['X3'] = np.random.random(len(X))
        cls.y = pd.Series(y)

        X_test, y_test = load_dataset('Corner', 100, random_seed=56)
        cls.X_test = pd.DataFrame(X_test, columns=['X1', 'X2'])
        cls.X_test['X3'] = np.random.random(len(X_test))
        cls.y_test = pd.Series(y_test)

        cls.regression_DT = DecisionTreeRegressor().fit(cls.X, cls.y)
        cls.regression_DT_np = DecisionTreeRegressor().fit(
            cls.X.values, cls.y.values)
        cls.regression_any = DummyModel()
        cls.classifier_DT = DecisionTreeClassifier().fit(cls.X, cls.y)
        cls.x_exp = pd.concat([(cls.X.iloc[:, 0] > 0.5) * 0.5,
                               (cls.X.iloc[:, 1] > 0.5) * 0.5,
                               (cls.X.iloc[:, 2] > 2) * 1],
                              axis=1)

    def test_vanilla_run(self):
        # vanilla run
        atk = AntakIA(self.X, self.y, self.regression_DT)
        run_antakia(atk, True)

    def test_vanilla_run_test(self):
        # vanilla run
        atk = AntakIA(self.X,
                      self.y,
                      self.regression_DT,
                      X_test=self.X_test,
                      y_test=self.y_test)
        run_antakia(atk, True)

    def test_shape_issue(self):
        # shape issue
        with pytest.raises(AssertionError):
            atk = AntakIA(self.X,
                          self.y.iloc[:10],
                          self.regression_DT,
                          X_exp=self.x_exp)
            run_antakia(atk, False)

    def test_vanilla_with_exp(self):
        # run with explanations
        atk = AntakIA(self.X, self.y, self.regression_DT, X_exp=self.x_exp)
        run_antakia(atk, True)

    def test_vanilla_with_non_Tree_and_exp(self):
        # run with non tree model
        atk = AntakIA(self.X, self.y, self.regression_any, X_exp=self.x_exp)
        run_antakia(atk, False)

    def test_vanilla_with_non_Tree_and_no_exp(self):
        # run with non tree model and no x_exp
        atk = AntakIA(self.X, self.y, self.regression_any)
        run_antakia(atk, False)

    def test_with_np_arrays(self):
        # run with np array
        atk = AntakIA(self.X.values, self.y.values, self.regression_DT_np)
        run_antakia(atk, False)

    def test_partial_np_array(self):
        # run with partial np array
        atk = AntakIA(self.X, self.y.values, self.regression_DT)
        run_antakia(atk, False)

    def test_partial_np_array2(self):
        # run with partial np array
        atk = AntakIA(self.X.values, self.y, self.regression_DT_np)
        run_antakia(atk, False)

    def test_with_np_arrays_exp(self):
        # run with np array and x_exp
        atk = AntakIA(self.X.values,
                      self.y.values,
                      self.regression_DT_np,
                      X_exp=self.x_exp.values)
        run_antakia(atk, False)

    def test_y_as_df(self):
        # run y as df
        atk = AntakIA(self.X, self.y.to_frame(), self.regression_DT)
        run_antakia(atk, False)

    def test_col_names_numeric(self):
        # run with numerical cols
        X2 = self.X.copy()
        X2.columns = range(len(self.X.T))
        model_DT = DecisionTreeRegressor().fit(X2, self.y)

        atk = AntakIA(X2, self.y, model_DT)
        run_antakia(atk, False)

    def test_random(self):
        for _ in range(10):
            atk = AntakIA(self.X, self.y, self.regression_DT)
            random_walk(atk, 20)

    def test_run_walk(self):
        atk = AntakIA(self.X, self.y, self.regression_DT)
        walk = [('change_tab', [1]), ('auto_cluster', []),
                ('toggle_select_region', [1]), ('edit', []),
                ('select_points', [0])]
        walk = [('select_points', [0])]
        run_walk(atk, walk)

    def test_classifier(self):
        atk = AntakIA(self.X, self.y, self.classifier_DT)
        run_antakia(atk, True)


def dummy_projection(_X, y, method, dim, callback, *args, **kwargs):
    callback(100, 0)
    return pd.DataFrame(_X.values[:, :dim], index=_X.index)


def dummy_exp(_X, model, method, task_type, callback, *args, **kwargs):
    callback(100, 0)
    return pd.DataFrame(_X.values, index=_X.index, columns=_X.columns)


@mock.patch('antakia.gui.app_bar.explanation_values.compute_explanations',
            wraps=dummy_exp)
@mock.patch('antakia_core.data_handler.projected_values.compute_projection',
            wraps=dummy_projection)
def run_antakia(atk: AntakIA, check, compute_proj, compute_exp):
    atk.start_gui()
    # assert both progress bar are full after start up

    gui = atk.gui
    assert get_widget(gui.splash.widget, '110').v_model == 100
    assert get_widget(gui.splash.widget, '210').v_model == 100
    if check:
        check_all(gui)
    # change colors
    for color in range(3):
        set_color(gui, color, check=check)
    # iterate over compute exp and projection
    # note compute and project are mocked
    if atk.X_exp is None:
        exp_range = [1, 2]
        compute = [2]
    else:
        exp_range = [0, 1, 2]
        compute = [1, 2]

    for exp in exp_range:
        set_exp_method(gui, exp, check=check)

    for proj in range(3):
        set_proj_method(gui, True, proj, check=check)
        if proj:
            edit_parameter(gui, True, check=check)
        set_proj_method(gui, False, proj, check=check)
    # iterate through tabs
    for tab in range(3):
        change_tab(gui, tab, check=check)
    # manipoulate selection
    change_tab(gui, 0, check=check)
    select_points(gui, True, check=check)
    select_points(gui, False, check=check)
    unselect(gui, False, check=check)
    # select points, find rules, validate
    select_points(gui, True, check=check)
    find_rules(gui, check=check)
    validate_rules(gui, check=check)
    # autocluster - select - subdivide
    auto_cluster(gui, check=check)
    toggle_select_region(gui, 1, check=check)
    subdivide(gui, check=check)
    # merge
    toggle_select_region(gui, 1, check=check)
    toggle_select_region(gui, 2, check=check)
    merge(gui, check=check)
    clear_region_selection(gui)
    # select - substitute - select model
    toggle_select_region(gui, 1, check=check)
    substitute(gui, check=check)
    select_model(gui, 0, check=check)
    validate_model(gui, check=check)


actions = {
    'select_dim': (select_dim, range(2)),
    'set_color': (set_color, range(3)),
    'set_exp_method': (set_exp_method, range(3)),
    'set_proj_method': (set_proj_method, range(2), range(3)),
    'edit_parameter': (edit_parameter, range(2)),
    'change_tab': (change_tab, range(3)),
    'select_points': (select_points, range(2)),
    'unselect': (unselect, range(2)),
    'find_rules': (find_rules, ),
    'validate_rules': (validate_rules, ),
    'auto_cluster': (auto_cluster, ),
    'toggle_select_region': (toggle_select_region, range(4)),
    'subdivide': (subdivide, ),
    'merge': (merge, ),
    'edit': (edit, ),
    'clear_selection': (clear_region_selection, ),
    'substitute': (substitute, ),
    'select_model': (select_model, range(10)),
    'validate_model': (validate_model, )
}


@mock.patch('antakia.gui.app_bar.explanation_values.compute_explanations',
            wraps=dummy_exp)
@mock.patch('antakia_core.data_handler.projected_values.compute_projection',
            wraps=dummy_projection)
def random_walk(atk, steps, compute_proj, compute_exp):
    atk.start_gui()
    gui = atk.gui

    k = 0
    walk = []
    action_list = list(actions.keys())
    while k <= steps:
        action = action_list[random.randint(0, len(actions) - 1)]
        action_fct, *action_param = actions[action]
        if len(action_param) > 0:
            params = [random.choice(value_list) for value_list in action_param]
        else:
            params = []
        try:
            action_fct(gui, *params, check=True)
            walk.append((action, params))
            k += 1
        except InteractionError:
            pass
        except:
            walk.append((action, params))
            print(walk)
            raise


@mock.patch('antakia.gui.app_bar.explanation_values.compute_explanations',
            wraps=dummy_exp)
@mock.patch('antakia_core.data_handler.projected_values.compute_projection',
            wraps=dummy_projection)
def run_walk(atk, walk, compute_proj, compute_exp):
    atk.start_gui()
    gui = atk.gui
    k = 0
    for action, params in walk:
        action_fct, *_ = actions[action]
        action_fct(gui, *params, check=True)
        k += 1
