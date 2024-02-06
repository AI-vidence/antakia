from collections import namedtuple

import numpy as np
import pandas as pd
# from dotenv import load_dotenv
from antakia.antakia import AntakIA
from antakia.compute.auto_cluster.auto_cluster import AutoCluster
from antakia.compute.skope_rule.skope_rule import skope_rules
from antakia.data_handler.projected_values import ProjectedValues
from antakia.gui.widgets import get_widget, app_widget, splash_widget
from antakia.utils.dummy_datasets import load_dataset
from antakia.utils.variable import Variable
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.data_handler.rules import Rule
from antakia.utils.utils import in_index, mask_to_rows
from tests.interactions import *
from tests.status_checks import check_all
from tests.utils_fct import compare_indexes
from sklearn.tree import DecisionTreeRegressor


# @mock.patch('antakia.antakia.AntakIA._get_shap_values')
def test_main():
    X, y = load_dataset('Corner', 1000, random_seed=42)
    X = pd.DataFrame(X, columns=['X1', 'X2'])
    X['X3'] = np.random.random(len(X))
    y = pd.Series(y)

    model = DecisionTreeRegressor().fit(X, y)
    x_exp = pd.concat([(X.iloc[:, 0] > 0.5) * 0.5, (X.iloc[:, 1] > 0.5) * 0.5, (X.iloc[:, 2] > 2) * 1], axis=1)

    atk = AntakIA(X, y, model, X_exp=x_exp)
    run_antakia(atk)


def run_antakia(atk: AntakIA, check=True):
    atk.start_gui()
    # assert both progress bar are full after start up
    assert get_widget(splash_widget, '110').v_model == 100
    assert get_widget(splash_widget, '210').v_model == 100

    gui = atk.gui
    check_all(gui, check)
    for color in range(3):
        set_color(gui, color)
        check_all(gui, check)
    for exp in range(3):
        if exp == 0 and atk.X_exp is not None:
            set_exp_method(gui, exp)
            check_all(gui, check)
        else:
            if not (atk.X_exp is None and exp == 1):
                compute_exp_method(gui, exp)
            check_all(gui, check)
    for exp in range(3):
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

