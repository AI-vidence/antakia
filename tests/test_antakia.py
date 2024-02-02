from collections import namedtuple

import numpy as np
import pandas as pd
# from dotenv import load_dotenv
from antakia.antakia import AntakIA
from sklearn.ensemble import GradientBoostingRegressor
from antakia.compute.dim_reduction.dim_reduction import compute_projection
from antakia.compute.auto_cluster.auto_cluster import AutoCluster
from antakia.compute.skope_rule.skope_rule import skope_rules
from antakia.data_handler.projected_values import ProjectedValues
from antakia.gui.widgets import get_widget, app_widget
from antakia.utils.dummy_datasets import load_dataset
from antakia.utils.variable import Variable
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.data_handler.rules import Rule
from antakia.utils.utils import in_index, mask_to_rows
from tests.utils import dr_callback, compare_indexes
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
    gui = atk.gui
    atk.start_gui()
    get_widget(app_widget, '13000203').click()  # click on compute shap
    gui.switch_dimension(None, None, True)
    gui.switch_dimension(None, None, False)

    selection = (X.iloc[:, 0] > 0.5)
    points = namedtuple('points', ['point_inds'])
    points(mask_to_rows(selection))

    gui.vs_hde._selection_event(None, points(mask_to_rows(selection)))  # select points
    get_widget(app_widget, "43010").click() # compute skope rules
    gui.vs_rules_wgt.rule_widget_list[0]._widget_value_changed('', '', 0.6) # change rule value

    vs_pv = ProjectedValues(atk.X, atk.y)
    proj_dim2 = vs_pv.get_projection(DimReducMethod.dimreduc_method_as_int("PCA"), 2)
    proj_dim3 = vs_pv.get_projection(DimReducMethod.dimreduc_method_as_int("PCA"), 3)
    assert proj_dim2.shape == (len(atk.X), 2)
    assert proj_dim3.shape == (len(atk.X), 3)
    assert compare_indexes(proj_dim2, atk.X)

    # Test Skope rules -------------

    vs_proj_df = vs_pv.get_proj_values(DimReducMethod.dimreduc_method_as_int('PaCMAP'), 2)

    x = vs_proj_df.iloc[:, 0]
    y = vs_proj_df.iloc[:, 1]
    selection_mask = (x > -22) & (x < -12) & (y > 10) & (y < 25)

    rules_list, score_dict = skope_rules(selection_mask, atk.X, atk.variables)

    skope_rules_indexes = Rule.rules_to_indexes(rules_list, atk.X)

    assert in_index(skope_rules_indexes, atk.X) is True

    # Test Rules --------------------

    variables = Variable.guess_variables(atk.X)
    pop_var = variables.get_var('Population')
    med_inc_var = variables.get_var('MedInc')

    rule_pop = Rule(500, "<=", pop_var, "<", 700)
    rule_med_inc = Rule(4.0, "<", med_inc_var, None, None)

    selection = Rule.rules_to_indexes([rule_pop, rule_med_inc], atk.X)
    assert in_index(selection, atk.X) is True

    # # Test auto cluster --------------------

    def ac_callback(*args):
        pass

    ac = AutoCluster(atk.X, ac_callback)
    found_clusters = ac.compute(atk.X_exp, 6)
    assert found_clusters.nunique() == 6

    print(f"Clusters {len(found_clusters)}")
    assert in_index(found_clusters.index, atk.X)


if __name__ == '__main__':
    test_main()
