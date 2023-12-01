import pandas as pd
# from dotenv import load_dotenv
from antakia.antakia import AntakIA
from sklearn.ensemble import GradientBoostingRegressor
from antakia.compute.dim_reduction.dim_reduction import compute_projection
from antakia.compute.auto_cluster.auto_cluster import AutoCluster
from antakia.compute.skope_rule.skope_rule import skope_rules
from antakia.data_handler.projected_values import ProjectedValues
from antakia.utils.variable import Variable
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.data_handler.rules import Rule
from antakia.utils.utils import in_index


# @mock.patch('antakia.antakia.AntakIA._get_shap_values')
def test_main():
    # _get_shap_values.return_value = pd.DataFrame()
    df = pd.read_csv('../../antakia/data/california_housing.csv').drop(['Unnamed: 0'], axis=1)

    # Remove outliers:
    df = df.loc[df['Population'] < 10000]
    df = df.loc[df['AveOccup'] < 6]
    df = df.loc[df['AveBedrms'] < 1.5]
    df = df.loc[df['HouseAge'] < 50]

    X = df.iloc[:, 0:8]  # the dataset
    y = df.iloc[:, 9]  # the target variable
    shap_values = df.iloc[:, [10, 11, 12, 13, 14, 15, 16, 17]]  # the SHAP values
    model = GradientBoostingRegressor(random_state=9)
    model.fit(X, y)

    variables_df = pd.DataFrame(
        {'col_index': [0, 1, 2, 3, 4, 5, 6, 7],
         'descr': ['Median income', 'House age', 'Average nb rooms', 'Average nb bedrooms', 'Population',
                   'Average occupancy', 'Latitude', 'Longitude'],
         'type': ['float64', 'int', 'float64', 'float64', 'int', 'float64', 'float64', 'float64'],
         'unit': ['k$', 'years', 'rooms', 'rooms', 'people', 'ratio', 'degrees', 'degrees'],
         'critical': [True, False, False, False, False, False, False, False],
         'lat': [False, False, False, False, False, False, True, False],
         'lon': [False, False, False, False, False, False, False, True]},
        index=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    )

    atk = AntakIA(X, y, model, variables_df, shap_values)

    # Functions

    def compare_indexes(df1, df2) -> bool:
        return df1.index.equals(df2.index)

    # Test Dim Reduction --------------------

    def dr_callback(*args):
        pass

    vs_pv = ProjectedValues(atk.X)
    vs_pv.set_proj_values(
        DimReducMethod.dimreduc_method_as_int("PaCMAP"),
        2,
        compute_projection(
            atk.X,
            atk.y,
            DimReducMethod.dimreduc_method_as_int("PaCMAP"),
            2,
            dr_callback
        )
    )

    assert vs_pv.get_proj_values(DimReducMethod.dimreduc_method_as_int('PaCMAP'), 2).shape == (atk.X.shape[0], 2)
    assert compare_indexes(vs_pv.get_proj_values(DimReducMethod.dimreduc_method_as_int('PaCMAP'), 2), atk.X) is True

    # es_pv_imported = ProjectedValues(atk.X_exp)
    # es_pv_imported.set_proj_values(
    # 	DimReducMethod.dimreduc_method_as_int("TSNE"),
    # 	3,
    # 	compute_projection(
    # 		atk.X_exp,
    # 		DimReducMethod.dimreduc_method_as_int("TSNE"),
    # 		3,
    # 		dr_callback
    # 	)
    # )

    # assert es_pv_imported.get_proj_values(DimReducMethod.dimreduc_method_as_int('TSNE'), 3).shape == (atk.X_exp.shape[0], 3)
    # assert compare_indexes(es_pv_imported.get_proj_values(DimReducMethod.dimreduc_method_as_int('TSNE'), 3), atk.X) is True

    # # Test explanation computation ---------

    # def exp_callback(*args):
    # 	pass

    # es_pv_shap = ProjectedValues(
    # 	compute_explanations(atk.X, atk.model, ExplanationMethod.SHAP, exp_callback)
    # )

    # assert es_pv_shap.X.shape == (atk.X.shape[0], atk.X.shape[1])
    # assert compare_indexes(es_pv_shap.X, atk.X) is True

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
