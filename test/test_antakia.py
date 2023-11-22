import pandas as pd
# from dotenv import load_dotenv
from skrules import Rule
from antakia.antakia import AntakIA
from sklearn.ensemble import GradientBoostingRegressor
from antakia.compute.auto_cluster.auto_cluster import AutoCluster
from antakia.data import ProjectedValues, DimReducMethod, Variable
from antakia.rules import Rule
import mock

# @mock.patch('antakia.antakia.AntakIA._get_shap_values')
def test_main():
	# _get_shap_values.return_value = pd.DataFrame()
	df = pd.read_csv('../antakia/data/california_housing.csv').drop(['Unnamed: 0'], axis=1)

	# Remove outliers:
	df = df.loc[df['Population']<10000] 
	df = df.loc[df['AveOccup']<6]
	df = df.loc[df['AveBedrms']<1.5]
	df = df.loc[df['HouseAge']<50]

	X = df.iloc[:,0:8] # the dataset
	y = df.iloc[:,9] # the target variable
	shap_values = df.iloc[:,[10,11,12,13,14,15,16,17]] # the SHAP values
	model = GradientBoostingRegressor(random_state = 9)
	model.fit(X, y)

	variables_df = pd.DataFrame(
		{'col_index': [0, 1, 2, 3, 4, 5, 6, 7],
		'descr': ['Median income', 'House age', 'Average nb rooms', 'Average nb bedrooms', 'Population', 'Average occupancy', 'Latitude', 'Longitude'],
		'type': ['float64', 'int', 'float64', 'float64', 'int', 'float64', 'float64', 'float64'],
		'unit': ['k$', 'years', 'rooms', 'rooms', 'people', 'ratio', 'degrees', 'degrees'],
		'critical': [True, False, False, False, False, False, False, False],
		'lat': [False, False, False, False, False, False, True, False],
		'lon': [False, False, False, False, False, False, False, True]},
		index=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
		)

	atk = AntakIA(X, y, model, variables_df, shap_values)

	def my_callback(*args):
		pass

	vs_pv = ProjectedValues(atk.X)
	vs_pv.set_proj_values(DimReducMethod.dimreduc_method_as_int("PaCMAP"), 2, atk.X)

	es_pv = ProjectedValues(atk.X_exp)
	es_pv.set_proj_values(DimReducMethod.dimreduc_method_as_int("TSNE"), 2, atk.X_exp)

	variables = Variable.guess_variables(atk.X)
	pop_var = variables[4]
	med_inc_var = variables[0]

	rule_pop = Rule(500, "<=", pop_var, "<", 700)
	rule_med_inc = Rule(4.0, "<", med_inc_var, None, None)

	print(f"Rule 1: {rule_pop}, {len(rule_pop.get_matching_indexes(atk.X))} matching indexes")
	print(f"Rule 2: {rule_med_inc}, {len(rule_med_inc.get_matching_indexes(atk.X))} matching indexes")

	selection = Rule.rules_to_indexes([rule_pop, rule_med_inc], atk.X)
	print(f"Rules 1 and 2: {Rule.multi_rules_to_string([rule_pop, rule_med_inc])}, {len(selection)} matching indexes")

	# ac = AutoCluster(atk.X, my_callback)
	# found_clusters = ac.compute(
	# 	atk.X_exp,
	# 	'auto'
	# 	)

	# print(f"We found : {len(found_clusters)}")


if __name__ == '__main__':
	test_main()
