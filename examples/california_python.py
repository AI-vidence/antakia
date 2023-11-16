import pandas as pd
df = pd.read_csv('../data/california_housing.csv').drop(['Unnamed: 0'], axis=1)

# Remove outliers:
df = df.loc[df['Population']<10000]
df = df.loc[df['AveOccup']<6]
df = df.loc[df['AveBedrms']<1.5]
df = df.loc[df['HouseAge']<50]

# Only San Francisco :
df = df.loc[(df['Latitude']<38.07)&(df['Latitude']>37.2)]
df = df.loc[(df['Longitude']>-122.5)&(df['Longitude']<-121.75)]

X = df.iloc[:,0:8] # the dataset
y = df.iloc[:,9] # the target variable
shapValues = df.iloc[:,[10,11,12,13,14,15,16,17]] # the SHAP values

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state = 9)
model.fit(X, y)

from antakia.data import ExplanationMethod

from antakia.antakia import AntakIA

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
# We call AntakIA with already computed SHAP values and a description of X variables :
atk = AntakIA(X, y, model, variables_df, shapValues)

# Here we provide the bare minimum to AntakIA :
# atk = AntakIA([X], y, model, [ExplanationMethod.NONE])

atk.start_gui()