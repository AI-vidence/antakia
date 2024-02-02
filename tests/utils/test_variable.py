import pandas as pd
import pytest

from antakia.utils.variable import Variable, DataVariables, var_from_symbol


def test_init_variable():
    var = Variable(0, 'var1', 'int', unit='seconds', descr='description', critical=True, continuous=True, lat=True,
                   lon=True)

    assert var.col_index == 0
    assert var.symbol == 'var1'
    assert var.type == 'int'
    assert var.unit == 'seconds'
    assert var.descr == 'description'
    assert var.critical
    assert var.continuous
    assert var.lat
    assert var.lon

    var = Variable(0, 'var2', 'int64')

    assert var.col_index == 0
    assert var.symbol == 'var2'
    assert var.type == 'int64'
    assert not var.critical
    assert var.continuous
    assert not var.lat
    assert not var.lon


def test_guess_variables():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame({"a": [4, 5, 6], "b": [7, 8, 9], "c": [10, 11, 12]})
    df3 = pd.DataFrame({"lat": [5]})
    df4 = pd.DataFrame({"long": [5]})

    assert Variable.guess_variables(df1) == DataVariables([])

    assert Variable.guess_variables(df2) == DataVariables(
        [Variable(0, 'a', 'int64', continuous=True),
         Variable(1, 'b', 'int64', continuous=True),
         Variable(2, 'c', 'int64', continuous=True)])

    assert Variable.guess_variables(df3) == DataVariables(
        [Variable(0, 'lat', 'int64', lat=True, continuous=True)])

    assert Variable.guess_variables(df4) == DataVariables(
        [Variable(0, 'long', 'int64', lon=True, continuous=True)])


def test_import_variable_df():
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

    assert Variable.import_variable_df(variables_df) == DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
         Variable(2, 'AveRooms', 'float64', descr='Average nb rooms', unit='rooms'),
         Variable(3, 'AveBedrms', 'float64', descr='Average nb bedrooms', unit='rooms'),
         Variable(4, 'Population', 'int', descr='Population', unit='people'),
         Variable(5, 'AveOccup', 'float64', descr='Average occupancy', unit='ratio'),
         Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])

    variables_df1 = pd.DataFrame(
        {'col_index': [0],
         'type': ['float64']},
        index=['MedInc']
    )

    assert Variable.import_variable_df(variables_df1) == DataVariables([Variable(0, 'MedInc', 'float64')])

    variables_df2 = pd.DataFrame({'colonne': [0],
                                  'type': ['float64']},
                                 index=['MedInc'])

    assert Variable.import_variable_df(variables_df2) == DataVariables([Variable(0, 'MedInc', 'float64')])

    with pytest.raises(KeyError):
        Variable.import_variable_df(variables_df1.drop('symbol', axis=1).reset_index(drop=True))
    with pytest.raises(KeyError):
        Variable.import_variable_df(variables_df1.drop('type', axis=1))


def test_import_variable_list():
    list_var = [{'col_index': 0, "symbol": 'a', 'type': 'int64'},
                {'col_index': 1, "symbol": 'b', 'type': 'int64'},
                {'col_index': 2, "symbol": 'c', 'type': 'int64'}]

    assert Variable.import_variable_list(list_var) == DataVariables(
        [Variable(0, 'a', 'int64'),
         Variable(1, 'b', 'int64'),
         Variable(2, 'c', 'int64')])

    list_var1 = [{'colonne_index': 0, "symbole": 'a', 'type_de_variable': 'int64'}]

    with pytest.raises(ValueError):
        Variable.import_variable_list(list_var1)


def test_repr():
    var1 = Variable(0, 'var1', 'int')
    assert repr(var1) == "var1, col#:0, type:int"

    var2 = Variable(0, 'var2', 'int', unit='seconds', descr='description', critical=True, continuous=False, lat=True,
                    lon=True)
    assert repr(
        var2) == "var2, col#:0, type:int, descr:description, unit:seconds, critical, categorical, is lat, is lon"


def test_str_dv():
    dv = DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])

    assert str(dv) == ('0) MedInc, col#:0, type:float64, descr:Median income, unit:k$, critical\n'
                       '6) Latitude, col#:6, type:float64, descr:Latitude, unit:degrees, is lat\n'
                       '7) Longitude, col#:7, type:float64, descr:Longitude, unit:degrees, is lon\n')


def test_sym_list():
    dv = DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
         Variable(2, 'AveRooms', 'float64', descr='Average nb rooms', unit='rooms'),
         Variable(3, 'AveBedrms', 'float64', descr='Average nb bedrooms', unit='rooms'),
         Variable(4, 'Population', 'int', descr='Population', unit='people'),
         Variable(5, 'AveOccup', 'float64', descr='Average occupancy', unit='ratio'),
         Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])

    assert dv.sym_list() == ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude',
                             'Longitude']


def test_get_var():
    dv = DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
         Variable(2, 'AveRooms', 'float64', descr='Average nb rooms', unit='rooms'),
         Variable(3, 'AveBedrms', 'float64', descr='Average nb bedrooms', unit='rooms'),
         Variable(4, 'Population', 'int', descr='Population', unit='people'),
         Variable(5, 'AveOccup', 'float64', descr='Average occupancy', unit='ratio'),
         Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])

    assert dv.get_var('MedInc') == Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True)


def test_len_dv():
    dv = DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
         Variable(2, 'AveRooms', 'float64', descr='Average nb rooms', unit='rooms'),
         Variable(3, 'AveBedrms', 'float64', descr='Average nb bedrooms', unit='rooms'),
         Variable(4, 'Population', 'int', descr='Population', unit='people'),
         Variable(5, 'AveOccup', 'float64', descr='Average occupancy', unit='ratio'),
         Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])

    assert len(dv) == 8


def test_eq_dv():
    dv = DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
         Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])
    dv1 = DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(3, 'AveBedrms', 'float64', descr='Average nb bedrooms', unit='rooms'),
         Variable(4, 'Population', 'int', descr='Population', unit='people'),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])

    assert not dv == dv1

    dv2 = DataVariables(
        [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
         Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
         Variable(3, 'AveBedrms', 'float64', descr='Average nb bedrooms', unit='rooms'),
         Variable(4, 'Population', 'int', descr='Population', unit='people'),
         Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
         Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)])

    assert not dv1 == dv2


def test_var_from_symbol():
    var_list = [Variable(0, 'MedInc', 'float64', descr='Median income', unit='k$', critical=True),
                Variable(1, 'HouseAge', 'int', descr='House age', unit='years'),
                Variable(2, 'AveRooms', 'float64', descr='Average nb rooms', unit='rooms'),
                Variable(3, 'AveBedrms', 'float64', descr='Average nb bedrooms', unit='rooms'),
                Variable(4, 'Population', 'int', descr='Population', unit='people'),
                Variable(5, 'AveOccup', 'float64', descr='Average occupancy', unit='ratio'),
                Variable(6, 'Latitude', 'float64', descr='Latitude', unit='degrees', lat=True),
                Variable(7, 'Longitude', 'float64', descr='Longitude', unit='degrees', lon=True)]

    assert var_from_symbol(var_list, 'AveRooms') == Variable(2, 'AveRooms', 'float64', descr='Average nb rooms',
                                                             unit='rooms')
    assert var_from_symbol(var_list, 'Colonne') is None
