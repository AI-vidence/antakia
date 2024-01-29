from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class Variable:
    """
    Describes each X column or Y value

    col_index : int
        The index of the column in the dataframe i come from (ds or xds)
        #TODO : I shoudl code an Abstract class for Dataset and ExplanationDataset
    symbol : str
        How it should be displayed in the GUI
    descr : str
        A description of the variable
    type : str
        The type of the variable
    sensible : bool
        Wether the variable is sensible or not
    contiuous : bool
    lat : bool
    lon : bool
    """

    def __init__(
            self,
            col_index: int,
            symbol: str,
            type: str,
            unit: str = None,
            descr: str = None,
            critical: bool = False,
            continuous: bool = True,
            lat: bool = False,
            lon: bool = False,
            **kwargs  # to ignore unknown args in building object
    ):
        self.col_index = col_index
        self.symbol = symbol
        self.type = type
        self.unit = unit
        self.descr = descr
        self.critical = critical
        self.continuous = continuous
        self.lat = lat
        self.lon = lon

    @staticmethod
    def guess_variables(X: pd.DataFrame) -> 'DataVariables':
        """
        Returns a list of Variable objects, one for each column in X.
        """
        variables = []
        for i, col in enumerate(X.columns):
            col_2 = str(col)
            col_2.replace("_", " ")
            var = Variable(i, col_2, X.dtypes[col])
            if col_2.lower() in ["latitude", "lat"]:
                var.lat = True
            if col_2.lower() in ["longitude", "long"]:
                var.lon = True
            var.continuous = Variable.is_continuous(X[col])
            variables.append(var)
        return DataVariables(variables)

    @staticmethod
    def import_variable_df(df: pd.DataFrame) -> 'DataVariables':
        """
        Import variables from a DataFrame
        """

        if "col_index" not in df.columns:
            df['col_index'] = np.arange(len(df))
        if 'symbol' not in df.columns:
            df['symbol'] = df.index
            if is_numeric_dtype(df['symbol']):
                raise KeyError('symbol (index) column is mandatory and should be string')
        if 'type' not in df.columns:
            raise KeyError('type column is mandatory')
        variables = df.apply(lambda row: Variable(**row), axis=1).to_list()
        return DataVariables(variables)

    @staticmethod
    def import_variable_list(var_list: list) -> 'DataVariables':
        """
        Import variables from a list of dicts
        """
        variables = []
        for i in range(len(var_list)):
            if isinstance(var_list[i], dict):
                item = var_list[i]
                if "col_index" in item and "symbol" in item and "type" in item:
                    var = Variable(**item)
                    variables.append(var)
                else:
                    raise ValueError(
                        "Variable must a list of {key:value} with mandatory keys : [col_index, symbol, type] and optional keys : [unit, descr, critical, continuous, lat, lon]"
                    )
        return DataVariables(variables)

    @staticmethod
    def is_continuous(serie: pd.Series) -> bool:
        # TODO : precise continuous definition
        id_first_true = (serie > 0).idxmax()
        id_last_true = (serie > 0)[::-1].idxmax()
        return all((serie > 0).loc[id_first_true:id_last_true] == True)

    def __repr__(self):
        """
        Displays the variable as a string
        """
        text = f"{self.symbol}, col#:{self.col_index}, type:{self.type}"
        if self.descr is not None:
            text += f", descr:{self.descr}"
        if self.unit is not None:
            text += f", unit:{self.unit}"
        if self.critical:
            text += ", critical"
        if not self.continuous:
            text += ", categorical"
        if self.lat:
            text += ", is lat"
        if self.lon:
            text += ", is lon"
        return text

    def __eq__(self, other):
        return (
                self.col_index == other.col_index and
                self.symbol == other.symbol and
                self.type == other.type and
                self.unit == other.unit and
                self.descr == other.descr and
                self.critical == other.critical and
                self.continuous == other.continuous and
                self.lat == other.lat and
                self.lon == other.lon
        )


class DataVariables:
    def __init__(self, variables: List[Variable]):
        self.variables = {var.symbol: var for var in variables}

    def __str__(self):
        text = ""
        for var in self.variables.values():
            text += str(var.col_index) + ") " + str(var) + "\n"
        return text

    def sym_list(self):
        return list(self.variables.keys())

    def get_var(self, symbol: str):
        return self.variables.get(symbol)

    def __len__(self):
        return len(self.variables)

    def __eq__(self, other):
        for i in self.variables.values():
            if i not in other.variables.values():
                return False
        for j in other.variables.values():
            if j not in self.variables.values():
                return False
        return True
        # return set(list(self.variables.values())) == set(list(other.variables.values()))

def var_from_symbol(variables: List[Variable], token: str) -> Variable:
    for var in variables:
        if var.symbol == token:
            return var
    return None
