import logging
from abc import ABC, abstractmethod
import numpy as np

import pandas as pd

import time

from sklearn.base import TransformerMixin
# TODO : these references to IPython should be removed in favor of a new scheme (see Wiki)
from sklearn.preprocessing import StandardScaler
import logging as logging
from antakia.utils import conf_logger

logger = logging.getLogger(__name__)
conf_logger(logger)


def is_valid_model(model) -> bool:
    return callable(getattr(model, "score")) and callable(getattr(model, "predict"))


class LongTask(ABC):
    """
    Abstract class to compute long tasks, often in a separate thread.

    Attributes
    ----------
    X : dataframe
    progress_updated : an optional callback function to call when progress is updated
    start_time : float
    progress:int
    """

    def __init__(self, X: pd.DataFrame=None, progress_updated: callable = None):
        if X is None:
            raise ValueError("You must provide a dataframe for a LongTask")
        self.X = X
        self.progress_updated = progress_updated
        self.start_time = time.time()

    def publish_progress(self, progress: int):
        self.progress_updated(self, progress, time.time() - self.start_time)

    @abstractmethod
    def compute(self, **kwargs) -> pd.DataFrame:
        """
        Method to compute the long task and update listener with the progress.
        """
        pass


# -----


class ExplanationMethod(LongTask):
    """
    Abstract class (see Long Task) to compute explaination values for the Explanation Space (ES)

    Attributes
    model : the model to explain
    explanation_method : SHAP or LIME
    """

    # Class attributes
    NONE = 0  # no explanation, ie: original values
    SHAP = 1
    LIME = 2

    def __init__(
            self,
            explanation_method: int,
            X: pd.DataFrame,
            model=None,
            progress_updated: callable = None,
    ):
        # TODO : do wee need X_all ?
        if not ExplanationMethod.is_valid_explanation_method(explanation_method):
            raise ValueError(explanation_method, " is a bad explanation method")
        self.explanation_method = explanation_method
        super().__init__(X, progress_updated)
        self.model = model

    @staticmethod
    def is_valid_explanation_method(method: int) -> bool:
        """
        Returns True if this is a valid explanation method.
        """
        return (
                method == ExplanationMethod.SHAP
                or method == ExplanationMethod.LIME
                or method == ExplanationMethod.NONE
        )

    @staticmethod
    def explanation_methods_as_list() -> list:
        return [ExplanationMethod.SHAP, ExplanationMethod.LIME]

    @staticmethod
    def explain_method_as_str(method: int) -> str:
        if method == ExplanationMethod.SHAP:
            return "SHAP"
        elif method == ExplanationMethod.LIME:
            return "LIME"
        else:
            raise ValueError(method, " is a bad explanation method")

    @staticmethod
    def explain_method_as_int(method: str) -> int:
        if method.upper() == "SHAP":
            return ExplanationMethod.SHAP
        elif method.upper() == "LIME":
            return ExplanationMethod.LIME
        else:
            raise ValueError(method, " is a bad explanation method")


# -----


class DimReducMethod(LongTask):
    """
    Class that allows to reduce the dimensionality of the data.

    Attributes
    ----------
    dimreduc_method : int, can be PCA, TSNE etc.
    dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class
    """

    # Class attributes methods
    dim_reduc_methods = ['PCA', 'TSNE', 'UMAP', 'PaCMAP']
    dimreduc_method = -1

    allowed_kwargs = []

    def __init__(
            self,
            dimreduc_method: int,
            dimreduc_model: type[TransformerMixin],
            dimension: int,
            X: pd.DataFrame,
            default_parameters: dict = None,
            progress_updated: callable = None,
    ):
        """
        Constructor for the DimReducMethod class.

        Parameters
        ----------
        dimreduc_method : int
            Dimension reduction methods among DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP
            We store it here (not in implementation class)
        dimension : int
            Target dimension. Can be DIM_TWO or DIM_THREE
            We store it here (not in implementation class)
        X : pd.DataFrame
            Stored in LongTask instance
        progress_updated : callable
            Stored in LongTask instance
        """
        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            raise ValueError(
                dimreduc_method, " is a Bbad dimensionality reduction method"
            )
        if not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        self.dimreduc_method = dimreduc_method
        self.default_parameters = default_parameters
        self.dimension = dimension
        self.dimreduc_model = dimreduc_model
        # IMPORTANT : we set the topic as for ex 'PCA/2' or 't-SNE/3' -> subscribers have to follow this scheme
        LongTask.__init__(self, X, progress_updated)

    @classmethod
    def dimreduc_method_as_str(cls, method: int) -> str:
        if method is None:
            return None
        elif 0 < method <= len(cls.dim_reduc_methods):
            return cls.dim_reduc_methods[method - 1]
        else:
            raise ValueError(f"{method} is an invalid dimensionality reduction method")

    @classmethod
    def dimreduc_method_as_int(cls, method: str) -> int:
        if method is None:
            return
        try:
            i = cls.dim_reduc_methods.index(method) + 1
            return i
        except ValueError:
            raise ValueError(f"{method} is an invalid dimensionality reduction method")

    @classmethod
    def dimreduc_methods_as_list(cls) -> list:
        return list(map(lambda x: x + 1, range(len(cls.dim_reduc_methods))))

    @classmethod
    def dimreduc_methods_as_str_list(cls) -> list:
        return cls.dim_reduc_methods.copy()

    @staticmethod
    def dimension_as_str(dim) -> str:
        if dim == 2:
            return "2D"
        elif dim == 3:
            return "3D"
        else:
            raise ValueError(f"{dim}, is a bad dimension")

    @classmethod
    def is_valid_dimreduc_method(cls, method: int) -> bool:
        """
        Returns True if it is a valid dimensionality reduction method.
        """
        return 0 <= method - 1 < len(cls.dim_reduc_methods)

    @staticmethod
    def is_valid_dim_number(dim: int) -> bool:
        """
        Returns True if dim is a valid dimension number.
        """
        return dim in [2, 3]

    def get_dimension(self) -> int:
        return self.dimension

    @classmethod
    def parameters(cls):
        return {}

    def compute(self, **kwargs) -> pd.DataFrame:
        self.publish_progress(0)
        kwargs['n_components'] = self.get_dimension()
        param = self.default_parameters.copy()
        param.update(kwargs)

        dim_red_model = self.dimreduc_model(**param)
        if hasattr(dim_red_model, 'fit_transform'):
            X_red = dim_red_model.fit_transform(self.X)
        else:
            dim_red_model.fit(self.X)
            X_red = dim_red_model.transform(self.X)
        X_red = pd.DataFrame(X_red)

        self.publish_progress(100)
        return X_red
    
    @classmethod
    def scale_value_space(cls, X: pd.DataFrame, y:pd.Series) -> pd.DataFrame:
        """
        Scale the values in X to be between 0 and 1.
        """
        return  (X-X.mean())/X.std()*np.abs(X.corrwith(y))


# ---------------------------------------------

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
            critical=False,
            continuous: bool = False,
            lat: bool = False,
            lon: bool = False,
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
    def guess_variables(X: pd.DataFrame) -> list:
        """
        Returns a list of Variable objects, one for each column in X.
        """
        variables = []
        for i in range(len(X.columns)):
            X.columns[i].replace("_", " ")
            var = Variable(i, X.columns[i], X.dtypes[X.columns[i]])
            if X.columns[i] in ["latitude", "Latitude", "Lat", "lat"]:
                var.lat = True
            if X.columns[i] in ["longitude", "Longitude", "Long", "long"]:
                var.lon = True
            var.continuous = Variable.is_continuous(X[X.columns[i]])
            variables.append(var)
        return variables

    @staticmethod
    def import_variable_df(df: pd.DataFrame) -> list:
        """
        Import variables from a DataFrame
        """
        # We just need to insert a new column with symbols (ie indexes)
        df.insert(loc=0, column="symbol", value=df.index)
        # and send it as list of dicts to import_variable_list
        return Variable.import_variable_list(df.to_dict("records"))

    @staticmethod
    def import_variable_list(var_list: list) -> list:
        """
        Import variables from a list of dicts
        """
        variables = []
        for i in range(len(var_list)):
            if isinstance(var_list[i], dict):
                item = var_list[i]
                if "col_index" in item and "symbol" in item and "type" in item:
                    var = Variable(item["col_index"], item["symbol"], item["type"])
                    if "unit" in item:
                        var.unit = item["unit"]
                    if "descr" in item:
                        var.descr = item["descr"]
                    if "critical" in item:
                        var.critical = item["critical"]
                    if "continuous" in item:
                        var.continuous = item["continuous"]
                    if "lat" in item:
                        var.lat = item["lat"]
                    if "lon" in item:
                        var.lon = item["lon"]
                    variables.append(var)
                else:
                    raise ValueError(
                        "Variable must a list of {key:value} with 'must keys' in [col_index, symbol, type] and optional keys in [unit, descr, critical, continuous, lat, lon]"
                    )
        return variables

    @staticmethod
    def is_continuous(serie: pd.Series) -> bool:
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

    @staticmethod
    def vars_to_string(variables: list) -> str:
        text = ""
        for i in range(len(variables)):
            var = variables[i]
            text += str(i) + ") " + var.__str__() + "\n"
        return text

    @staticmethod
    def vars_to_sym_list(variables: list) -> list:
        symbols = []
        for var in variables:
            symbols.append(var.symbol)
        return symbols


def var_from_symbol(variables: list, token: str) -> Variable:
    for var in variables:
        if var.symbol == token:
            return var
    return None
