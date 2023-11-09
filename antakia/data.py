import logging
from abc import ABC, abstractmethod

import pandas as pd

import time

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

    def __init__(self, X: pd.DataFrame, progress_updated: callable = None):
        if X is None:
            raise ValueError("You must provide a dataframe for a LongTask")
        self.X = X
        self.progress_updated = progress_updated
        self.start_time = time.time()

    def publish_progress(self, progress: int):
        self.progress_updated(self, progress, time.time() - self.start_time)

    @abstractmethod
    def compute(self) -> pd.DataFrame:
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
    PCA = 1
    TSNE = 2
    UMAP = 3
    PaCMAP = 4

    def __init__(
        self,
        dimreduc_method: int,
        dimension: int,
        X: pd.DataFrame,
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
        self.dimension = dimension
        # IMPORTANT : we set the topic as for ex 'PCA/2' or 't-SNE/3' -> subscribers have to follow this scheme
        LongTask.__init__(self, X, progress_updated)

    @staticmethod
    def dimreduc_method_as_str(method: int) -> str:
        if method == DimReducMethod.PCA:
            return "PCA"
        elif method == DimReducMethod.TSNE:
            return "t-SNE"
        elif method == DimReducMethod.UMAP:
            return "UMAP"
        elif method == DimReducMethod.PaCMAP:
            return "PaCMAP"
        elif method is None:
            return None
        else:
            raise ValueError(method, " is an invalid dimensionality reduction method")

    @staticmethod
    def dimreduc_method_as_int(method: str) -> int:
        if method == "PCA":
            return DimReducMethod.PCA
        elif method == "t-SNE":
            return DimReducMethod.TSNE
        elif method == "UMAP":
            return DimReducMethod.UMAP
        elif method == "PaCMAP":
            return DimReducMethod.PaCMAP
        elif method is None:
            return None
        else:
            raise ValueError(method, " is an invalid dimensionality reduction method")

    @staticmethod
    def dimreduc_methods_as_list() -> list:
        return [
            DimReducMethod.PCA,
            DimReducMethod.TSNE,
            DimReducMethod.UMAP,
            DimReducMethod.PaCMAP,
        ]

    @staticmethod
    def dimreduc_methods_as_str_list() -> list:
        return ["PCA", "t-SNE", "UMAP", "PaCMAP"]

    @staticmethod
    def dimension_as_str(dim) -> str:
        if dim == 2:
            return "2D"
        elif dim == 3:
            return "3D"
        else:
            raise ValueError(dim, " is a bad dimension")

    @staticmethod
    def is_valid_dimreduc_method(method: int) -> bool:
        """
        Returns True if it is a valid dimensionality reduction method.
        """
        return (
            method == DimReducMethod.PCA
            or method == DimReducMethod.TSNE
            or method == DimReducMethod.UMAP
            or method == DimReducMethod.PaCMAP
        )

    @staticmethod
    def is_valid_dim_number(dim: int) -> bool:
        """
        Returns True if dim is a valid dimension number.
        """
        return dim == 2 or dim == 3

    def get_dimension(self) -> int:
        return self.dimension


# ================================================


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


def vars_to_string(variables: list) -> str:
    text = ""
    for i in range(len(variables)):
        var = variables[i]
        text += str(i) + ") " + var.__str__() + "\n"
    return text


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


class ProjectedValues:
    """
    A class to hold a Dataframe X, and its projected values (with various dimensionality reduction methods).

    Instance attributes
    ------------------
    X  : pandas.Dataframe
        The dataframe to be used by AntakIA
    _X_proj : a dict with four values :
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected X values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected X values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected X values
        - key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected X values
    """

    def __init__(self, X: pd.DataFrame):
        """
        Constructor of the class ProjectedValues.

        Parameters
        ----------
        X : pandas.Dataframe
            The dataframe provided by the user.
        """
        self.X = X
        self._X_proj = {}

    def __str__(self):
        text = f"X's shape : {self.X.shape}"
        for proj_method in DimReducMethod.dimreduc_methods_as_list():
            for dim in [2, 3]:
                if self.is_available(proj_method, dim):
                    text += f"\n{DimReducMethod.dimreduc_method_as_str(proj_method)}/{DimReducMethod.dimension_as_str(dim)} : {self.get_proj_values(proj_method, dim).shape}"
        return text

    def get_length(self) -> int:
        return self.X.shape[0]

    def get_proj_values(self, dimreduc_method: int, dimension: int) -> pd.DataFrame:
        """Returns de projected X values using a dimensionality reduction method and target dimension (2 or 3)

        Args:
            dimreduc_method (int): the dimensionality reduction method
            dimension (int, optional): Defaults to DimReducMethod.DIM_TWO.

        Returns:
            pd.DataFrame: the projected X values. May be None
        """
        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            raise ValueError("Bad dimensionality reduction method")
        if not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        if (
            dimreduc_method not in self._X_proj
            or dimension not in self._X_proj[dimreduc_method]
            or self._X_proj[dimreduc_method][dimension] is None
        ):
            # ProjectedValues is a "datastore", its role is not trigger projection computations
            return None
        else:
            return self._X_proj[dimreduc_method][dimension]

    def is_available(self, dimreduc_method: int, dimension: int) -> bool:
        return self.get_proj_values(dimreduc_method, dimension) is not None

    def set_proj_values(
        self, dimreduc_method: int, dimension: int, values: pd.DataFrame
    ):
        """Set X_proj alues for this dimensionality reduction and  dimension."""

        # TODO we may want to check values.shape and raise value error if it does not match
        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            raise ValueError("Bad dimensionality reduction method")
        if not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError("Bad dimension number")

        if dimreduc_method not in self._X_proj:
            self._X_proj[
                dimreduc_method
            ] = {}  # We create a new dict for this dimreduc_method
        if dimension not in self._X_proj[dimreduc_method]:
            self._X_proj[dimreduc_method][
                dimension
            ] = {}  # We create a new dict for this dimension

        self._X_proj[dimreduc_method][dimension] = values

    def get_shape(self):
        """Returns the shape of the used dataset"""
        return self._X.shape
