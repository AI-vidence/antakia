import logging
from abc import ABC, abstractmethod

import pandas as pd

# TODO : these references to IPython should be removed in favor of a new scheme (see Wiki)
from sklearn.preprocessing import StandardScaler

from antakia.utils import confLogger

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()

def is_valid_model(model) -> bool:
    return callable(getattr(model, "score")) and callable(getattr(model, "predict"))

class LongTask(ABC):
    """
    Abstract class to compute long tasks, often in a separate thread.

    Attributes
    ----------
    _longTaskType : int
        Can be LongTask.EXPLAINATION or LongTask.DIMENSIONALITY_REDUCTION
    _X : dataframe
    _progress : int
        The progress of the long task, between 0 and 100.
    """

    # Class attributes : LongTask types
    EXPLANATION = 1
    DIMENSIONALITY_REDUCTION = 2

    def __init__(self, type: int, X: pd.DataFrame):
        if not LongTask.is_valid_longtask_type(type):
            raise ValueError(type, " is a bad long task type")
        if X is None:
            raise ValueError("You must provide a dataframe for a LongTask")
        self._longtask_type = type
        self._X = X
        self._progress = 0

    def get_X(self) -> pd.DataFrame:
        return self._X

    def get_progress(self) -> int:
        logger.debug(f"LongTask.getProgress : returning {self._progress}")
        return self._progress

    def set_progress(self, progress: int):
        """
        Method to send the progress of the long task.

        Parameters
        ----------
        progress : int
            An integer between 0 and 100.
        """
        self._progress

    @abstractmethod
    def get_topic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        pass

    def __call__(self):
        logger.debug("LongTask.__call__ : I'm going to upadte the subscriber")
        self.set_progress(self._progress)

    @staticmethod
    def is_valid_longtask_type(type: int) -> bool:
        """
        Returns True if the type is a valid LongTask type.
        """
        return type == LongTask.EXPLANATION or type == LongTask.DIMENSIONALITY_REDUCTION

    def get_longtask_type(self) -> int:
        return self._longtask_type

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """
        Method to compute the long task and update listener with the progress.
        """
        pass


class ExplanationMethod(LongTask):
    """
    Abstract class (see Long Task) to compute explaination values for the Explanation Space (ES)

    Attributes
    _model : the model to explain
    _explanation_method : SHAP or LIME
    """

    # Class attributes
    NONE = 0 # no explanation, ie: original values
    SHAP = 1
    LIME = 2

    def __init__(
        self,
        explanation_method: int,
        X: pd.DataFrame,
        model= None
    ):
        # TODO : do wee need X_all ?
        super().__init__(LongTask.EXPLANATION, X)
        self._model = model
        self._explanation_method = explanation_method

    # Overloading the setProgress
    def set_progress(self, progress: int):
        """
        Method to send the progress of the long task.

        Parameters
        ----------
        progress : int
            An integer between 0 and 100.
        """
        self._progress = progress

    def get_model(self):
        return self._model

    @staticmethod
    def is_valid_explanation_method(method: int) -> bool:
        """
        Returns True if this is a valid explanation method.
        """
        return method == ExplanationMethod.SHAP or method == ExplanationMethod.LIME or method == ExplanationMethod.NONE 
    

    def get_topic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        return ExplanationMethod.explain_method_as_str(self.explanation_method)

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

class DimReducMethod(LongTask):
    """
    Class that allows to reduce the dimensionality of the data.

    Attributes
    ----------
    _dimreduc_method : int, can be PCA, TSNE etc.
    _dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class
    _base_space : int
    """
    # Class attributes methods
    PCA = 1
    TSNE = 2
    UMAP = 3
    PaCMAP = 4

    def __init__(
        self, dimreduc_method: int, dimension: int, X: pd.DataFrame
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
        """
        self._dimreduc_method = dimreduc_method
        self._dimension = dimension

        LongTask.__init__(self, LongTask.DIMENSIONALITY_REDUCTION, X)

    def get_dimreduc_method(self) -> int:
        return self._dimreduc_method

    def get_dimension(self) -> int:
        return self._dimension

    def get_topic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        # Should look like "VS/PCA/2D" or "ES.SHAP/t-SNE/3D"
        return (
            DimReducMethod.base_space_as_str(self.get_base_space())
            + "/"
            + DimReducMethod.dimreducmethod_as_str(self.get_dimreduc_method())
            + "/"
            + DimReducMethod.getDimensionAsStr(self.get_dimension())
        )

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
        if dim == DimReducMethod.DIM_TWO:
            return "2D"
        elif dim == DimReducMethod.DIM_THREE:
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


# ================================================

class Variable:
    """ 
        Describes each X column or Y value

        _col_index : int
            The index of the column in the dataframe i come from (ds or xds)
            #TODO : I shoudl code an Abstract class for Dataset and ExplanationDataset
        _symbol : str
            How it should be displayed in the GUI
        _descr : str
            A description of the variable
        _type : str
            The type of the variable
        _sensible : bool
            Wether the variable is sensible or not
        _contiuous : bool
        _lat : bool
        _lon : bool
    """

    def __init__(self, col_index:int, symbol: str, type: str):
        self.col_index = col_index
        self.symbol = symbol
        self.type = type

        self.descr = None
        self.sensible = False
        self.continuous = False
        self.explained = False
        self.explain_method = None
        self.lat = False
        self.lon = False
        
    @staticmethod
    def guess_variables(X: pd.DataFrame) -> list:
        """
        Returns a list of Variable objects, one for each column in X.
        """
        variables = []
        for i in range(len(X.columns)):
            var = Variable(i, X.columns[i], X.dtypes[X.columns[i]])
            if X.columns[i] in ["latitude", "Latitude", "Lat", "lat"]:
                var._lat = True
            if X.columns[i] in ["longitude", "Longitude", "Long", "long"]:
                var._lon = True
            variables.append(var)
        return variables

class ProjectedValues():
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
        return f"ProjectedValues object with {self.X.shape[0]} obs and {self.X.shape[1]} variables"
    
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
            raise ValueError("Bad dimension number")

        df = None

        if dimreduc_method not in self._X_proj:
            df = None
        elif dimension not in self._X_proj[dimreduc_method]:
            df = None
        else:
            df = self._X_proj[dimreduc_method][dimension]
        return df
    
    def is_available(self, dimreduc_method: int, dimension: int)->bool :
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

        logger.debug(
            f"DS.setProjValues : I set new values for {DimReducMethod.dimreduc_method_as_str(dimreduc_method)} in {dimension}D proj"
        )
        self._X_proj[dimreduc_method][dimension] = values

    def get_shape(self):
        """Returns the shape of the used dataset"""
        return self._X.shape