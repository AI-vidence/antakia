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


class Model(ABC):
    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        pass


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
    _model : Model to explain
    _explanation_method : SHAP or LIME
    """

    # Class attributes
    SHAP = 0
    LIME = 1

    def __init__(
        self,
        explanation_method: int,
        X: pd.DataFrame,
        model: Model = None
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

    def get_model(self) -> Model:
        return self._model

    @staticmethod
    def is_valid_explanation_method(method: int) -> bool:
        """
        Returns True if this is a valid explanation method.
        """
        return method == ExplanationMethod.SHAP or method == ExplanationMethod.LIME

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
    # TODO : do we need XAll ?
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

    VS = 0
    ES = 1  # Shoud not be allowed
    ES_SHAP = 3
    ES_LIME = 4

    PCA = 1
    TSNE = 2
    UMAP = 3
    PaCMAP = 4

    DIM_TWO = 2
    DIM_THREE = 3

    def __init__(
        self, base_space: int, dimreduc_method: int, dimension: int, X: pd.DataFrame
    ):
        """
        Constructor for the DimReducMethod class.

        Parameters
        ----------
        base_space : int
            Can be : VS, ES.SHAP or ES.LIME
            We store it here (not in implementation class)
        dimreduc_method : int
            Dimension reduction methods among DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP
            We store it here (not in implementation class)
        dimension : int
            Target dimension. Can be DIM_TWO or DIM_THREE
            We store it here (not in implementation class)
        X : pd.DataFrame
            Stored in LongTask instance
        """
        assert base_space is not DimReducMethod.ES
        self.base_space = base_space
        self._dimreduc_method = dimreduc_method
        self._dimension = dimension

        LongTask.__init__(self, LongTask.DIMENSIONALITY_REDUCTION, X)

    def get_base_space(self) -> int:
        return self.base_space

    @staticmethod
    def es_base_space(explain_method: int) -> int:
        if explain_method == ExplanationMethod.SHAP:
            return DimReducMethod.ES_SHAP
        elif explain_method == ExplanationMethod.LIME:
            return DimReducMethod.ES_LIME
        else:
            raise ValueError(explain_method, " is a bad explaination method")

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
    def base_space_as_str(base_space: int) -> str:
        if baseSpace == DimReducMethod.VS:
            return "VS"
        elif baseSpace == DimReducMethod.ES_SHAP:
            return "ES_SHAP"
        elif baseSpace == DimReducMethod.ES_LIME:
            return "ES_LIME"
        else:
            raise ValueError(base_space, " is a bad base space")

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
        return dim == DimReducMethod.DIM_TWO or dim == DimReducMethod.DIM_THREE


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
        _explained : bool
        _explain_method : int
        _contiuous : bool
        _lat : bool
        _lon : bool
    """

    def __init__(self, col_index:int, symbol: str, type: str):
        self._col_index = col_index
        self._symbol = symbol
        self._type = type

        self._descr = None
        self._sensible = False
        self._continuous = False
        self._explained = False
        self._explain_method = None
        self._lat = False
        self._lon = False

    def get_symbol(self) -> str:
        return self._symbol
    
    def get_col_index(self) -> int:
        return self._col_index  


class ProjectedValues():
    """
    An  class to hold a Dataframe X, and its projected values (with various dimensionality reduction methods).

    Instance attributes
    ------------------
    _X  : pandas.Dataframe
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
        self._X = X
        self._X_proj = {}


    def __str__(self):
        return f"ProjectedValues object with {self._X.shape[0]} obs and {self._X.shape[1]} variables"
    
    def get_length(self) -> int:
        return self._X.shape[0]

    def get_full_values(self) -> pd.DataFrame:
        """
        Returns X, non projected ("full") values.
        """
        return self._X

    def proj_values(self, dimreduc_method: int, dimension: int) -> pd.DataFrame:
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

class Dataset:
    """
    Stores X (with its projected values) and Y (target and precidted values).
    It also stores the list of Variables : initalized by Dataset but may be enriched by the user.
    
    Instance attributes :
    ---------------------
        The dataframe to be used by AntakIA
    _X_proj : a ProjectedValues object holing X and its projected values
    _y : a pd.Series with target values for y
    _y_pred : stores predicted values for y, computed at initialization time with the model
    _variables : a list of Variable objects
    """

    def __init__(self, X:pd.DataFrame, y= pd.series) -> None:
        self._X_proj = ProjectedValues(X)
        self._y = y
        self._y_pred = None
        


# =============================================================================


class ExplanationDataset:
    """
    ExplanationDataset class.

    An Explanations object holds the explanations values of the model.
    It is the "explained version" of a Dataset object.

    Instance Attributes
    --------------------
    _shapValues : dict
        - key IMPORTED : a pandas Dataframe containing the SHAP values provided by the user.
        - key COMPUTED : a pandas Dataframe containing the SHAP values computed by AntakIA.
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected SHAP values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected SHAP values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected SHAP values
        -  key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected SHAP values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected SHAP values
    _limeValues : dict
        - key IMPORTED : a pandas Dataframe containing the LIME values provided by the user.
        - key COMPUTED : a pandas Dataframe containing the LIME values computed by AntakIA.
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected LIME values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected LIME values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected LIME values
        -  key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected LIME values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected LIME values

    """

    # Class attributes
    # Characterizes an AntaKIA-computed Dataframe of explainations
    COMPUTED = -1
    # Characterizes an user-provided Dataframe of explainations
    IMPORTED = -2
    # We have both computed and imported values
    BOTH = -3
    # User doesn't care
    ANY = -4

    @staticmethod
    def is_valid_origin(originValue: int) -> bool:
        return originValue in [
            ExplanationDataset.COMPUTED,
            ExplanationDataset.IMPORTED,
            ExplanationDataset.ANY,
        ]

    def __init__(self, values: pd.DataFrame, explanation_method: int):
        """
        Constructor of the class ExplanationDataset.

        Parameters :
        ------------
        values : pandas.Dataframe
            The dataframe containing explanations values provided by the user (IMPORTED)
        explanation_method : int
            Must be ExplanationMethod.SHAP or ExplanationMethod.LIME
        """

        if explanation_method == ExplanationMethod.SHAP:
            self._shap_values = {self.IMPORTED: values}
            self._lime_values = {self.IMPORTED: None}
        elif explanation_method == ExplanationMethod.LIME:
            self._shap_values = {self.IMPORTED: None}
            self._lime_values = {self.IMPORTED: values}
        else:
            raise ValueError(
                "explanation_method must be ExplanationMethod.SHAP or ExplanationMethod.LIME"
            )

    @staticmethod
    def origin_by_str(origin_value: int) -> str:
        if origin_value == ExplanationDataset.IMPORTED:
            return "imported"
        elif origin_value == ExplanationDataset.COMPUTED:
            return "computed"
        elif origin_value == ExplanationDataset.BOTH:
            return "both"
        elif origin_value == ExplanationDataset.ANY:
            return "any"
        else:
            raise ValueError(origin_value, " is a bad origin")

    def get_full_values(self, explanation_method: int, origin: int = ANY):
        """Looks for imported or computed explaned values with no dimension reduction, hence "full values"
        Parameters:
        explanation_method = see ExplanationMethod integer constants
        origin = see ExplanationDataset integer constants, defaults to ANY ie. we return imported or computed
        Returns: a pandas dataframe or None
        """

        if not ExplanationMethod.is_valid_explanation_method(explanation_method):
            raise ValueError(
                explanation_method, " is a bad explanation method"
            )

        stored_explanations, df = None, None

        if explanation_method == ExplanationMethod.SHAP:
            stored_explanations = self._shap_values
        else:
            stored_explanations = self._lime_values

        if origin != ExplanationDataset.ANY:
            # User wants a specific origin
            # logger.debug(f"XDS.get_full_values : {ExplanationMethod.explanation_method_as_str(explanation_method)} values with {ExplanationDataset.getOriginByStr(origin)} origin required")
            if not ExplanationDataset.is_valid_origin(origin):
                raise ValueError(origin, " is a bad origin")
            if origin in stored_explanations:
                # logger.debug(f"XDS.get_full_values : found {ExplanationMethod.explanation_method_as_str(explanation_method)} values with {ExplanationDataset.getOriginByStr(origin)} origin ")
                df = stored_explanations[origin]
            else:
                # logger.debug(f"XDS.get_full_values : could not find {ExplanationMethod.explanation_method_as_str(explanation_method)} values with {ExplanationDataset.getOriginByStr(origin)} origin ")
                df = None
        else:
            # logger.debug(f"XDS.get_full_values : user wants {ExplanationMethod.explanation_method_as_str(explanation_method)} values whatever the origin")
            if ExplanationDataset.IMPORTED in stored_explanations:
                # logger.debug(f"XDS.get_full_values : found {ExplanationMethod.explanation_method_as_str(explanation_method)} values with imported origin ")
                df = stored_explanations[ExplanationDataset.IMPORTED]
            elif ExplanationDataset.COMPUTED in stored_explanations:
                # logger.debug(f"XDS.get_full_values : found {ExplanationMethod.explanation_method_as_str(explanation_method)} values with computed origin ")
                df = self._shap_values[ExplanationDataset.COMPUTED]
            else:
                # logger.debug(f"XDS.get_full_values : no {ExplanationMethod.explanation_method_as_str(explanation_method)} values")
                df = None

        return df

    def proj_values(
        self, explanation_method: int, dimreduc_method: int, dimension: int
    ) -> pd.DataFrame:
        """
        Looks for projecte values for a given  explanation method.
        We don't store projected values per origin : if explanations are computed for a given method (say SHAP), we we'll erase the previous projected values for an imported set/

        Parameters
        ----------
        explanation_method : int
            Should be ExplanationMethod.SHAP or ExplanationMethod.LIME
        dimreduc_method : int
            Should be DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP.
        dimension : int
            The dimension of the projection (2 or 3).

        Returns
        -------
        values : pandas.Dataframe
            The values for this explanation_method, dimreduc_method and dimension.
            Returns None if not computed yet.
        """

        if not ExplanationMethod.is_valid_explanation_method(explanation_method):
            raise ValueError(
                explanation_method, " is a bad explanation method "
            )
        if dimreduc_method is not None and not DimReducMethod.is_valid_dimreduc_method(
            dimreduc_method
        ):
            raise ValueError(
                dimreduc_method, " is a bad dimensionality reduction method"
            )
        if dimension is not None and not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        stored_proj_values, df = None, None

        if explanation_method == ExplanationMethod.SHAP:
            stored_proj_values = self._shap_values
        else:
            stored_proj_values = self._lime_values

        if dimreduc_method in stored_proj_values:
            if (
                stored_proj_values[dimreduc_method] is not None
                and dimension in stored_proj_values[dimreduc_method]
            ):
                df = stored_proj_values[dimreduc_method][dimension]

        return df

    def set_full_values(
        self, explanation_method: int, values: pd.DataFrame, origin: int
    ):
        """
        Use to set explained valued (not projected values)
        Parameters:
        explanation_method : see ExplanationMethod integer constants, can be SHAP or LIME
        values : a pandas dataframe
        origin : see ExplanationDataset integer constants, can be IMPORTED or COMPUTED
        """
        if not ExplanationMethod.is_valid_explanation_method(explanation_method):
            raise ValueError(
                explanation_method, " is a bad explanation method"
            )
        if not ExplanationDataset.is_valid_origin(origin):
            raise ValueError(origin, " is a bad origin")

        if explanation_method == ExplanationMethod.SHAP:
            our_store = self._shap_values
        else:
            our_store = self._lime_values

        if origin not in our_store:
            our_store[origin] = {}

        our_store[origin] = values

    def set_proj_values(
        self,
        explanation_method: int,
        dimreduc_method: int,
        dimension: int,
        values: pd.DataFrame,
    ):
        """Set values for this ExplanationDataset, given an explanation method, a dimensionality reduction and a dimension.

        Args:
            explanation_method : int
                SHAP or LIME (see ExplanationMethod class in compute.py)
            values : pd.DataFrame
                The values to set.
            dimreduc_method : int
                Method of dimension reduction. Can be None if values are not projected.
            dimension (int, optional): dimension of projection. Can be None if values are not projected.
        """
        if not ExplanationMethod.is_valid_explanation_method(explanation_method):
            raise ValueError(
                explanation_method, " is a bad explanation method"
            )
        if dimreduc_method is not None and not DimReducMethod.is_valid_dimreduc_method(
            dimreduc_method
        ):
            raise ValueError(
                dimreduc_method, " is a bad dimensionality reduction method"
            )
        if dimension is not None and not DimReducMethod.is_valid_dim_number(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        our_store = None

        if explanation_method == ExplanationMethod.SHAP:
            our_store = self._shap_values
        else:
            our_store = self._lime_values

        if dimreduc_method not in our_store:
            our_store[dimreduc_method] = {}
        if dimension not in our_store[dimreduc_method]:
            our_store[dimreduc_method][dimension] = {}

        our_store[dimreduc_method][dimension] = values

    def is_explanation_available(self, explanation_method: int, origin: int) -> bool:
        """Tells wether we have this explanation method values or not for this origin"""
        if not ExplanationMethod.is_valid_explanation_method(explanation_method):
            raise ValueError(
                explanation_method, " is a bad explaination method"
            )
        if not ExplanationDataset.is_valid_origin(origin):
            raise ValueError(origin, " is a bad origin")

        storedValues = None
        if explanation_method == ExplanationMethod.SHAP:
            storedValues = self._shap_values
        else:
            storedValues = self._lime_values

        if origin == ExplanationDataset.ANY:
            return self.is_explanation_available(
                ExplanationDataset.IMPORTED
            ) or self.is_explanation_available(ExplanationDataset.COMPUTED)
        else:
            if origin in storedValues and storedValues[origin] is not None:
                # logger.debug(f"is_explanation_available : yes we have {ExplanationMethod.explanation_method_as_str(explanation_method)}")
                return True

    def _desc_proj_helper(
        self, text: str, explanation_method: int, dimreduc_method: int
    ) -> str:
        if explanation_method == ExplanationMethod.SHAP:
            explain_dict = self._shap_values
        elif explanation_method == ExplanationMethod.LIME:
            explain_dict = self._lime_values
        else:
            raise ValueError(
                explanation_method, " is a bad explaination method method"
            )

        if not DimReducMethod.is_valid_dimreduc_method(dimreduc_method):
            raise ValueError(
                dimreduc_method, " is a bad dimensionality reduction method"
            )

        if (
            dimreduc_method in explain_dict
            and explain_dict[dimreduc_method] is not None
        ):
            text += (
                ExplanationMethod.explain_method_as_str(explanation_method)
                + " values dict has "
                + DimReducMethod.dimreducmethod_as_str(dimreduc_method)
                + " projected values :\n"
            )
            if (
                DimReducMethod.DIM_TWO in explain_dict[dimreduc_method]
                and explain_dict[dimreduc_method][DimReducMethod.DIM_TWO] is not None
            ):
                text += (
                    "    2D :"
                    + str(
                        explain_dict[dimreduc_method][DimReducMethod.DIM_TWO].shape[0]
                    )
                    + " observations, "
                    + str(
                        self._shap_values[dimreduc_method][
                            DimReducMethod.DIM_TWO
                        ].shape[1]
                    )
                    + " features\n"
                )
            if (
                DimReducMethod.DIM_THREE in explain_dict[dimreduc_method]
                and explain_dict[dimreduc_method][DimReducMethod.DIM_THREE]
                is not None
            ):
                text += (
                    "    3D :"
                    + str(
                        explain_dict[dimreduc_method][DimReducMethod.DIM_THREE].shape[
                            0
                        ]
                    )
                    + " observations, "
                    + str(
                        explain_dict[dimreduc_method][DimReducMethod.DIM_THREE].shape[
                            1
                        ]
                    )
                    + " features\n"
                )

        return text

    def _desc_explain_helper(self, text: str, explanation_method: int) -> str:
        if explanation_method == ExplanationMethod.SHAP:
            explain_dict = self._shap_values
        elif explanation_method == ExplanationMethod.LIME:
            explain_dict = self._lime_values
        else:
            raise ValueError(
                explanation_method, " is a bad explaination method"
            )

        if explain_dict is None or len(explain_dict) == 0:
            text += (
                ExplanationMethod.explain_method_as_str(explanation_method)
                + " values dict is empty\n"
            )
        else:
            if (
                ExplanationDataset.IMPORTED in explain_dict
                and explain_dict[ExplanationDataset.IMPORTED] is not None
            ):
                text += (
                    ExplanationMethod.explain_method_as_str(explanation_method)
                    + " values dict has imported values :\n"
                )
                text += (
                    "    "
                    + str(explain_dict[ExplanationDataset.IMPORTED].shape[0])
                    + " observations, "
                    + str(explain_dict[ExplanationDataset.IMPORTED].shape[1])
                    + " features\n"
                )

            if (
                ExplanationDataset.COMPUTED in explain_dict
                and explain_dict[ExplanationDataset.COMPUTED] is not None
            ):
                text += (
                    ExplanationMethod.explain_method_as_str(explanation_method)
                    + " values dict has computed values :\n"
                )
                text += (
                    "    "
                    + str(explain_dict[ExplanationDataset.COMPUTED].shape[0])
                    + " observations, "
                    + +str(explain_dict[ExplanationDataset.COMPUTED].shape[1])
                    + " features\n"
                )

        for proj_method in DimReducMethod.getDimReducMethodsAsList():
            text += self._desc_proj_helper(text, explanation_method, proj_method)

        return text

    def __repr__(self) -> str:
        text = "ExplanationDataset object :\n"
        text += "---------------------------\n"

        for explain_method in ExplanationMethod.explanation_methods_as_list():
            text += self._desc_explain_helper(text, explain_method)
        return text
