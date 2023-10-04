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

from antakia.utils import simpleType


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

    def __init__(self, longTaskType: int, X: pd.DataFrame):
        if not LongTask.isValidLongTaskType(longTaskType):
            raise ValueError(longTaskType, " is a bad long task type")
        if X is None:
            raise ValueError("You must provide a dataframe for a LongTask")
        self._longTaskType = longTaskType
        self._X = X
        self._progress = 0

    def getX(self) -> pd.DataFrame:
        return self._X

    def getProgress(self) -> int:
        logger.debug(f"LongTask.getProgress : returning {self._progress}")
        return self._progress

    def setProgress(self, progress: int):
        """
        Method to send the progress of the long task.

        Parameters
        ----------
        progress : int
            An integer between 0 and 100.
        """
        self._progress

    @abstractmethod
    def getTopic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        pass

    def __call__(self):
        logger.debug("LongTask.__call__ : I'm going to upadte the subscriber")
        self.setProgress(self._progress)

    @staticmethod
    def isValidLongTaskType(type: int) -> bool:
        """
        Returns True if the type is a valid LongTask type.
        """
        return type == LongTask.EXPLANATION or type == LongTask.DIMENSIONALITY_REDUCTION

    def getLongTaskType(self) -> int:
        return self._longTaskType

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
    _explainationType : SHAP or LIME
    """

    # Class attributes types
    SHAP = 0
    LIME = 1

    def __init__(
        self,
        explainationType: int,
        X: pd.DataFrame,
        model: Model = None,
        userComputedValuesForProjection: bool = True,
    ):
        # TODO : do wee need X_all ?
        super().__init__(LongTask.EXPLANATION, X)
        self._model = model
        self._explainationType = explainationType

    # Overloading the setProgress
    def setProgress(self, progress: int):
        """
        Method to send the progress of the long task.

        Parameters
        ----------
        progress : int
            An integer between 0 and 100.
        """
        self._progress = progress

    def getModel(self) -> Model:
        return self._model

    @staticmethod
    def isValidExplanationType(type: int) -> bool:
        """
        Returns True if the type is a valid explanation type.
        """
        return type == ExplanationMethod.SHAP or type == ExplanationMethod.LIME

    def getTopic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        return ExplanationMethod.getExplanationMethodAsStr(self._explainationType)

    @staticmethod
    def getExplanationMethodsAsList() -> list:
        return [ExplanationMethod.SHAP, ExplanationMethod.LIME]

    @staticmethod
    def getExplanationMethodAsStr(type: int) -> str:
        if type == ExplanationMethod.SHAP:
            return "SHAP"
        elif type == ExplanationMethod.LIME:
            return "LIME"
        else:
            raise ValueError(type, " is a bad explaination type")


class DimReducMethod(LongTask):
    # TODO : do we need XAll ?
    """
    Class that allows to reduce the dimensionality of the data.

    Attributes
    ----------
    _dimReducType : int, can be PCA, TSNE etc.
    _dimension : int
        Dimension reduction methods require a dimension parameter
        We store it in the abstract class
    """
    # Class attributes types

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
        self, baseSpace: int, dimReducType: int, dimension: int, X: pd.DataFrame
    ):
        """
        Constructor for the DimReducMethod class.

        Parameters
        ----------
        baseSpace : int
            Can be : VS, ES.SHAP or ES.LIME
            We store it here (not in implementation class)
        dimeReducType : int
            Dimension reduction methods among DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP
            We store it here (not in implementation class)
        dimension : int
            Target dimension. Can be DIM_TWO or DIM_THREE
            We store it here (not in implementation class)
        X : pd.DataFrame
            Stored in LongTask instance
        """
        assert baseSpace is not DimReducMethod.ES
        self._baseSpace = baseSpace
        self._dimReducType = dimReducType
        self._dimension = dimension

        LongTask.__init__(self, LongTask.DIMENSIONALITY_REDUCTION, X)

    def getBaseSpace(self) -> int:
        return self._baseSpace

    @staticmethod
    def getESBaseSpace(explainMethod: int) -> int:
        if explainMethod == ExplanationMethod.SHAP:
            return DimReducMethod.ES_SHAP
        elif explainMethod == ExplanationMethod.LIME:
            return DimReducMethod.ES_LIME
        else:
            raise ValueError(explainMethod, " is a bad explaination method")

    def getDimReducType(self) -> int:
        return self._dimReducType

    def getDimension(self) -> int:
        return self._dimension

    def getTopic(self) -> str:
        """Required by pubssub.
        Identifier for the long task being computed.

        Returns:
            str: a Topic for pubsub
        """
        # Should look like "VS/PCA/2D" or "ES.SHAP/t-SNE/3D"
        return (
            DimReducMethod.getBaseSpaceAsStr(self.getBaseSpace())
            + "/"
            + DimReducMethod.getDimReducMethodAsStr(self.getDimReducType())
            + "/"
            + DimReducMethod.getDimensionAsStr(self.getDimension())
        )

    @staticmethod
    def getBaseSpaceAsStr(baseSpace: int) -> str:
        if baseSpace == DimReducMethod.VS:
            return "VS"
        elif baseSpace == DimReducMethod.ES_SHAP:
            return "ES_SHAP"
        elif baseSpace == DimReducMethod.ES_LIME:
            return "ES_LIME"
        else:
            raise ValueError(baseSpace, " is a bad base space")

    @staticmethod
    def getDimReducMethodAsStr(type: int) -> str:
        if type == DimReducMethod.PCA:
            return "PCA"
        elif type == DimReducMethod.TSNE:
            return "t-SNE"
        elif type == DimReducMethod.UMAP:
            return "UMAP"
        elif type == DimReducMethod.PaCMAP:
            return "PaCMAP"
        elif type is None:
            return None
        else:
            raise ValueError(type, " is an invalid dimensionality reduction method")

    @staticmethod
    def getDimReducMethodAsInt(type: str) -> int:
        if type == "PCA":
            return DimReducMethod.PCA
        elif type == "t-SNE":
            return DimReducMethod.TSNE
        elif type == "UMAP":
            return DimReducMethod.UMAP
        elif type == "PaCMAP":
            return DimReducMethod.PaCMAP
        elif type is None:
            return None
        else:
            raise ValueError(type, " is an invalid dimensionality reduction method")

    @staticmethod
    def getDimReducMethodsAsList() -> list:
        return [
            DimReducMethod.PCA,
            DimReducMethod.TSNE,
            DimReducMethod.UMAP,
            DimReducMethod.PaCMAP,
        ]

    @staticmethod
    def getDimReducMethodsAsStrList() -> list:
        return ["PCA", "t-SNE", "UMAP", "PaCMAP"]

    @staticmethod
    def getDimensionAsStr(dim) -> str:
        if dim == DimReducMethod.DIM_TWO:
            return "2D"
        elif dim == DimReducMethod.DIM_THREE:
            return "3D"
        else:
            raise ValueError(dim, " is a bad dimension")

    @staticmethod
    def isValidDimReducType(type: int) -> bool:
        """
        Returns True if the type is a valid dimensionality reduction type.
        """
        return (
            type == DimReducMethod.PCA
            or type == DimReducMethod.TSNE
            or type == DimReducMethod.UMAP
            or type == DimReducMethod.PaCMAP
        )

    @staticmethod
    def isValidDimNumber(type: int) -> bool:
        """
        Returns True if the type is a valid dimension number.
        """
        return type == DimReducMethod.DIM_TWO or type == DimReducMethod.DIM_THREE


# ================================================

class Variable:
    """ 
        Describes each X column or Y value

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

    def __init__(self, symbol: str, type: str):
        self._symbol = symbol
        self._type = type

        self._descr = None
        self._sensible = False
        self._continuous = False
        self._explained = False
        self._explain_method = None
        self._lat = False
        self._lon = False

    def getSymbol(self) -> str:
        return self._symbol


class Dataset:
    """
    Dataset class.

    Instance attributes
    ------------------
    _X  : pandas.Dataframe
        The dataframe to be used by AntakIA
    _X_proj : dict :
        - key DimReducMethod.PCA : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PCA-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PCA-projected X values
        - key DimReducMethod.TSNE : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D TSNE-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D TSNE-projected X values
        - key DimReducMethod.UMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D UMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D UMAP-projected X values
        -  key DimReducMethod.PaCMAP : a dict with :
            - key DimReducMethod.DIM_TWO : a pandas Dataframe with 2D PaCMAP-projected X values
            - key DimReducMethod.DIM_THREE :  a pandas Dataframe with 3D PaCMAP-projected X values
    _X_all: pandas.Dataframe
        The entire dataframe. X may be smaller than Xall if the frac method has been used.
    _X_scaled : pandas.Dataframe
        The dataframe with normalized (scaled) values.
    _variables : list of Variable
        Describes each column of the dataset
    _y : pandas.Series
        Target values
    _y_pred : pandas.Series
        The Serie containing the predictions of the model. Computed at construction time.
    """

    # Class attributes for X values
    REGULAR = 1
    SCALED = 2

    # Class attributes for Y values
    TARGET = 0
    PREDICTED = 1

    def __init__(self, X: pd.DataFrame, csv: str = None, y: pd.Series = None):
        """
        Constructor of the class Dataset.

        Parameters
        ----------
        X : pandas.Dataframe
            The dataframe containing the dataset provided by the user.
        csv : str (optional)
            The path to the csv file containing the dataset.
            #TODO : we shoudl use a Path or File object
        y : pandas.series (optional)
            The series containing the target values.
            TODO : is it compulsory ?
        """

        if X is None and csv is None:
            raise ValueError("You must provide a dataframe or a CSV file")
        if X is not None and csv is not None:
            raise ValueError(
                "You must provide either a dataframe or a CSV file, not both"
            )
        if X is not None:
            self._X = X
        else:
            self._X = pd.read_csv(csv)

        self._X_proj = {}  # Empty dict
        self._explanations = None

        # We remove spaces in the column names
        self._X.columns = [
            self._X.columns[i].replace(" ", "_") for i in range(len(self._X.columns))
        ]
        self._X = self._X.reset_index(drop=True)

        self._X_all = X
        self._y = y
        self._y_pred = None  # TODO : could be included in the dataset ?
        self._X_scaled = pd.DataFrame(StandardScaler().fit_transform(X))
        self._X_scaled.columns = X.columns

        self._variables = []
        for col in self._X.columns:
            var = Variable(col.title, col.dtype)
            var._explained = False  # since we're un Dataset constructor
            if col.title in ["longitude", "Longitude", "Long", "long"]:
                var._lon = True            
            if col.title in ["latitude", "Latitude", "Lat", "lat"]:
                var._lon = True
            self._variables.append(var)


    def __str__(self):
        text = "Dataset object :\n"
        text += "------------------\n"
        text += "- Number of observations:" + str(self._X.shape[0]) + "\n"
        text += "- Number of variables: " + str(self._X.shape[1]) + "\n"
        return text

    # TODO : is it useful ?
    def __len__(self):
        return self._X.shape[0]
    
    def get_variables(self) -> list:
        return self._variables

    def get_var_values(self, variable: Variable) -> pd.Series:
        return self._X[variable.getSymbol()]

    def getFullValues(self, flavour: int = REGULAR) -> pd.DataFrame:
        """
        Access non projected values for the dataset as a DataFrame.

        Parameters
        ----------
        flavour : int
            The flavour of the X values to return. Must be Dataset.REGULAR or Dataset.SCALED.

        Returns
        -------
        pd.DataFrame :
            The X values for the given flavour
        """
        if flavour == Dataset.REGULAR:
            return self._X
        elif flavour == Dataset.SCALED:
            return self._X_scaled
        else:
            raise ValueError("Bad flavour value")

    def isValidXFlavour(flavour: int) -> bool:
        """
        Returns True if the flavour is valid, False otherwise.
        """
        return flavour == Dataset.REGULAR or flavour == Dataset.SCALED

    def isValidYFlavour(flavour: int) -> bool:
        """
        Returns True if the flavour is valid, False otherwise.
        """
        return flavour == Dataset.TARGET or flavour == Dataset.PREDICTED

    def getProjValues(self, dimReducMethodType: int, dimension: int) -> pd.DataFrame:
        """Returns de projected X values using a dimensionality reduction method and target dimension (2 or 3)

        Args:
            dimReducMethodType (int): the type of dimensionality reduction method
            dimension (int, optional): Defaults to DimReducMethod.DIM_TWO.

        Returns:
            pd.DataFrame: the projected X values. May be None
        """
        if not DimReducMethod.isValidDimReducType(dimReducMethodType):
            raise ValueError("Bad dimensionality reduction type")
        if not DimReducMethod.isValidDimNumber(dimension):
            raise ValueError("Bad dimension number")

        df = None

        if dimReducMethodType not in self._X_proj:
            df = None
        elif dimension not in self._X_proj[dimReducMethodType]:
            df = None
        else:
            # logger.debug(f"Dataset.getProjValues : found a {DimReducMethod.getDimReducMethodAsStr(dimReducMethodType)} proj in {dimension}D for VS values")
            df = self._X_proj[dimReducMethodType][dimension]

        # logger.debug(f"Dataset.getProjValues : returnning {simpleType(df)}")

        return df

    def setProjValues(
        self, dimReducMethodType: int, dimension: int, values: pd.DataFrame
    ):
        """Set X_proj alues for this dimensionality reduction and  dimension."""

        # TODO we may want to check values.shape and raise value error if it does not match
        if not DimReducMethod.isValidDimReducType(dimReducMethodType):
            raise ValueError("Bad dimensionality reduction type")
        if not DimReducMethod.isValidDimNumber(dimension):
            raise ValueError("Bad dimension number")

        if dimReducMethodType not in self._X_proj:
            self._X_proj[
                dimReducMethodType
            ] = {}  # We create a new dict for this dimReducMethodType
        if dimension not in self._X_proj[dimReducMethodType]:
            self._X_proj[dimReducMethodType][
                dimension
            ] = {}  # We create a new dict for this dimension

        logger.debug(
            f"DS.setProjValues : I set new values for {DimReducMethod.getDimReducMethodAsStr(dimReducMethodType)} in {dimension}D proj"
        )
        self._X_proj[dimReducMethodType][dimension] = values

    def getYValues(self, flavour: int = TARGET) -> pd.Series:
        """
        Returns the y values of the dataset as a Series, depending on the flavour.
        """
        if flavour == Dataset.TARGET:
            return self._y
        elif flavour == Dataset.PREDICTED:
            return self._y_pred
        else:
            raise ValueError("Bad flavour value for Y")

    def setYValues(self, y: pd.Series, flavour: int = TARGET):
        """
        Sets the y values of the dataset as a Series, depending on the flavour.
        """
        if type(y) is int:
            raise ValueError(
                "Dataset.setYValues you must provide a Pandas Series, not an int)"
            )
        if y is None or len(y) == 0:
            raise ValueError("Dataset.setYValues you must provide some Y data)")
        else:
            if flavour == Dataset.TARGET:
                self._y = y
            elif flavour == Dataset.PREDICTED:
                self._y_pred = y
            else:
                raise ValueError("Bad flavour value for Y")

    def getShape(self):
        """Returns the shape of the used dataset"""
        return self._X.shape

    def setLatLon(self, lat: str, long: str):
        """
        Sets the longitude and latitude columns of the dataset.

        Parameters
        ---------
        long : str
            The name of the longitude column.
        lat : str
            The name of the latitude column.
        """
        self._lat = lat
        self._long = long

    def getLatLon(self) -> (float, float):
        """
        Returns the longitude and latitude columns of the dataset.

        Returns
        -------
        lat : str
            The name of the latitude column.
        long : str
            The name of the longitude column.
        """
        return (self._lat, self._long)


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
    def isValidOriginType(originValue: int) -> bool:
        return originValue in [
            ExplanationDataset.COMPUTED,
            ExplanationDataset.IMPORTED,
            ExplanationDataset.ANY,
        ]

    def __init__(self, values: pd.DataFrame, explanationType: int):
        """
        Constructor of the class ExplanationDataset.

        Parameters :
        ------------
        values : pandas.Dataframe
            The dataframe containing explanations values provided by the user (IMPORTED)
        explanationType : int
            Must be ExplainedValues.SHAP or ExplainedValues.LIME
        """

        if explanationType == ExplanationMethod.SHAP:
            self._shapValues = {self.IMPORTED: values}
            self._limeValues = {self.IMPORTED: None}
        elif explanationType == ExplanationMethod.LIME:
            self._shapValues = {self.IMPORTED: None}
            self._limeValues = {self.IMPORTED: values}
        else:
            raise ValueError(
                "explanationType must be ExplainedValues.SHAP or ExplainedValues.LIME"
            )

    @staticmethod
    def getOriginByStr(originValue: int) -> str:
        if originValue == ExplanationDataset.IMPORTED:
            return "imported"
        elif originValue == ExplanationDataset.COMPUTED:
            return "computed"
        elif originValue == ExplanationDataset.BOTH:
            return "both"
        elif originValue == ExplanationDataset.ANY:
            return "any"
        else:
            raise ValueError(originValue, " is a bad origin type")

    def getFullValues(self, explainationMethodType: int, origin: int = ANY):
        """Looks for imported or computed explaned values with no dimension reduction, hence "FullValues"
        Parameters:
        explainationMethodType = see ExplanationMethod integer constants
        origin = see ExplanationDataset integer constants, defaults to ANY ie. we return imported or computed
        Returns: a pandas dataframe or None
        """

        if not ExplanationMethod.isValidExplanationType(explainationMethodType):
            raise ValueError(
                explainationMethodType, " is a bad explanation method type"
            )

        storedExplanations, df = None, None

        if explainationMethodType == ExplanationMethod.SHAP:
            storedExplanations = self._shapValues
        else:
            storedExplanations = self._limeValues

        if origin != ExplanationDataset.ANY:
            # User wants a specific origin
            # logger.debug(f"XDS.getFullValues : {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} values with {ExplanationDataset.getOriginByStr(origin)} origin required")
            if not ExplanationDataset.isValidOriginType(origin):
                raise ValueError(origin, " is a bad origin type")
            if origin in storedExplanations:
                # logger.debug(f"XDS.getFullValues : found {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} values with {ExplanationDataset.getOriginByStr(origin)} origin ")
                df = storedExplanations[origin]
            else:
                # logger.debug(f"XDS.getFullValues : could not find {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} values with {ExplanationDataset.getOriginByStr(origin)} origin ")
                df = None
        else:
            # logger.debug(f"XDS.getFullValues : user wants {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} values whatever the origin")
            if ExplanationDataset.IMPORTED in storedExplanations:
                # logger.debug(f"XDS.getFullValues : found {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} values with imported origin ")
                df = storedExplanations[ExplanationDataset.IMPORTED]
            elif ExplanationDataset.COMPUTED in storedExplanations:
                # logger.debug(f"XDS.getFullValues : found {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} values with computed origin ")
                df = self._shapValues[ExplanationDataset.COMPUTED]
            else:
                # logger.debug(f"XDS.getFullValues : no {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} values")
                df = None

        # logger.debug(f"XDS.getFullValues : we return {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} {ExplanationDataset.getOriginByStr(origin)} values : {simpleType(df)}")
        return df

    def getProjValues(
        self, explainationMethodType: int, dimReducMethodType: int, dimension: int
    ) -> pd.DataFrame:
        """
        Looks for projecte values for a given type of explanation method.
        We don't store projected values per origin : if explanations are computed for a given method (say SHAP), we we'll erase the previous projected values for an imported set/

        Parameters
        ----------
        explainationMethodType : int
            Should be ExplanationMethod.SHAP or ExplanationMethod.LIME
        DimReducMethodType : int
            Should be DimReducMethod.PCA, DimReducMethod.TSNE, DimReducMethod.UMAP or DimReducMethod.PaCMAP.
        dimension : int
            The dimension of the projection (2 or 3).

        Returns
        -------
        values : pandas.Dataframe
            The values for this explainationMethodType, DimReducMethodType and dimension.
            Returns None if not computed yet.
        """

        if not ExplanationMethod.isValidExplanationType(explainationMethodType):
            raise ValueError(
                explainationMethodType, " is a bad explanation method type"
            )
        if dimReducMethodType is not None and not DimReducMethod.isValidDimReducType(
            dimReducMethodType
        ):
            raise ValueError(
                dimReducMethodType, " is a bad dimensionality reduction type"
            )
        if dimension is not None and not DimReducMethod.isValidDimNumber(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        storedProjValues, df = None, None

        if explainationMethodType == ExplanationMethod.SHAP:
            storedProjValues = self._shapValues
        else:
            storedProjValues = self._limeValues

        if dimReducMethodType in storedProjValues:
            if (
                storedProjValues[dimReducMethodType] is not None
                and dimension in storedProjValues[dimReducMethodType]
            ):
                df = storedProjValues[dimReducMethodType][dimension]

        # if df is not None :
        #     # logger.debug(f"XDS.getProjValues : proj {DimReducMethod.getDimReducMethodAsStr(dimReducMethodType)} in {dimension} dim FOUND in our {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} explanations")
        # else :
        # logger.debug(f"XDS.getProjValues : proj {DimReducMethod.getDimReducMethodAsStr(dimReducMethodType)} in {dimension}D NOT found in our {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} explanations")

        # logger.debug(f"XDS.getProjValues : we return {simpleType(df)}")
        return df

    def setFullValues(
        self, explainationMethodType: int, values: pd.DataFrame, origin: int
    ):
        """
        Use to set explained valued (not projected values)
        Parameters:
        explainationMethodType : see ExplanationMethod integer constants, can be SHAP or LIME
        values : a pandas dataframe
        origin : see ExplanationDataset integer constants, can be IMPORTED or COMPUTED
        """
        if not ExplanationMethod.isValidExplanationType(explainationMethodType):
            raise ValueError(
                explainationMethodType, " is a bad explanation method type"
            )
        if not ExplanationDataset.isValidOriginType(origin):
            raise ValueError(origin, " is a bad origin type")

        if explainationMethodType == ExplanationMethod.SHAP:
            ourStore = self._shapValues
        else:
            ourStore = self._limeValues

        if origin not in ourStore:
            ourStore[origin] = {}

        # logger.debug(f"XDS.setFullValues : we store {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} {ExplanationDataset.getOriginByStr(origin)} values with {simpleType(values)}")

        ourStore[origin] = values

    def setProjValues(
        self,
        explainationMethodType: int,
        dimReducMethodType: int,
        dimension: int,
        values: pd.DataFrame,
    ):
        """Set values for this ExplanationDataset, given an explanation method, a dimensionality reduction and a dimension.

        Args:
            explainationMethodType : int
                SHAP or LIME (see ExplanationMethod class in compute.py)
            values : pd.DataFrame
                The values to set.
            dimReducMethodTypev : int
                Type of dimensuion reduction. Can be None if values are not projected.
            dimension (int, optional): dimension of projection. Can be None if values are not projected.
        """
        if not ExplanationMethod.isValidExplanationType(explainationMethodType):
            raise ValueError(
                explainationMethodType, " is a bad explanation method type"
            )
        if dimReducMethodType is not None and not DimReducMethod.isValidDimReducType(
            dimReducMethodType
        ):
            raise ValueError(
                dimReducMethodType, " is a bad dimensionality reduction type"
            )
        if dimension is not None and not DimReducMethod.isValidDimNumber(dimension):
            raise ValueError(dimension, " is a bad dimension number")

        ourStore = None

        if explainationMethodType == ExplanationMethod.SHAP:
            ourStore = self._shapValues
        else:
            ourStore = self._limeValues

        if dimReducMethodType not in ourStore:
            ourStore[dimReducMethodType] = {}
        if dimension not in ourStore[dimReducMethodType]:
            ourStore[dimReducMethodType][dimension] = {}

        logger.debug(
            f"XDS.setProjValues : we store {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)} with {DimReducMethod.getDimReducMethodAsStr(dimReducMethodType)} proj in {dimension} dim -> {simpleType(values)}"
        )

        ourStore[dimReducMethodType][dimension] = values

    def isExplanationAvailable(self, explainationMethodType: int, origin: int) -> bool:
        """Tells wether we have this explanation method values or not for this origin"""
        if not ExplanationMethod.isValidExplanationType(explainationMethodType):
            raise ValueError(
                explainationMethodType, " is a bad explaination method type"
            )
        if not ExplanationDataset.isValidOriginType(origin):
            raise ValueError(origin, " is a bad origin type")

        storedValues = None
        if explainationMethodType == ExplanationMethod.SHAP:
            storedValues = self._shapValues
        else:
            storedValues = self._limeValues

        if origin == ExplanationDataset.ANY:
            return self.isExplanationAvailable(
                ExplanationDataset.IMPORTED
            ) or self.isExplanationAvailable(ExplanationDataset.COMPUTED)
        else:
            if origin in storedValues and storedValues[origin] is not None:
                # logger.debug(f"isExplanationAvailable : yes we have {ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)}")
                return True

    def _descProjHelper(
        self, text: str, explainationMethodType: int, dimReducMethodType: int
    ) -> str:
        if explainationMethodType == ExplanationMethod.SHAP:
            explainDict = self._shapValues
        elif explainationMethodType == ExplanationMethod.LIME:
            explainDict = self._limeValues
        else:
            raise ValueError(
                explainationMethodType, " is a bad explaination method type"
            )

        if not DimReducMethod.isValidDimReducType(dimReducMethodType):
            raise ValueError(
                dimReducMethodType, " is a bad dimensionality reduction type"
            )

        if (
            dimReducMethodType in explainDict
            and explainDict[dimReducMethodType] is not None
        ):
            text += (
                ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)
                + " values dict has "
                + DimReducMethod.getDimReducMethodAsStr(dimReducMethodType)
                + " projected values :\n"
            )
            if (
                DimReducMethod.DIM_TWO in explainDict[dimReducMethodType]
                and explainDict[dimReducMethodType][DimReducMethod.DIM_TWO] is not None
            ):
                text += (
                    "    2D :"
                    + str(
                        explainDict[dimReducMethodType][DimReducMethod.DIM_TWO].shape[0]
                    )
                    + " observations, "
                    + str(
                        self._shapValues[dimReducMethodType][
                            DimReducMethod.DIM_TWO
                        ].shape[1]
                    )
                    + " features\n"
                )
            if (
                DimReducMethod.DIM_THREE in explainDict[dimReducMethodType]
                and explainDict[dimReducMethodType][DimReducMethod.DIM_THREE]
                is not None
            ):
                text += (
                    "    3D :"
                    + str(
                        explainDict[dimReducMethodType][DimReducMethod.DIM_THREE].shape[
                            0
                        ]
                    )
                    + " observations, "
                    + str(
                        explainDict[dimReducMethodType][DimReducMethod.DIM_THREE].shape[
                            1
                        ]
                    )
                    + " features\n"
                )

        return text

    def _descExplainHelper(self, text: str, explainationMethodType: int) -> str:
        if explainationMethodType == ExplanationMethod.SHAP:
            explainDict = self._shapValues
        elif explainationMethodType == ExplanationMethod.LIME:
            explainDict = self._limeValues
        else:
            raise ValueError(
                explainationMethodType, " is a bad explaination method type"
            )

        if explainDict is None or len(explainDict) == 0:
            text += (
                ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)
                + " values dict is empty\n"
            )
        else:
            if (
                ExplanationDataset.IMPORTED in explainDict
                and explainDict[ExplanationDataset.IMPORTED] is not None
            ):
                text += (
                    ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)
                    + " values dict has imported values :\n"
                )
                text += (
                    "    "
                    + str(explainDict[ExplanationDataset.IMPORTED].shape[0])
                    + " observations, "
                    + str(explainDict[ExplanationDataset.IMPORTED].shape[1])
                    + " features\n"
                )

            if (
                ExplanationDataset.COMPUTED in explainDict
                and explainDict[ExplanationDataset.COMPUTED] is not None
            ):
                text += (
                    ExplanationMethod.getExplanationMethodAsStr(explainationMethodType)
                    + " values dict has computed values :\n"
                )
                text += (
                    "    "
                    + str(explainDict[ExplanationDataset.COMPUTED].shape[0])
                    + " observations, "
                    + +str(explainDict[ExplanationDataset.COMPUTED].shape[1])
                    + " features\n"
                )

        for projType in DimReducMethod.getDimReducMethodsAsList():
            text += self._descProjHelper(text, explainationMethodType, projType)

        return text

    def __repr__(self) -> str:
        text = "ExplanationDataset object :\n"
        text += "---------------------------\n"

        for explainType in ExplanationMethod.getExplanationMethodsAsList():
            text += self._descExplainHelper(text, explainType)
        return text
