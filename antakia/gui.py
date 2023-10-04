# Datascience imports
import logging

# Others imports
import time

# Warnings imports
import warnings
from copy import deepcopy
from importlib.resources import files

import ipyvuetify as v
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display
from ipywidgets import Layout, widgets
from sklearn import ensemble, linear_model
from sklearn.ensemble import RandomForestRegressor

from antakia.antakia import get_default_submodels

# Internal imports
from antakia.compute import (
    DimReducMethod,
    ExplanationMethod,
    computeExplanations,
    computeProjection,
    createBeeswarm,
)
from antakia.data import (  # noqa: E402
    Dataset,
    ExplanationDataset,
    ExplanationMethod,
    Model,
)
from antakia.potato import *
from antakia.utils import _function_models as function_models
from antakia.utils import (  # noqa: E402
    confLogger,
    overlapHandler,  # noqa: E402
    proposeAutoDyadicClustering,
)
from antakia.gui_utils import (
    get_app_graph, 
    get_splash_graph, 
    widget_at_address,
    AntakiaExplorer,
    createMenuBar,
    wrap_in_a_tooltip,
    add_model_slideItem,
    RuleVariableRefiner
)


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
# warnings.simplefilter(action="ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()


class GUI:
    """
    GUI class.

    A GUI instance contains all the data and variables needed to run the interface.
    The interface is built using ipyvuetify and plotly.
    It heavily relies on the IPyWidgets framework.

    Instance Attributes
    ---------------------
    _ds : Dataset
        dataset, potentially used to train the model
    _xds : ExplanationDataset
        explained data (optional)
    _model : Model
        the model to be explained
    _selection : a Potato object
        The `Potato` object containing the current selection.
    _projectionVS : a list [int, int]
        VS current projection as a list, eg. [TSNE, 2]
    _projectionES : a list [int, int]
        ES current projection as a list, eg. [UMAP, 3]
    _explanationES : a list [int, int]
        current explanation method : eg. [SHAP, IMPORTED]

    Widgets:
    _out : Output Widget
        used in the output cell
    _app_graph : a graph of nested Widgets
    _color : Pandas Series : color for each Y point
    _opacity : float
    _paCMAPparams = dictionnary containing the parameters for the PaCMAP projection
        nested keys are "previous" / "current", then "VS" / "ES", then "n_neighbors" / "MN_ratio" / "FP_ratio"
    
    _regionColor : # TODO : understand
    _regionsTable : # TODO : understand
    _regions : list of Potato objects

    _save_rules useful to keep the initial rules from the skope-rules, in order to be able to reset the rules
    _otherColumns : to keep track of the columns that are not used in the rules !
    _activate_histograms : to know if the histograms are activated or not (bug ipywidgets !). If they are activated, we have to update the histograms.
    __dyadicClusteringResult : to keep track  of the entire results from the dyadic-clustering
    _autoClusterRegionColors: the color of the Regions created by the automatic dyadic clustering
    _dyadicClusteringLabels :  to keep track of the labels from the automatic-clustering, used for the colors !
    modelIndex : to know which sub_model is selected by the user.
    _subModelsScores : to keep track of the scores of the sub-models
    _backups : A list of backups
    

    """

    # Class attributes
    VS = 0
    ES = 1

    def __init__(
        self,
        ds: Dataset,
        model: Model,
        xds: ExplanationDataset = None,
        defaultProjection: int = DimReducMethod.PaCMAP,
        dimension: int = DimReducMethod.DIM_TWO,
    ):
        """
        GUI Class constructor.

        Parameters
        ----------
        ds : Dataset
            dataset, potentially used to train the model
        model : Model
            trained model to explain
        xds : ExplainationDataset
            explained data (optional)
        defaultProjection : int
            The default projection to use. See constants in DimReducMethod class
        dimension : int
            The default dimension to use. See constants in DimReducMethod class
        """

        self._ds = ds
        self._xds = xds
        self._model = model
        if model is None:
            raise ValueError("AntakIA requires a valid model")
        if self._ds.getYValues(Dataset.TARGET) is None:
            raise ValueError("The provided Dataset doesn't contain any Y values")
        if self._ds.getYValues(Dataset.PREDICTED) is None:
            self._ds.setYValues(
                self._model.predict(self._ds.getFullValues()), Dataset.PREDICTED
            )

        if self._xds is None:
            (
                self._explanationES[0],
                self._explanationES[1],
            ) = None  # That is we need to compute it
        else:
            if self._xds.isExplanationAvailable(
                ExplanationMethod.SHAP, ExplanationDataset.IMPORTED
            ):
                self._explanationES = [
                    ExplanationMethod.SHAP,
                    ExplanationDataset.IMPORTED,
                ]
            elif self._xds.isExplanationAvailable(
                ExplanationMethod.LIME, ExplanationDataset.IMPORTED
            ):
                self._explanationES = [
                    ExplanationMethod.LIME,
                    ExplanationDataset.IMPORTED,
                ]
            else:
                logger.debugg("__init__ : empty explanation dataset")

        self._selection = None

        if not DimReducMethod.isValidDimReducType(defaultProjection):
            raise ValueError(defaultProjection, " is an invalid projection type")

        if not DimReducMethod.isValidDimNumber(dimension):
            raise ValueError(dimension, " is an invalid dimension")

        self._projectionVS = [defaultProjection, dimension]
        self._projectionES = [defaultProjection, dimension]

        self._out = widgets.Output()  # Ipwidgets output widget
        self._app_graph = None
        self._color = None
        self._opacity = 1.0
        self._paCMAPparams = {
            "previous": {
                "VS": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
                "ES": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
            },
            "current": {
                "VS": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
                "ES": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
            },
        }  # A dict ("previous", "currrent") of dict ("VS, "ES") of dict("n_neighbors", "MN_ratio", "FP_ratio")

        # ---------- Vrac -------------
        self._regionColor = None  # a lislt of what ?
        self._regionsTable = None
        self._regions = []  # a list of Potato objects
        # TODO : understand the following
        self._featureClass1 = None
        self._featureClass2 = None
        self._featureClass3 = None
        self._save_rules = None  # useful to keep the initial rules from the skope-rules, in order to be able to reset the rules
        self._otherColumns = (
            None  # to keep track of the columns that are not used in the rules !
        )
        self._activate_histograms = False  # to know if the histograms are activated or not (bug ipywidgets !). If they are activated, we have to update the histograms.
        self._histogramBinNum = 50
        self._autoClusterRegionColors = (
            []
        )  # the color of the Regions created by the automatic dyadic clustering
        self._dyadicClusteringLabels = None  # to keep track of the labels from the automatic-clustering, used for the colors !
        self.__dyadicClusteringResult = (
            None  # to keep track  of the entire results from the dyadic-clustering
        )
        self._sub_models = [
            linear_model.LinearRegression(),
            RandomForestRegressor(random_state=9),
            ensemble.GradientBoostingRegressor(random_state=9),
        ]
        self._subModelsScores = None  # to keep track of the scores of the sub-models
        self.modelIndex = one  # to know which sub_model is selected by the user.

        self._backups = []


    def get_dataset(self) -> Dataset:
        return self._ds
    
    def get_explanation_dataset(self) -> ExplanationDataset:
        return self._xds
    
    def get_selection(self) -> Potato:
        return self._selection

    def update_explanation_select(self):
        """Update the explanationMenduDict widget and only enable items that are available"""

        if self._explanationSelect is not None:
            shapImported = self._xds.isExplanationAvailable(
                ExplanationMethod.SHAP, ExplanationDataset.IMPORTED
            )
            shapComputed = self._xds.isExplanationAvailable(
                ExplanationMethod.SHAP, ExplanationDataset.COMPUTED
            )
            limeImported = self._xds.isExplanationAvailable(
                ExplanationMethod.LIME, ExplanationDataset.IMPORTED
            )
            limeComputed = self._xds.isExplanationAvailable(
                ExplanationMethod.LIME, ExplanationDataset.COMPUTED
            )

            newItems = [deepcopy(a) for a in self._explanationSelect.items]

            newItems = [
                {"text": "SHAP (imported)", "disabled": not shapImported},
                {"text": "SHAP (computed)", "disabled": not shapComputed},
                {"text": "LIME (imported)", "disabled": not limeImported},
                {"text": "LIME (computed)", "disabled": not limeComputed},
            ]

            self._explanationSelect.items = newItems

    def check_explanation(self):
        """Ensure ES computation of explanations have been done"""

        if not self._xds.isExplanationAvailable(
            self._explanationES[0], self._explanationES[1]
        ):
            if self._explanationES[1] == ExplanationDataset.IMPORTED:
                raise ValueError(
                    "You asked for an imported explanation but you did not import it"
                )
            self._xds.setFullValues(
                self._explanationES[0],
                computeExplanations(
                    self._ds.getFullValues(Dataset.REGULAR),
                    self._model,
                    self._explanationES[0],
                ),
                ExplanationDataset.COMPUTED,
            )
            logger.debug(
                f"check_explanation : we had to compute a new {ExplanationDataset.getOriginByStr(self._explanationES[1])} {ExplanationMethod.getExplanationMethodAsStr(self._explanationES[0])} values"
            )
            self.redraw_graph(GUI.ES)
        else:
            logger.debug("check_explanation : nothing to do")

    def check_both_projections(self):
        """ Check that VS and ES figures have their projection computed
        """
        self.check_projection(GUI.VS)
        self.check_projection(GUI.ES)

    def check_projection(self, side: int):
        """ Check that the figure has its projection computed. "side" refers to VS or ES class attributes
        """
        logger.debug("We are in check_projection")
        if side not in [GUI.VS, GUI.ES]:
            raise ValueError(side, " is an invalid side")

        baseSpace = projType = dim = X = params = df = sideStr = None

        # We prepare values before calling computeProjection in a generic way
        if side == GUI.VS:
            baseSpace = DimReducMethod.VS
            X = self._ds.getFullValues(Dataset.REGULAR)
            projType = self._projectionVS[0]
            dim = self._projectionVS[1]
            projValues = self._ds.getProjValues(projType, dim)
            sideStr = "VS"
        else:
            if self._explanationES[0] == ExplanationMethod.SHAP:
                baseSpace = DimReducMethod.ES_SHAP
            else:
                baseSpace = DimReducMethod.ES_LIME
            X = self._xds.getFullValues(self._explanationES[0], self._explanationES[1])
            projType = self._projectionES[0]
            dim = self._projectionES[1]
            projValues = self._xds.getProjValues(
                self._explanationES[0], projType, self._projectionES[1]
            )
            sideStr = "ES"

        newPacMAPParams = False
        if projType == DimReducMethod.PaCMAP:
            params = {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2}
            if (
                self._paCMAPparams["current"][sideStr]["n_neighbors"]
                != self._paCMAPparams["previous"][sideStr]["n_neighbors"]
            ):
                params["n_neighbors"] = self._paCMAPparams["current"][sideStr][
                    "n_neighbors"
                ]
                self._paCMAPparams["previous"][sideStr]["n_neighbors"] = params[
                    "n_neighbors"
                ]  # We store the new "previous" value
                newPacMAPParams = True
            if (
                self._paCMAPparams["current"][sideStr]["MN_ratio"]
                != self._paCMAPparams["previous"][sideStr]["MN_ratio"]
            ):
                params["MN_ratio"] = self._paCMAPparams["current"][sideStr]["MN_ratio"]
                self._paCMAPparams["previous"][sideStr]["MN_ratio"] = params[
                    "MN_ratio"
                ]  # We store the new "previous" value
                newPacMAPParams = True
            if (
                self._paCMAPparams["current"][sideStr]["FP_ratio"]
                != self._paCMAPparams["previous"][sideStr]["FP_ratio"]
            ):
                params["FP_ratio"] = self._paCMAPparams["current"][sideStr]["FP_ratio"]
                self._paCMAPparams["previous"][sideStr]["FP_ratio"] = params[
                    "FP_ratio"
                ]  # We store the new "previous" value
                newPacMAPParams = True
            logger.debug(
                f"check_projection({sideStr}) : previous params = {self._paCMAPparams['previous'][sideStr]['n_neighbors']}, {self._paCMAPparams['previous'][sideStr]['MN_ratio']}, {self._paCMAPparams['previous'][sideStr]['FP_ratio']}"
            )
            logger.debug(
                f"check_projection({sideStr}) : current params = {self._paCMAPparams['current'][sideStr]['n_neighbors']}, {self._paCMAPparams['current'][sideStr]['MN_ratio']}, {self._paCMAPparams['current'][sideStr]['FP_ratio']}"
            )

        if newPacMAPParams:
            logger.debug(
                f"check_projection({sideStr}) : new PaCMAP proj with new params['n_neighbors']={params['n_neighbors']}, params['MN_ratio']={params['MN_ratio']}, params['FP_ratio']={params['FP_ratio']}"
            )
            df = computeProjection(
                baseSpace,
                X,
                projType,
                dim,
                n_neighbors=params["n_neighbors"],
                MN_ratio=params["MN_ratio"],
                FP_ratio=params["FP_ratio"],
            )
        elif projValues is None:
            logger.debug(
                f"check_projection({sideStr}) : new {DimReducMethod.getDimReducMethodAsStr(projType)} projection"
            )
            df = computeProjection(baseSpace, X, projType, dim)
        else:
            logger.debug(f"check_projection({sideStr}) : nothing to do")

        # We set the new projected values
        if df is not None:
            if side == GUI.VS:
                self._ds.setProjValues(projType, dim, df)
            else:
                self._xds.setProjValues(self._explanationES[0], projType, dim, df)
            self.redraw_graph(side)

    def redraw_both_graphs(self):
        """ Redraws both figures
        """
        self.redraw_graph(GUI.VS)
        self.redraw_graph(GUI.ES)

    def redraw_graph(self, side: int):
        """ Redraws one figure. "side" refers to VS or ES class attributes
        """
        if side not in [GUI.VS, GUI.ES]:
            raise ValueError(side, " is an invalid side")

        projValues = figure = projType = dim = None

        if self._aeVS is None or self._aeES is None:
            return -1  # we quit this function

        if side == GUI.VS:
            sideStr = "VS"
            dim = self._projectionVS[1]
            projType = self._projectionVS[0]
            projValues = self._ds.getProjValues(projType, dim)
            figure = widget_at_address(self._app_graph, "2001")
            explainStr = ""
        else:
            sideStr = "ES"
            dim = self._projectionES[1]
            projType = self._projectionES[0]
            projValues = self._xds.getProjValues(self._explanationES[0], projType, dim)
            figure = widget_at_address(self._app_graph, "2011")
            explainStr = f"{ExplanationDataset.getOriginByStr(self._explanationES[1])} {ExplanationMethod.getExplanationMethodAsStr(self._explanationES[0])} with"

        if projValues is not None and figure is not None:
            with figure.batch_update():
                figure.data[0].marker.opacity = self._opacity
                figure.data[0].marker.color = self._color
                figure.data[0].customdata = self._color
                figure.data[0].x, figure.data[0].y = (projValues[0]), (projValues[1])
                if dim == DimReducMethod.DIM_THREE:
                    figure.data[0].z = projValues[2]
            logger.debug(f"updateFigure({sideStr}) : figure updated")
        else:
            logger.debug(
                f"updateFigure({sideStr}) : don't have the proper proj for {explainStr} {DimReducMethod.getDimReducMethodAsStr(projType)} in {dim} dimension"
            )

    def get_selection(self):
        """ Returns the current selection.
        """
        return self._selection

    def __repr__(self):
        self.display_GUI()
        return ""

    def display_GUI(self):
        """ Renders the interface
        """
        display(self._out)

        # ------------- Splash screen -----------------
        splash_graph = get_splash_graph()

        # We display the splash screen
        with self._out:
            display(splash_graph)

        splash_graph.children[1].children[2].children[0].v_model = "Values space ... "
        splash_graph.children[2].children[2].v_model = (
            "Default dimension reduction : "
            + DimReducMethod.getDimReducMethodAsStr(self._projectionVS[0])
            + " in "
            + DimReducMethod.getDimensionAsStr(self._projectionVS[1])
            + " ..."
        )

        # We render the figures
        self.check_explanation()
        self.check_both_projections()

        # We remove the Splahs screen
        self._out.clear_output(wait=True)
        del splash_graph

        # ------------- Main app  -----------------
        self._app_graph = get_app_graph()

        # We hide the 2 Progress Circular widgets (used to tell when a computation is in progress)
        widget_at_address(self._app_graph, "12020").layout.visibility = "hidden"
        widget_at_address(self._app_graph, "12120").layout.visibility = "hidden"

        # --------- Two AntakiaExporer ------------
        widget_at_address(self._app_graph, "2001") = AntakiaExplorer(self._ds, None, False)
        widget_at_address(self._app_graph, "2011") = AntakiaExplorer(self._ds, self._xds, True)

    
        # --------- Set point color ------------
        def change_color(*args, opacity: bool = True):
            # Allows you to change the color of the dots when you click on the buttons
            choice = self._app_graph.children[1].children[1].children[1].v_model
            self._color = None
            if choice == "y":
                self._color = self._ds.getYValues(Dataset.REGULAR)
            elif choice == "y^":
                self._color = self._ds.getYValues(Dataset.PREDICTED)
            elif choice == "current selection":
                self._color = ["grey"] * len(self._ds.getFullValues(Dataset.REGULAR))
                for i in range(len(self._selection.getIndexes())):
                    color[self._selection.getIndexes()[i]] = "blue"
            elif choice == "residual":
                self._color = self._ds.getYValues(Dataset.REGULAR) - self._ds.getYValues(
                    Dataset.PREDICTED
                )
                self._color = [abs(i) for i in color]
            elif choice == "regions":
                self._color = [0] * len(self._ds.getFullValues(Dataset.REGULAR))
                for i in range(len(self._ds.getFullValues(Dataset.REGULAR))):
                    for j in range(len(self._regions)):
                        if i in self._regions[j].getIndexes():
                            self._color[i] = j + 1
            elif choice == "not selected":
                self._color = ["red"] * len(self._ds.getXValues(Dataset.REGULAR))
                if len(self._regions) > 0:
                    for i in range(len(self._ds.getXValues(Dataset.REGULAR))):
                        for j in range(len(self._regions)):
                            if i in self._regions[j].getIndexes():
                                self._color[i] = "grey"
            elif choice == "auto":
                self._color = self._autoClusterRegionColors

            # Let's redraw
            self._redraw_both_graphs()

        # Set "change" event on the Button Toggle used to chose color
        widget_at_address(self._app_graph, "111").on_event("change", change_color)

        # Now both figures are defined :
        self.check_both_projections()
        self.redraw_both_graphs()

        # We set a tooltip to the 2D-3D Switch 
        widget_at_address(self._app_graph, "10") = wrap_in_a_tooltip(
            widget_at_address(self._app_graph, "10"), "Dimension of the projection"
        )

        def switch_dimension(*args):
            """
            Called when the switch changes.
            We compute the 3D proj if needed
            We theh call the AntakiaExplorer to update its figure
            """
            # We invert the dimension for both graphs
            self._projectionVS[1] = 5 - self._projectionVS[1]
            self._projectionES[1] = 5 - self._projectionES[1] 
            self.check_both_projections()
            self.redraw_both_graphs()

        widget_at_address(self._app_graph, "102").on_event("change", switch_dimension)


        # ----- Submodels slides -------
        # Let's populate the SlideGroup (306010) with our default submodels :
        for model in get_default_submodels() :
            add_model_slideItem(widget_at_address(self._app_graph, "306010"), model)

        def select_submodel(widget, event, data, args: bool = True):
            # TODO : we need to use the Varibale list in _ds
            if args is True:
                for slide_item in widget_at_address(self._app_graph, "306010").children :
                    slide_item.children[0].color = "white"
                widget.color = "blue lighten-4"
            for slide_item in widget_at_address(self._app_graph, "306010").children :
                if slide_item.children[0].color == "blue lighten-4":
                    # TODO : define selected sub_model
                    pass
        
        # We retrieve the SlideGroup (306010)
        for slide_item in widget_at_address(self._app_graph, "306010").children :
            slide_item.on_event("click", select_submodel)

        # ---- Tab 2 : refinement ----

        # Let's create 3 RuleVariableRefiner  :
        widget_at_address(self._app_graph, "30501").children = [
            RuleVariableRefiner(self, self._ds.get_variables[0]),
            RuleVariableRefiner(self, self._ds.get_variables[1]),
            RuleVariableRefiner(self, self._ds.get_variables[2])
        ]

        def update_histograms_with_rules(min, max, rule_index):
            total_list = (
                self._ds.getFullValues()
                .index[
                    self._ds.getFullValues()[self._selection.getVSRules()[rule_index][2]
                    ].between(min, max)
                ]
                .tolist()
            )
            for i in range(len(self._selection.getVSRules())):
                min = self._selection.getVSRules()[i][0]
                max = self._selection.getVSRules()[i][4]
                if i != rule_index:
                    temp_list = (
                        self._ds.getXValues()
                        .index[
                            self._ds.getXValues()[
                                self._selection.getVSRules()[i][2]
                            ].between(min, max)
                        ]
                        .tolist()
                    )
                    total_list = [g for g in total_list if g in temp_list]
            if self._selection.getMapIndexes() is not None:
                total_list = [
                    g for g in total_list if g in self._gui.get_selection().getMapIndexes()
                ]
            for i in range(len(self._selection.getVSRules())):
                for refiner in widget_at_address(self._app_graph, "30501").children:
                    # We retrieve the refiner's histogram :
                    with widget_at_address(refiner.get_widget(), "001001").batch_update():
                        widget_at_address(refiner.get_widget(), "001001").data[0].x = self._ds.getXValues()[self._selection.getVSRules()[i][2]][total_list]
                    
                    # We retrieve the refiner's beeswarm :
                    if widget_at_address(refiner.get_widget(), "001011").v_model :
                        with widget_at_address(refiner.get_widget(), "001011").batch_update():
                            y_color = [0] * len(self._xds.getFullValues(self._explanationES[0]))
                            if i == rule_index:
                                indexs = (
                                    self._ds.getXValues()
                                    .index[
                                        self._ds.getXValues()[
                                            self._selection.getVSRules()[i][2]
                                        ].between(min, max)
                                    ]
                                    .tolist()
                                )
                            else:
                                indexs = (
                                    self._ds.getFullValues().index[
                                        self._ds.getXValues()[
                                            self._selection.getVSRules()[i][2]
                                        ].between(
                                            self._selection.getVSRules()[i][0],
                                            self._selection.getVSRules()[i][4],
                                        )
                                    ].tolist()
                                )
                            for j in range(
                                len(self._xds.getFullValues(self._explanationES[0]))
                            ):
                                if j in total_list:
                                    y_color[j] = "blue"
                                elif j in indexs:
                                    y_color[j] = "#85afcb"
                                else:
                                    y_color[j] = "grey"
                            widget_at_address(refiner.get_widget(), "001011").data[0].marker.color = y_color

        def update_graph_with_rules():
            """ Redraws both graph with points in blue or grey depending on the rules
            """
            self._selection.setType(Potato.REFINED_SKR)
            newSet = self._selection.setIndexesWithSKR(True)
            y_color_skope = []
            for i in range(len(self._ds.getFullValues())):
                if i in newSet:
                    y_color_skope.append("blue")
                else:
                    y_color_skope.append("grey")

            self._color = y_color_skope
            # Let's redraw
            self._redraw_both_graphs()

        # when the value of a slider is modified, the histograms and graphs are modified
        def updateOnSkopeRule1Change(widget, event, data):
            if widget.__class__.__name__ == "RangeSlider":
                skopeSliderGroup1.children[0].v_model = skopeSlider1.v_model[0]
                skopeSliderGroup1.children[2].v_model = skopeSlider1.v_model[1]
            else:
                if (
                    skopeSliderGroup1.children[0].v_model == ""
                    or skopeSliderGroup1.children[2].v_model == ""
                ):
                    return
                else:
                    skopeSlider1.v_model = [
                        float(skopeSliderGroup1.children[0].v_model),
                        float(skopeSliderGroup1.children[2].v_model),
                    ]
            newList = [
                g
                for g in list(
                    self._ds.getXValues()[self._selection.getVSRules()[0][2]].values
                )
                if g >= skopeSlider1.v_model[0] and g <= skopeSlider1.v_model[1]
            ]
            with histogram1.batch_update():
                histogram1.data[1].x = newList
            if self._activate_histograms:
                update_histograms_with_rules(skopeSlider1.v_model[0], skopeSlider1.v_model[1], 0)
            if realTimeUpdateCheck1.v_model:
                self._selection.getVSRules()[0][0] = float(
                    deepcopy(skopeSlider1.v_model[0])
                )
                self._selection.getVSRules()[0][4] = float(
                    deepcopy(skopeSlider1.v_model[1])
                )
                ourVSSkopeCard.children = gui_utils.createRuleCard(
                    self._selection.ruleListToStr()
                )
                update_graph_with_rules()

        def updateOnSkopeRule2Change(widget, event, data):  # 1194
            if widget.__class__.__name__ == "RangeSlider":
                skopeSliderGroup2.children[0].v_model = skopeSlider2.v_model[0]
                skopeSliderGroup2.children[2].v_model = skopeSlider2.v_model[1]
            newList = [
                g
                for g in list(
                    self._ds.getXValues()[self._selection.getVSRules()[1][2]].values
                )
                if g >= skopeSlider2.v_model[0] and g <= skopeSlider2.v_model[1]
            ]
            with histogram2.batch_update():
                histogram2.data[1].x = newList
            if self._activate_histograms:
                update_histograms_with_rules(skopeSlider2.v_model[0], skopeSlider2.v_model[1], 1)
            if realTimeUpdateCheck2.v_model:
                self._selection.getVSRules()[1][0] = float(
                    deepcopy(skopeSlider2.v_model[0])
                )
                self._selection.getVSRules()[1][4] = float(
                    deepcopy(skopeSlider2.v_model[1])
                )
                ourVSSkopeCard.children = gui_utils.createRuleCard(
                    self._selection.ruleListToStr()
                )
                update_graph_with_rules()

        def updateOnSkopeRule3Change(widget, event, data):
            if widget.__class__.__name__ == "RangeSlider":
                skopeSliderGroup3.children[0].v_model = skopeSlider3.v_model[0]
                skopeSliderGroup3.children[2].v_model = skopeSlider3.v_model[1]
            newList = [
                g
                for g in list(
                    self._ds.getXValues()[self._selection.getVSRules()[2][2]].values
                )
                if g >= skopeSlider3.v_model[0] and g <= skopeSlider3.v_model[1]
            ]
            with histogram3.batch_update():
                histogram3.data[1].x = newList
            if self._activate_histograms:
                update_histograms_with_rules(skopeSlider3.v_model[0], skopeSlider3.v_model[1], 2)
            if realTimeUpdateCheck3.v_model:
                self._selection.getVSRules()[2][0] = float(
                    deepcopy(skopeSlider3.v_model[0])
                )
                self._selection.getVSRules()[2][4] = float(
                    deepcopy(skopeSlider3.v_model[1])
                )
                ourVSSkopeCard.children = gui_utils.createRuleCard(
                    self._selection.ruleListToStr()
                )
                update_graph_with_rules()

        # cwhen the user validates the updates he makes on a rule
        def onSkopeSlider1Change(*change):
            a = deepcopy(float(skopeSlider1.v_model[0]))
            b = deepcopy(float(skopeSlider1.v_model[1]))
            self._selection.getVSRules()[0][0] = a
            self._selection.getVSRules()[0][4] = b
            ourVSSkopeCard.children = gui_utils.createRuleCard(
                self._selection.ruleListToStr()
            )
            update_graph_with_rules()
            updateSubModelsScores(None)

        def onSkopeSlider2Change(*change):
            self._selection.getVSRules()[1][0] = float(skopeSlider2.v_model[0])
            self._selection.getVSRules()[1][4] = float(skopeSlider2.v_model[1])
            ourVSSkopeCard.children = gui_utils.createRuleCard()
            update_graph_with_rules()
            updateSubModelsScores(None)

        def onSkopeSlider3Change(*change):
            self._selection.getVSRules()[2][0] = float(skopeSlider3.v_model[0])
            self._selection.getVSRules()[2][4] = float(skopeSlider3.v_model[1])
            ourVSSkopeCard.children = gui_utils.createRuleCard()
            update_graph_with_rules()
            updateSubModelsScores(None)

        validateSkopeChangeBtn1.on_event("click", onSkopeSlider1Change)
        validateSkopeChangeBtn2.on_event("click", onSkopeSlider2Change)
        validateSkopeChangeBtn3.on_event("click", onSkopeSlider3Change)

        def updateSubModelsScores(temp):
            # TODO we should implement DATASET.length()
            boolList = [True] * len(self._ds.getXValues())
            for i in range(len(self._ds.getXValues())):
                for j in range(len(self._selection.getVSRules())):
                    colIndex = list(self._ds.getXValues().columns).index(
                        self._selection.getVSRules()[j][2]
                    )
                    if (
                        self._selection.getVSRules()[j][0]
                        > self._ds.getXValues().iloc[i, colIndex]
                        or self._ds.getXValues().iloc[i, colIndex]
                        > self._selection.getVSRules()[j][4]
                    ):
                        boolList[i] = False
            temp = [i for i in range(len(self._ds.getXValues())) if boolList[i]]

            result_models = function_models(
                self._ds.getXValues().iloc[temp, :],
                self.atk.dataset.y.iloc[temp],
                self.sub_models,
            )
            newScore = []
            for i in range(len(self.sub_models)):
                newScore.append(
                    compute.function_score(
                        self.atk.dataset.y.iloc[temp], result_models[i][-2]
                    )
                )
            initialScore = compute.function_score(
                self.atk.dataset.y.iloc[temp], self.atk.dataset.y_pred[temp]
            )
            if initialScore == 0:
                delta = ["/"] * len(self.sub_models)
            else:
                delta = [
                    round(100 * (initialScore - newScore[i]) / initialScore, 1)
                    for i in range(len(self.sub_models))
                ]

            self._subModelsScores = []
            for i in range(len(self.sub_models)):
                self._subModelsScores.append(
                    [
                        newScore[i],
                        initialScore,
                        delta[i],
                    ]
                )

            # to generate a string for the scores
            # TODO: different score for the classification! Recall/precision!
            def _scoreToStr(i):
                if newScore[i] == 0:
                    return (
                        "MSE = "
                        + str(newScore[i])
                        + " (against "
                        + str(initialScore)
                        + ", +"
                        + "∞"
                        + "%)"
                    )
                else:
                    if round(100 * (initialScore - newScore[i]) / initialScore, 1) > 0:
                        return (
                            "MSE = "
                            + str(newScore[i])
                            + " (against "
                            + str(initialScore)
                            + ", +"
                            + str(
                                round(
                                    100 * (initialScore - newScore[i]) / initialScore, 1
                                )
                            )
                            + "%)"
                        )
                    else:
                        return (
                            "MSE = "
                            + str(newScore[i])
                            + " (against "
                            + str(initialScore)
                            + ", "
                            + str(
                                round(
                                    100 * (initialScore - newScore[i]) / initialScore, 1
                                )
                            )
                            + "%)"
                        )

            for i in range(len(self.sub_models)):
                subModelslides.children[i].children[0].children[
                    1
                ].children = _scoreToStr(i)

        # Called when the user clicks on the "add" button and computes the rules
        def updateSkopeRules(*sender):
            loadingModelsProgLinear.class_ = "d-flex"

            self._activate_histograms = True

            if self._selection.getYMaskList() is None:
                ourESSkopeText.children[1].children = [
                    widgets.HTML("Please select points")
                ]
                ourVSSkopeText.children[1].children = [
                    widgets.HTML("Please select points")
                ]
            elif (
                0 not in self._selection.getYMaskList()
                or 1 not in self._selection.getYMaskList()
            ):
                ourESSkopeText.children[1].children = [
                    widgets.HTML("You can't choose everything/nothing !")
                ]
                ourVSSkopeText.children[1].children = [
                    widgets.HTML("You can't choose everything/nothing !")
                ]
            else:
                # We compute the Skope Rules for ES
                self._selection.applySkopeRules(0.2, 0.2)
                print(self._selection.getVSRules())
                # If no rule for one of the two, nothing is displayed
                if self._selection.hasARulesDefined() is False:
                    ourESSkopeText.children[1].children = [
                        widgets.HTML("No rule found")
                    ]
                    ourVSSkopeText.children[1].children = [
                        widgets.HTML("No rule found")
                    ]
                # Otherwise we display the rules
                else:
                    # We start with VS space
                    ourVSSkopeText.children[0].children[3].children = [
                        "p = "
                        + str(self._selection.getVSScore()[0])
                        + "%"
                        + " r = "
                        + str(self._selection.getVSScore()[1])
                        + "%"
                        + " ext. of the tree = "
                        + str(self._selection.getVSScore()[2])
                    ]

                    # There we find the values ​​of the skope to use them for the sliders
                    columns_rules = [
                        self._selection.getVSRules()[i][2]
                        for i in range(len(self._selection.getVSRules()))
                    ]
                    new_columns_rules = []
                    for i in range(len(columns_rules)):
                        if columns_rules[i] not in new_columns_rules:
                            new_columns_rules.append(columns_rules[i])
                    columns_rules = new_columns_rules

                    other_columns = [
                        g
                        for g in self._ds.getXValues().columns
                        if g not in columns_rules
                    ]

                    addAnotherFeatureWgt.explanationsMenuDict = other_columns
                    addAnotherFeatureWgt.v_model = other_columns[0]

                    # self._selection.getVSRules() = self._selection.getVSRules()

                    ourVSSkopeCard.children = gui_utils.createRuleCard(
                        self._selection.ruleListToStr()
                    )

                    [new_y, marker] = createBeeswarm(
                        self._ds,
                        self._xds,
                        self._explanationES[0],
                        self._selection.getVSRules()[0][2],
                    )
                    beeswarm1.data[0].y = deepcopy(new_y)
                    beeswarm1.data[0].x = self._xds.getFullValues(
                        self._explanationES[0]
                    )[columns_rules[0]]
                    beeswarm1.data[0].marker = marker

                    if (
                        len(
                            set(
                                [
                                    self._selection.getVSRules()[i][2]
                                    for i in range(len(self._selection.getVSRules()))
                                ]
                            )
                        )
                        > 1
                    ):
                        [new_y, marker] = createBeeswarm(
                            self._ds,
                            self._xds,
                            self._explanationES[0],
                            self._selection.getVSRules()[1][2],
                        )
                        beeswarm2.data[0].y = deepcopy(new_y)
                        beeswarm2.data[0].x = self._xds.getFullValues(
                            self._explanationES[0]
                        )[columns_rules[1]]
                        beeswarm2.data[0].marker = marker

                    if (
                        len(
                            set(
                                [
                                    self._selection.getVSRules()[i][2]
                                    for i in range(len(self._selection.getVSRules()))
                                ]
                            )
                        )
                        > 2
                    ):
                        [new_y, marker] = createBeeswarm(
                            self._ds,
                            self._xds,
                            self._explanationES[0],
                            self._selection.getVSRules()[2][2],
                        )
                        beeswarm3.data[0].y = deepcopy(new_y)
                        beeswarm3.data[0].x = self._xds.getFullValues(
                            self._explanationES[0]
                        )[columns_rules[2]]
                        beeswarm3.data[0].marker = marker

                    y_shape_skope = []
                    y_color_skope = []
                    y_opa_skope = []
                    for i in range(len(self._ds.getXValues())):
                        if i in self._selection.getIndexes():
                            y_shape_skope.append("circle")
                            y_color_skope.append("blue")
                            y_opa_skope.append(0.5)
                        else:
                            y_shape_skope.append("cross")
                            y_color_skope.append("grey")
                            y_opa_skope.append(0.5)
                    colorChoiceBtnToggle.v_model = "Current selection"
                    changeMarkersColor(None)

                    skopeAccordion.children = [
                        accordionGrp1,
                    ]

                    accordionGrp1.children[0].children[0].children = (
                        "X1 (" + columns_rules[0].replace("_", " ") + ")"
                    )

                    if len(columns_rules) > 1:
                        skopeAccordion.children = [
                            accordionGrp1,
                            accordionGrp2,
                        ]
                        accordionGrp2.children[0].children[0].children = (
                            "X2 (" + columns_rules[1].replace("_", " ") + ")"
                        )
                    if len(columns_rules) > 2:
                        skopeAccordion.children = [
                            accordionGrp1,
                            accordionGrp2,
                            in_accordion3_n,
                        ]
                        in_accordion3_n.children[0].children[0].children = (
                            "X3 (" + columns_rules[2].replace("_", " ") + ")"
                        )

                    _featureClass1 = gui_utils.create_class_selector(
                        self,
                        columns_rules[0],
                        self._selection.getVSRules()[0][0],
                        self._selection.getVSRules()[0][4],
                        fig_size=figureSizeSlider.v_model,
                    )  # 1490
                    if len(columns_rules) > 1:
                        _featureClass2 = gui_utils.create_class_selector(
                            self,
                            columns_rules[1],
                            self._selection.getVSRules()[1][0],
                            self._selection.getVSRules()[1][4],
                            fig_size=figureSizeSlider.v_model,
                        )
                    if len(columns_rules) > 2:
                        _featureClass3 = gui_utils.create_class_selector(
                            self,
                            columns_rules[2],
                            self._selection.getVSRules()[2][0],
                            self._selection.getVSRules()[2][4],
                            fig_size=figureSizeSlider.v_model,
                        )

                    for ii in range(len(_featureClass1.children[2].children)):
                        _featureClass1.children[2].children[ii].on_event(
                            "change", updateOnContinuousChanges1
                        )

                    for ii in range(len(_featureClass2.children[2].children)):
                        _featureClass2.children[2].children[ii].on_event(
                            "change", updateOnContinuousChanges2
                        )

                    for ii in range(len(_featureClass3.children[2].children)):
                        _featureClass3.children[2].children[ii].on_event(
                            "change", updateOnContinuousChanges3
                        )

                    if (
                        self._xds.getLatLon()[0] in colums_rules
                        and self._xds.getLatLon()[0] in colums_rules
                    ):
                        addMapBtn.disabled = False
                    else:
                        addMapBtn.disabled = True

                    skopeSlider1.min = -10e10
                    skopeSlider1.max = 10e10
                    skopeSlider2.min = -10e10
                    skopeSlider2.max = 10e10
                    skopeSlider3.min = -10e10
                    skopeSlider3.max = 10e10

                    skopeSlider1.max = max(self._ds.getXValues()[columns_rules[0]])
                    skopeSlider1.min = min(self._ds.getXValues()[columns_rules[0]])
                    skopeSlider1.v_model = [
                        self._selection.getVSRules()[0][0],
                        self._selection.getVSRules()[0][-1],
                    ]
                    [
                        skopeSliderGroup1.children[0].v_model,
                        skopeSliderGroup1.children[2].v_model,
                    ] = [skopeSlider1.v_model[0], skopeSlider1.v_model[1]]

                    if len(self._selection.getVSRules()) > 1:
                        skopeSlider2.max = max(self._ds.getXValues()[columns_rules[1]])
                        skopeSlider2.min = min(self._ds.getXValues()[columns_rules[1]])
                        skopeSlider2.v_model = [
                            self._selection.getVSRules()[1][0],
                            self._selection.getVSRules()[1][-1],
                        ]
                        [
                            skopeSliderGroup2.children[0].v_model,
                            skopeSliderGroup2.children[2].v_model,
                        ] = [skopeSlider2.v_model[0], skopeSlider2.v_model[1]]

                    if len(self._selection.getVSRules()) > 2:
                        skopeSlider3.max = max(self._ds.getXValues()[columns_rules[2]])
                        skopeSlider3.min = min(self._ds.getXValues()[columns_rules[2]])
                        skopeSlider3.v_model = [
                            self._selection.getVSRules()[2][0],
                            self._selection.getVSRules()[2][-1],
                        ]
                        [
                            skopeSliderGroup3.children[0].v_model,
                            skopeSliderGroup3.children[2].v_model,
                        ] = [
                            skopeSlider3.v_model[0],
                            skopeSlider3.v_model[1],
                        ]

                    with histogram1.batch_update():
                        histogram1.data[0].x = list(
                            self._ds.getXValues()[columns_rules[0]]
                        )
                        df_respect1 = self._selection.respectOneRule(0)
                        histogram1.data[1].x = list(df_respect1[columns_rules[0]])
                    if (
                        len(
                            set(
                                [
                                    self._selection.getVSRules()[i][2]
                                    for i in range(len(self._selection.getVSRules()))
                                ]
                            )
                        )
                        > 1
                    ):
                        with histogram2.batch_update():
                            histogram2.data[0].x = list(
                                self._ds.getXValues()[columns_rules[1]]
                            )
                            df_respect2 = self._selection.respectOneRule(1)
                            histogram2.data[1].x = list(df_respect2[columns_rules[1]])
                    if (
                        len(
                            set(
                                [
                                    self._selection.getVSRules()[i][2]
                                    for i in range(len(self._selection.getVSRules()))
                                ]
                            )
                        )
                        > 2
                    ):
                        with histogram3.batch_update():
                            histogram3.data[0].x = list(
                                self._ds.getXValues()[columns_rules[2]]
                            )
                            df_respect3 = self._selection.respectOneRule(2)
                            histogram3.data[1].x = list(df_respect3[columns_rules[2]])

                    update_histograms_with_rules(
                        skopeSlider1.v_model[0], skopeSlider1.v_model[1], 0
                    )

                    ourVSSkopeText.children[0].children[3].children = [
                        # str(skope_rules_clf.rules_[0])
                        # + "\n"
                        "p = "
                        + str(self._selection.getESScore()[0])
                        + "%"
                        + " r = "
                        + str(self._selection.getESScore()[1])
                        + "%"
                        + " ext. of the tree ="
                        + str(self._selection.getESScore()[2])
                    ]
                    ourESSkopeCard.children = gui_utils.createRuleCard(
                        self._selection.ruleListToStr(False)
                    )  # We want ES Rules printed
                    updateSubModelsScores(self._selection.getIndexes())

                    accordionGrp1.children[0].disabled = False
                    accordionGrp2.children[0].disabled = False
                    in_accordion3_n.children[0].disabled = False

            skopeSlider1.on_event("input", updateOnSkopeRule1Change)
            skopeSlider2.on_event("input", updateOnSkopeRule2Change)
            skopeSlider3.on_event("input", updateOnSkopeRule3Change)

            skopeSliderGroup1.children[0].on_event("input", updateOnSkopeRule1Change)
            skopeSliderGroup1.children[2].on_event("input", updateOnSkopeRule1Change)

            loadingModelsProgLinear.class_ = "d-none"

            self._save_rules = deepcopy(self._selection.getVSRules())

            changeMarkersColor(None)

        def reinitSkopeRules(*b):
            self._selection.setVSRules(self._save_rules)
            updateSkopeRules(None)
            updateSubModelsScores(None)

        reinitSkopeBtn.on_event("click", reinitSkopeRules)

        # Here to see the values ​​of the selected points (VZ and ES)
        out_selec = v.Layout(
            style_="min-width: 47%; max-width: 47%",
            children=[
                v.Html(
                    tag="h4",
                    children=["Select points on the figure to see their values ​​here"],
                )
            ],
        )

        out_selec_SHAP = v.Layout(
            style_="min-width: 47%; max-width: 47%",
            children=[
                v.Html(
                    tag="h4",
                    children=[
                        "Select points on the figure to see their SHAP values ​​here"
                    ],
                )
            ],
        )

        out_selec_all = v.Alert(
            max_height="400px",
            style_="overflow: auto",
            elevation="0",
            children=[
                v.Row(
                    class_="d-flex flex-row justify-space-between",
                    children=[
                        out_selec,
                        v.Divider(class_="ma-2", vertical=True),
                        out_selec_SHAP,
                    ],
                ),
            ],
        )

        # To see the data of the current selection
        out_accordion = v.ExpansionPanels(
            class_="ma-2",
            children=[
                v.ExpansionPanel(
                    children=[
                        v.ExpansionPanelHeader(children=["Data selected"]),
                        v.ExpansionPanelContent(children=[out_selec_all]),
                    ]
                )
            ],
        )

        findClusterBtn = v.Btn(
            class_="ma-1 mt-2 mb-0",
            elevation="2",
            children=[v.Icon(children=["mdi-magnify"]), "Find clusters"],
        )

        clustersSlider = v.Slider(
            style_="width : 30%",
            class_="ma-3 mb-0",
            min=2,
            max=20,
            step=1,
            v_model=3,
            disabled=True,
        )

        clustersSliderTxt = v.Html(
            tag="h3",
            class_="ma-3 mb-0",
            children=["Number of clusters " + str(clustersSlider.v_model)],
        )

        def clusterSliderGrp(*b):
            clustersSliderTxt.children = [
                "Number of clusters " + str(clustersSlider.v_model)
            ]

        clustersSlider.on_event("input", clusterSliderGrp)

        clustersNumberCheck = v.Checkbox(
            v_model=True, label="Optimal number of clusters :", class_="ma-3"
        )

        def clustersNumberCheckChange(*b):
            clustersSlider.disabled = clustersNumberCheck.v_model

        clustersNumberCheck.on_event("change", clustersNumberCheckChange)

        clusterGrp = v.Layout(
            class_="d-flex flex-row",
            children=[
                findClusterBtn,
                clustersNumberCheck,
                clustersSlider,
                clustersSliderTxt,
            ],
        )

        new_df = pd.DataFrame([], columns=["Region #", "Number of points"])
        columns = [{"text": c, "sortable": True, "value": c} for c in new_df.columns]
        clusterResultsTable = v.DataTable(
            class_="w-100",
            style_="width : 100%",
            v_model=[],
            show_select=False,
            headers=columns,
            explanationsMenuDict=new_df.to_dict("records"),
            item_value="Region #",
            item_key="Region #",
            hide_default_footer=True,
        )
        clusterResults = v.Row(
            children=[
                v.Layout(
                    class_="flex-grow-0 flex-shrink-0",
                    children=[v.Btn(class_="d-none", elevation=0, disabled=True)],
                ),
                v.Layout(
                    class_="flex-grow-1 flex-shrink-0",
                    children=[clusterResultsTable],
                ),
            ],
        )

        # Allows you to make dyadic clustering
        def dynamicClustering(*b):
            loadingClustersProgLinear.class_ = "d-flex"
            if clustersNumberCheck.v_model:
                result = proposeAutoDyadicClustering(
                    self._ds.getXValues(Dataset.SCALED),
                    self._xds.getFullValues(self._explanationES[0]),
                    3,
                    True,
                )
            else:
                result = proposeAutoDyadicClustering(
                    self._ds.getXValues(Dataset.SCALED),
                    self._xds.getFullValues(self._explanationES[0]),
                    clustersSlider.v_model,
                    False,
                )
            self._dyadicClusteringResult = result
            labels = result[1]
            self._dyadicClusteringLabels = labels
            with self._leftVSFigure.batch_update():
                self._leftVSFigure.data[0].marker.color = labels
                self._leftVSFigure.update_traces(marker=dict(showscale=False))
            with self._rightESFigure.batch_update():
                self._rightESFigure.data[0].marker.color = labels
            with self._leftVSFigure3D.batch_update():
                self._leftVSFigure3D.data[0].marker.color = labels
                self._leftVSFigure3D.update_traces(marker=dict(showscale=False))
            with self._rightESFigure3D.batch_update():
                self._rightESFigure3D.data[0].marker.color = labels
            regionsLabels = result[0]
            new_df = []
            for i in range(len(regionsLabels)):
                new_df.append(
                    [
                        i + 1,
                        len(regionsLabels[i]),
                        str(
                            round(
                                len(regionsLabels[i])
                                / len(self._ds.getXValues())
                                * 100,
                                2,
                            )
                        )
                        + "%",
                    ]
                )
            new_df = pd.DataFrame(
                new_df,
                columns=["Region #", "Number of points", "Percentage of the dataset"],
            )
            data = new_df.to_dict("records")
            columns = [
                {"text": c, "sortable": False, "value": c} for c in new_df.columns
            ]
            clusterResultsTable = v.DataTable(
                class_="w-100",
                style_="width : 100%",
                show_select=False,
                single_select=True,
                v_model=[],
                headers=columns,
                explanationsMenuDict=data,
                item_value="Region #",
                item_key="Region #",
                hide_default_footer=True,
            )
            all_chips = []
            all_radio = []
            N_antakiaStepsCard = len(regionsLabels)
            Multip = 100
            debut = 0
            fin = (N_antakiaStepsCard * Multip - 1) * (1 + 1 / (N_antakiaStepsCard - 1))
            pas = (N_antakiaStepsCard * Multip - 1) / (N_antakiaStepsCard - 1)
            scale_colors = np.arange(debut, fin, pas)
            a = 0
            for i in scale_colors:
                color = sns.color_palette(
                    "viridis", N_antakiaStepsCard * Multip
                ).as_hex()[round(i)]
                all_chips.append(v.Chip(class_="rounded-circle", color=color))
                all_radio.append(v.Radio(class_="mt-4", value=str(a)))
                a += 1
            all_radio[-1].class_ = "mt-4 mb-0 pb-0"
            part_for_radio = v.RadioGroup(
                v_model=None,
                class_="mt-10 ml-7",
                style_="width : 10%",
                children=all_radio,
            )
            part_for_chips = v.Col(
                class_="mt-10 mb-2 ml-0 d-flex flex-column justify-space-between",
                style_="width : 10%",
                children=all_chips,
            )
            clusterResults = v.Row(
                children=[
                    v.Layout(
                        class_="flex-grow-0 flex-shrink-0", children=[part_for_radio]
                    ),
                    v.Layout(
                        class_="flex-grow-0 flex-shrink-0", children=[part_for_chips]
                    ),
                    v.Layout(
                        class_="flex-grow-1 flex-shrink-0",
                        children=[clusterResultsTable],
                    ),
                ],
            )
            tabOneSelectionColumn.children = tabOneSelectionColumn.children[:-1] + [
                clusterResults
            ]

            colorChoiceBtnToggle.v_model = "Clustering auto"

            tabOneSelectionColumn.children[-1].children[0].children[0].on_event(
                "change", selectedClusterChange
            )
            loadingClustersProgLinear.class_ = "d-none"
            return N_antakiaStepsCard

        # When we select a region created by the automatic dyadic clustering
        def selectedClusterChange(widget, event, data):  # 1803
            result = _dyadicClusteringResult
            labels = result[1]
            index = tabOneSelectionColumn.children[-1].children[0].children[0].v_model
            liste = [i for i, d in enumerate(labels) if d == float(index)]
            lassoSelection(None, None, None, liste)
            colorChoiceBtnToggle.v_model = "Clustering auto"
            changeMarkersColor(opacity=False)

        findClusterBtn.on_event("click", dynamicClustering)

        # Called as soon as the points are selected (step 1)
        # TODO : we should type and provide hints fort this fonction
        def lassoSelection(trace, points, selector, *args):  # 1815
            if len(args) > 0:
                passedList = args[0]
                selectedDots = passedList
            else:
                selectedDots = points.point_inds  # ids
            self._selection = Potato(
                self._ds, self._xds, self._explanationES[0], selectedDots, Potato.LASSO
            )
            if len(selectedDots) == 0:
                selectionCard.children[0].children[1].children = "0 point !"
                selectionText.value = startInitialText
                return

            selectionCard.children[0].children[1].children = (
                str(len(selectedDots))
                + " points selected ("
                + str(round(len(selectedDots) / self._ds.__len__() * 100, 2))
                + "% of the overall)"
            )
            selectionText.value = (
                initialText
                + str(len(selectedDots))
                + " points selected ("
                + str(
                    round(
                        len(selectedDots)
                        / len(self._ds.getFullValues(Dataset.REGULAR))
                        * 100,
                        2,
                    )
                )
                + "% of the overall)"
            )
            opa = []  # stands for Opacity
            for i in range(len(self._rightESFigure.data[0].x)):
                if i in selectedDots:
                    opa.append(1)
                else:
                    opa.append(0.1)

            with self._leftVSFigure.batch_update():
                self._leftVSFigure.data[0].marker.opacity = opa
            with self._rightESFigure.batch_update():
                self._rightESFigure.data[0].marker.opacity = opa

            # TODO : why duplicate X ?
            XX = self._ds.getFullValues(Dataset.REGULAR).copy()

            XXmean = (
                pd.DataFrame(
                    XX.iloc[self._selection.getIndexes(), :]
                    .mean(axis=0)
                    .values.reshape(1, -1),
                    columns=XX.columns,
                )
                .round(2)
                .rename(index={0: "Mean of the selection"})
            )
            XXmean_tot = (
                pd.DataFrame(XX.mean(axis=0).values.reshape(1, -1), columns=XX.columns)
                .round(2)
                .rename(index={0: "Mean of the whole dataset"})
            )
            XXmean = pd.concat([XXmean, XXmean_tot], axis=0)
            SHAP_mean = (
                pd.DataFrame(
                    self._xds.getFullValues(self._explanationES[0])
                    .iloc[self._selection.getIndexes(), :]
                    .mean(axis=0)
                    .values.reshape(1, -1),
                    columns=self._xds.getFullValues(self._explanationES[0]).columns,
                )
                .round(2)
                .rename(index={0: "Mean of the selection"})
            )
            SHAP_mean_tot = (
                pd.DataFrame(
                    self._xds.getFullValues(self._explanationES[0])
                    .mean(axis=0)
                    .values.reshape(1, -1),
                    columns=self._xds.getFullValues(self._explanationES[0]).columns,
                )
                .round(2)
                .rename(index={0: "Mean of the whole dataset"})
            )
            SHAP_mean = pd.concat([SHAP_mean, SHAP_mean_tot], axis=0)

            XXmean.insert(
                loc=0,
                column=" ",
                value=["Mean of the selection", "Mean of the whole dataset"],
            )
            data = XXmean.to_dict("records")
            columns = [
                {"text": c, "sortable": True, "value": c} for c in XXmean.columns
            ]

            out_selec_table_means = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                explanationsMenuDict=data.copy(),
                hide_default_footer=True,
                disable_sort=True,
            )

            data = XX.iloc[self._selection.getIndexes(), :].round(3).to_dict("records")
            columns = [{"text": c, "sortable": True, "value": c} for c in XX.columns]

            out_selec_table = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                explanationsMenuDict=data.copy(),
            )

            out_selec.children = [
                v.Col(
                    class_="d-flex flex-column justify-center align-center",
                    children=[
                        v.Html(tag="h3", children=["Values Space"]),
                        out_selec_table_means,
                        v.Divider(class_="ma-6"),
                        v.Html(tag="h4", children=["Entire dataset:"], class_="mb-2"),
                        out_selec_table,
                    ],
                )
            ]

            SHAP_mean.insert(
                loc=0,
                column=" ",
                value=["Mean of the selection", "Mean of the whole dataset"],
            )
            data = SHAP_mean.to_dict("records")
            columns = [
                {"text": c, "sortable": True, "value": c} for c in SHAP_mean.columns
            ]

            out_selec_table_means = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                explanationsMenuDict=data.copy(),
                hide_default_footer=True,
                disable_sort=True,
            )

            data = (
                self._xds.getFullValues(self._explanationES[0])
                .iloc[self._selection.getIndexes(), :]
                .round(3)
                .to_dict("records")
            )
            columns = [
                {"text": c, "sortable": True, "value": c}
                for c in self._xds.getFullValues(self._explanationES[0]).columns
            ]

            out_selec_table = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                explanationsMenuDict=data.copy(),
            )

            out_selec_SHAP.children = [
                v.Col(
                    class_="d-flex flex-column justify-center align-center",
                    children=[
                        v.Html(tag="h3", children=["Explanatory Space"]),
                        out_selec_table_means,
                        v.Divider(class_="ma-6"),
                        v.Html(tag="h4", children=["Entire dataset:"], class_="mb-2"),
                        out_selec_table,
                    ],
                )
            ]
            # End lasso

        # Called when validating a tile to add it to the set of Regions
        def newRegionValidated(*args):
            if len(args) == 0:
                pass
            elif self._selection in self._regions:
                print("AntakIA WARNING: this region is already in the set of Regions")
            else:
                self._selection.setType(Potato.REGION)
                if self.modelIndex is None:
                    modelName = None
                    modelScore = [1, 1, 1]
                else:
                    modelName = self.sub_models[self.modelIndex].__class__.__name__
                    modelScore = self._subModelsScores[self.modelIndex]
                if self._selection.getVSRules() is None:
                    return

                # self._selection.setIndexesWithRules() # TODO not sur wa have to call that

                newIndexes = deepcopy(self._selection.getIndexes())
                self._selection.sub_model["name"], self.selection.sub_model["score"] = (
                    modelName,
                    modelScore,
                )
                # We check that all the points of the new region belong only to it: we will modify the existing tiles
                self._regions = overlapHandler(self._regions, newIndexes)
                self._selection.setNewIndexes(newIndexes)

            self._regionColor = [0] * self._ds.__len__()
            if self._regions is not None:
                for i in range(len(self._regionColor)):
                    for j in range(len(self._regions)):
                        if i in self._regions[j].getIndexes():
                            self._regionColor[i] = j + 1
                            break

            toute_somme = 0
            temp = []
            score_tot = 0
            score_tot_glob = 0
            autre_toute_somme = 0
            for i in range(len(self._regions)):
                if self._regions[i].getSubModel()["score"] is None:
                    temp.append(
                        [
                            i + 1,
                            len(self._regions[i]),
                            np.round(
                                len(self._regions[i])
                                / len(self._ds.getFullValues(Dataset.REGULAR))
                                * 100,
                                2,
                            ),
                            "/",
                            "/",
                            "/",
                            "/",
                        ]
                    )
                else:
                    temp.append(
                        [
                            i + 1,
                            len(self._regions[i]),
                            np.round(
                                len(self._regions[i])
                                / len(self._ds.getXValues())
                                * 100,
                                2,
                            ),
                            self._regions[i].getSubModel()["model"],
                            self._regions[i].getSubModel()["score"][0],
                            self._regions[i].getSubModel()["score"][1],
                            str(self._regions[i].getSubModel()["score"][2]) + "%",
                        ]
                    )
                    score_tot += self._regions[i].getSubModel()["score"][0] * len(
                        self._regions[i]
                    )
                    score_tot_glob += self._regions[i].getSubModel()["score"][1] * len(
                        self._regions[i]
                    )
                    autre_toute_somme += len(self._regions[i])
                toute_somme += len(self._regions[i])
            if autre_toute_somme == 0:
                score_tot = "/"
                score_tot_glob = "/"
                percent = "/"
            else:
                score_tot = round(score_tot / autre_toute_somme, 3)
                score_tot_glob = round(score_tot_glob / autre_toute_somme, 3)
                percent = (
                    str(round(100 * (score_tot_glob - score_tot) / score_tot_glob, 1))
                    + "%"
                )
            temp.append(
                [
                    "Total",
                    toute_somme,
                    np.round(toute_somme / len(self._ds) * 100, 2),
                    "/",
                    score_tot,
                    score_tot_glob,
                    percent,
                ]
            )
            pd.DataFrame(
                temp,
                columns=[
                    "Region #",
                    "Number of points",
                    "% of the dataset",
                    "Model",
                    "Score of the sub-model (MSE)",
                    "Score of the global model (MSE)",
                    "Gain in MSE",
                ],
            )

        validateRegionBtn.on_event("click", newRegionValidated)
        validateSkopeBtn.on_event("click", updateSkopeRules)

        self._aeVS.getFigureWidget().data[0].on_selection(lassoSelection)
        self._aeES.getFigureWidget().data[0].on_selection(lassoSelection)

        def figureSizeChanged(*args):  # 2121
            with self._aeVS.getFigureWidget().batch_update():
                self._aeVS.getFigureWidget().layout.width = int(
                    figureSizeSlider.v_model
                )
            with self._aeES.getFigureWidget().batch_update():
                self._aeES.getFigureWidget().layout.width = int(
                    figureSizeSlider.v_model
                )
            for i in range(len(allHistograms)):
                with allHistograms[i].batch_update():
                    allHistograms[i].layout.width = 0.9 * int(figureSizeSlider.v_model)
                with allBeeSwarms[i].batch_update():
                    allBeeSwarms[i].layout.width = 0.9 * int(figureSizeSlider.v_model)

        figureSizeSlider.on_event("input", figureSizeChanged)

        addSkopeBtn = v.Btn(
            class_="ma-4 pa-2 mb-1",
            children=[v.Icon(children=["mdi-plus"]), "Add a rule"],
        )

        addAnotherFeatureWgt = v.Select(
            class_="mr-3 mb-0",
            explanationsMenuDict=["/"],
            v_model="/",
            style_="max-width : 15%",
        )

        addMapBtn = v.Btn(
            class_="ma-4 pa-2 mb-1",
            children=[v.Icon(class_="mr-4", children=["mdi-map"]), "Display the map"],
            color="white",
            disabled=True,
        )

        def displayMap(widget, event, data):
            if widget.color == "white":
                mapPartLayout.class_ = "d-flex justify-space-around ma-0 pa-0"
                widget.color = "error"
                widget.children = [widget.children[0]] + ["Hide the map"]
                _save_lat_rule = [
                    self._selection.getVSRules()[i]
                    for i in range(len(self._selection.getVSRules()))
                    if self._selection.getVSRules()[i][2] == self._ds.getLatLon([0])
                ]
                _save_long_rule = [
                    self._selection.getVSRules()[i]
                    for i in range(len(self._selection.getVSRules()))
                    if self._selection.getVSRules()[i][2] == self._ds.getLatLon([1])
                ]
                count = 0
                for i in range(len(self._selection.getVSRules())):
                    if (
                        self._selection.getVSRules()[i - count][2]
                        == self.atk.dataset.lat
                        or self._selection.getVSRules()[i - count][2]
                        == self.atk.dataset.long
                    ):
                        self._selection.getVSRules().pop(i - count)
                        count += 1
                for i in range(len(skopeAccordion.children)):
                    if skopeAccordion.children[i].children[0].children[0].children[0][
                        4:-1
                    ] in [self._ds.getLatLon([0]), self._ds.getLatLon([1])]:
                        skopeAccordion.children[i].disabled = True
                ourVSSkopeCard.children = gui_utils.createRuleCard(
                    self._selection.ruleListToStr()
                )
                update_graph_with_rules()
            else:
                self.selection.setIndexesFromMap(None)
                widget.color = "white"
                mapPartLayout.class_ = "d-none ma-0 pa-0"
                widget.children = [widget.children[0]] + ["Display the map"]
                self._selection.setVSRules(
                    self._selection.getVSRules() + _save_lat_rule + _save_long_rule
                )
                ourVSSkopeCard.children = gui_utils.createRuleCard(
                    self._selection.ruleListToStr()
                )
                update_graph_with_rules()
                for i in range(len(skopeAccordion.children)):
                    skopeAccordion.children[i].disabled = False

        addButtonsGrp = v.Row(
            children=[addSkopeBtn, addAnotherFeatureWgt, v.Spacer(), addMapBtn]
        )

        # function called when we add a feature to the rules. We instanciate the exact same things than from the previous features
        # (beeswarms, histograms, etc...)
        def addSkopeRule(*b):
            new_rule = [0] * 5
            column = addAnotherFeatureWgt.v_model
            if self._otherColumns is None:
                return
            self._otherColumns = [a for a in self._otherColumns if a != column]
            new_rule[2] = column
            new_rule[0] = round(
                min(list(self._ds.getFullValues(Dataset.REGULAR)[column].values)), 1
            )
            new_rule[1] = "<="
            new_rule[3] = "<="
            new_rule[4] = round(
                max(list(self._ds.getFullValues(Dataset.REGULAR)[column].values)), 1
            )
            self._selection.getVSRules().append(new_rule)
            ourVSSkopeCard.children = gui_utils.createRuleCard(
                self._selection.ruleListToStr()
            )

            (
                newValidateChange,
                newSkopeSlider,
                newHistogram,
            ) = gui_utils.createNewFeatureRuleGUI(
                self, new_rule, column, self._histogramBinNum, figureSizeSlider.v_model
            )

            allHistograms.append(newHistogram)

            # TODO to understand
            def newFunctionChangeValidate(*change):  # 2211
                ii = -1
                for i in range(
                    len(self._selection.getVSRules())
                ):  # Why only VS rules ?
                    if self._selection.getVSRules()[i][2] == column_2:
                        ii = int(i)
                a = deepcopy(float(newSkopeSlider.v_model[0]))
                b = deepcopy(float(newSkopeSlider.v_model[1]))
                self._selection.getVSRules()[ii][0] = a
                self._selection.getVSRules()[ii][4] = b
                self._selection.getVSRules()[ii][0] = a
                self._selection.getVSRules()[ii][4] = b
                ourVSSkopeCard.children = gui_utils.createRuleCard(
                    self._selection.ruleListToStr()
                )
                update_graph_with_rules()
                updateSubModelsScores(None)

            newValidateChange.on_event("click", newFunctionChangeValidate)

            newRealTimeGraphCheck = v.Checkbox(
                v_model=False, label="Real-time updates on the graphs", class_="ma-3"
            )

            newTextAndSliderGrp = v.Layout(
                children=[
                    v.TextField(
                        style_="max-width:100px",
                        v_model=newSkopeSlider.v_model[0],
                        hide_details=True,
                        type="number",
                        density="compact",
                    ),
                    newSkopeSlider,
                    v.TextField(
                        style_="max-width:100px",
                        v_model=newSkopeSlider.v_model[1],
                        hide_details=True,
                        type="number",
                        density="compact",
                    ),
                ],
            )

            def validateNewUpdate(*args):
                if newRealTimeGraphCheck.v_model:
                    newValidateChange.disabled = True
                else:
                    newValidateChange.disabled = False

            newRealTimeGraphCheck.on_event("change", validateNewUpdate)

            newToBeDefinedGrp = widgets.HBox([newValidateChange, newRealTimeGraphCheck])

            allnewWidgetsColumn = widgets.VBox(
                [newTextAndSliderGrp, newHistogram, newToBeDefinedGrp]
            )

            column_shap = column + "_shap"
            y_histo_shap = [0] * len(self._xds.getFullValues())
            new_beeswarm = go.FigureWidget(
                data=[
                    go.Scatter(
                        x=self._xds.getFullValues(self._explanationES[0])[column_shap],
                        y=y_histo_shap,
                        mode="markers",
                    )
                ]
            )
            new_beeswarm.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=200,
                width=0.9 * int(figureSizeSlider.v_model),
            )
            new_beeswarm.update_yaxes(visible=False, showticklabels=False)
            [new_y, marker] = createBeeswarm(self, _explanationES[0], column)
            new_beeswarm.data[0].y = new_y
            new_beeswarm.data[0].x = self._xds.getFullValues(self._explanationES[0])[
                column_shap
            ]
            new_beeswarm.data[0].marker = marker

            newBeeswarmColorChosen = v.Row(
                class_="pt-3 mt-0 ml-4",
                children=[
                    "Value of Xi",
                    v.Switch(
                        class_="ml-3 mr-2 mt-0 pt-0",
                        v_model=False,
                        label="",
                    ),
                    "Current selection",
                ],
            )

            def noBeeswarmColorChosen(*args):
                if newBeeswarmColorChosen.children[1].v_model is False:
                    marker = createBeeswarm(
                        self,
                        _explanationES[0],
                        self._selection.getVSRules()[
                            len(self._selection.getVSRules()) - 1
                        ][2],
                    )[1]
                    new_beeswarm.data[0].marker = marker
                    new_beeswarm.update_traces(marker=dict(showscale=True))
                else:
                    update_histograms_with_rules(
                        newSkopeSlider.v_model[0],
                        newSkopeSlider.v_model[1],
                        0,
                    )
                    new_beeswarm.update_traces(marker=dict(showscale=False))

            newBeeswarmColorChosen.children[1].on_event("change", noBeeswarmColorChosen)

            new_beeswarm_tot = widgets.VBox([newBeeswarmColorChosen, new_beeswarm])
            new_beeswarm_tot.layout.margin = "0px 0px 0px 20px"

            allBeeSwarms_total.append(new_beeswarm_tot)

            if not beeSwarmCheck.v_model:
                new_beeswarm_tot.layout.display = "none"

            allBeeSwarms.append(new_beeswarm)

            allBeeSwarmsColorChosers.append(newBeeswarmColorChosen)

            addAnotherFeatureWgt.explanationsMenuDict = self._otherColumns
            addAnotherFeatureWgt.v_model = self._otherColumns[0]

            newSkopeDeleteBtn = v.Btn(
                color="error",
                class_="ma-2 ml-4 pa-1",
                elevation="3",
                icon=True,
                children=[v.Icon(children=["mdi-delete"])],
            )

            def deleteNewSkopex(*b):
                column_2 = newSkopeSlider.label
                ii = 0
                for i in range(len(self._selection.getVSRules())):
                    if self._selection.getVSRules()[i][2] == column_2:
                        ii = i
                        break
                allBeeSwarms_total.pop(ii)
                allHistograms.pop(ii)
                self._selection.getVSRules().pop(ii)
                allBeeSwarms.pop(ii)
                allBeeSwarmsColorChosers.pop(ii)
                self._otherColumns = [column_2] + self._otherColumns
                ourVSSkopeCard.children = gui_utils.createRuleCard(
                    self._selection.ruleListToStr()
                )
                addAnotherFeatureWgt.explanationsMenuDict = self._otherColumns
                addAnotherFeatureWgt.v_model = self._otherColumns[0]
                skopeAccordion.children = [
                    a for a in skopeAccordion.children if a != newFeatureAccordion_n
                ]
                for i in range(
                    ii,
                    len(
                        [
                            skopeAccordion.children[a]
                            for a in range(len(skopeAccordion.children))
                            if skopeAccordion.children[a].disabled is False
                        ]
                    ),
                ):
                    col = (
                        "X"
                        + str(i + 1)
                        + " ("
                        + self._selection.getVSRules()[i][2]
                        + ")"
                    )
                    skopeAccordion.children[i].children[0].children[0].children = [col]

                if addAnotherFeatureWgt.v_model in [
                    self._ds.getLatLon()[0],
                    self._ds.getLatLon()[1],
                ]:
                    if self._ds.getLatLon()[0] in [
                        self._selection.getVSRules()[i][2]
                        for i in range(len(self._selection.getVSRules()))
                    ] and self._ds.getLatLon()[1] in [
                        self._selection.getVSRules()[i][2]
                        for i in range(len(self._selection.getVSRules()))
                    ]:
                        addMapBtn.disabled = False
                    else:
                        addMapBtn.disabled = True
                update_graph_with_rules()

            newSkopeDeleteBtn.on_event("click", deleteNewSkopex)

            newIsContinuousCheck = v.Checkbox(
                v_model=True, label="is continuous?"
            )  # 2367

            newRightSideColumn = v.Col(
                children=[newSkopeDeleteBtn, newIsContinuousCheck],
                class_="d-flex flex-column align-center justify-center",
            )

            newFeatureGrp = gui_utils.create_class_selector(
                self,
                self._selection.getVSRules()[-1][2],
                min=min(
                    list(
                        self._ds.getFullValues(Dataset.REGULAR)[
                            newSkopeSlider.label
                        ].values
                    )
                ),
                max=max(
                    list(
                        self._ds.getFullValues(Dataset.REGULAR)[
                            newSkopeSlider.label
                        ].values
                    )
                ),
                fig_size=figureSizeSlider.v_model,
            )  # 2371

            def newContinuousChange(widget, event, data):
                column_2 = newSkopeSlider.label
                index = 0
                for i in range(len(self._selection.getVSRules())):
                    if self._selection.getVSRules()[i][2] == column_2:
                        index = i
                        break
                if widget.v_model is True and widget == newRightSideColumn.children[1]:
                    newFeatureAccordion.children = [allnewWidgetsColumn] + list(
                        newFeatureAccordion.children[1:]
                    )
                    count = 0
                    for i in range(len(self._selection.getVSRules())):
                        if (
                            self._selection.getVSRules()[i - count][2]
                            == self._selection.getVSRules()[index][2]
                            and i - count != index
                        ):
                            self._selection.getVSRules().pop(i - count)
                            count += 1
                    self._selection.getVSRules()[index][0] = newSkopeSlider.v_model[0]
                    self._selection.getVSRules()[index][4] = newSkopeSlider.v_model[1]
                    ourVSSkopeCard.children = gui_utils.createRuleCard(
                        self._selection.ruleListToStr()
                    )
                    update_graph_with_rules()
                else:
                    newFeatureAccordion.children = [newFeatureGrp] + list(
                        newFeatureAccordion.children[1:]
                    )
                    l = []
                    for i in range(len(newFeatureGrp.children[2].children)):
                        if newFeatureGrp.children[2].children[i].v_model:
                            l.append(int(newFeatureGrp.children[2].children[i].label))
                    if len(l) == 0:
                        widget.v_model = True
                        return
                    column = deepcopy(self._selection.getVSRules()[index][2])
                    count = 0
                    for i in range(len(self._selection.getVSRules())):
                        if self._selection.getVSRules()[i - count][2] == column:
                            self._selection.getVSRules().pop(i - count)
                            count += 1
                    croissant = 0
                    for ele in l:
                        self._selection.getVSRules().insert(
                            index + croissant,
                            [ele - 0.5, "<=", column, "<=", ele + 0.5],
                        )
                        croissant += 1
                    ourVSSkopeCard.children = gui_utils.createRuleCard(
                        self._selection.ruleListToStr()
                    )
                    update_graph_with_rules()

            newRightSideColumn.children[1].on_event("change", newContinuousChange)

            for ii in range(len(newFeatureGrp.children[2].children)):
                newFeatureGrp.children[2].children[ii].on_event(
                    "change", newContinuousChange
                )

            newFeatureAccordion = widgets.HBox(
                [allnewWidgetsColumn, new_beeswarm_tot, newRightSideColumn],
                layout=Layout(align_explanationsMenuDict="center"),
            )

            newFeatureAccordion_n = v.ExpansionPanels(
                class_="ma-2 mb-1",
                children=[
                    v.ExpansionPanel(
                        children=[
                            v.ExpansionPanelHeader(children=["Xn"]),
                            v.ExpansionPanelContent(children=[newFeatureAccordion]),
                        ]
                    )
                ],
            )

            skopeAccordion.children = [*skopeAccordion.children, newFeatureAccordion_n]
            name_colcol = "X" + str(len(skopeAccordion.children)) + " (" + column + ")"
            skopeAccordion.children[-1].children[0].children[0].children = name_colcol

            with newHistogram.batch_update():
                new_list = [
                    g
                    for g in list(
                        self._ds.getFullValues(Dataset.REGULAR)[column].values
                    )
                    if g >= newSkopeSlider.v_model[0] and g <= newSkopeSlider.v_model[1]
                ]
                newHistogram.data[1].x = new_list

                column_2 = newSkopeSlider.label
                new_list_rule = (
                    self._ds.getFullValues(Dataset.REGULAR)
                    .index[
                        self._ds.getFullValues(Dataset.REGULAR)[column_2].between(
                            newSkopeSlider.v_model[0],
                            newSkopeSlider.v_model[1],
                        )
                    ]
                    .tolist()
                )
                new_list_tout = new_list_rule.copy()
                for i in range(1, len(self._selection.getVSRules())):
                    new_list_temp = (
                        self._ds.getFullValues(Dataset.REGULAR)
                        .index[
                            self._ds.getFullValues(Dataset.REGULAR)[
                                self._selection.getVSRules()[i][2]
                            ].between(
                                self._selection.getVSRules()[i][0],
                                self._selection.getVSRules()[i][4],
                            )
                        ]
                        .tolist()
                    )
                    new_list_tout = [g for g in new_list_tout if g in new_list_temp]
                new_list_tout_new = self._ds.getXValues()[column_2][new_list_tout]
                newHistogram.data[2].x = new_list_tout_new

            def newSkopeValueChange(*b1):  # 2466
                newTextAndSliderGrp.children[0].v_model = newSkopeSlider.v_model[0]
                newTextAndSliderGrp.children[2].v_model = newSkopeSlider.v_model[1]
                column_2 = newSkopeSlider.label
                ii = 0
                for i in range(len(self._selection.getVSRules())):
                    if self._selection.getVSRules()[i][2] == column_2:
                        ii = i
                        break
                new_list = [
                    g
                    for g in list(self._ds.getXValues()[column_2].values)
                    if g >= newSkopeSlider.v_model[0] and g <= newSkopeSlider.v_model[1]
                ]
                with newHistogram.batch_update():
                    newHistogram.data[1].x = new_list
                if activate_histograms:
                    update_histograms_with_rules(
                        newSkopeSlider.v_model[0],
                        newSkopeSlider.v_model[1],
                        ii,
                    )
                if newRealTimeGraphCheck.v_model:
                    self._selection.getVSRules()[ii - 1][0] = float(
                        deepcopy(newSkopeSlider.v_model[0])
                    )
                    self._selection.getVSRules()[ii - 1][4] = float(
                        deepcopy(newSkopeSlider.v_model[1])
                    )
                    ourVSSkopeCard.children = gui_utils.createRuleCard(
                        self._selection.ruleListToStr()
                    )
                    update_graph_with_rules()

            newSkopeSlider.on_event("input", newSkopeValueChange)

            if newSkopeSlider.label in [
                self._xds.getLatLon([0]),
                self._xds.getLatLon()[0],
            ]:
                if self._xds.getLatLon()[0] in [
                    self._selection.getVSRules()[i][2]
                    for i in range(len(self._selection.getVSRules()))
                ] and self._xds.getLatLon()[1] in [
                    self._selection.getVSRules()[i][2]
                    for i in range(len(self._selection.getVSRules()))
                ]:
                    addMapBtn.disabled = False
                else:
                    addMapBtn.disabled = True

        # End addSkopeRule

        newRegionValidated()

        addSkopeBtn.on_event("click", addSkopeRule)

        dimReducVSSettingsMenu = gui_utils.createSettingsMenu(
            self._aeVS.getProjectionSliders(),
            "Settings of the projection in the Values Space",
        )
        dimReducESSettingsMenu = gui_utils.createSettingsMenu(
            self._aeES.getProjectionSliders(),
            "Settings of the projection in the Explanations Space",
        )

        widgets.HBox(
            [
                self._aeVS.getProjectionSelect(),
                v.Layout(children=[dimReducVSSettingsMenu]),
                busyVSHBox,
            ]
        )
        widgets.HBox(
            [
                self._aeES.getProjectionSelect(),
                v.Layout(children=[dimReducESSettingsMenu]),
                busyESHBox,
            ]
        )

        resetOpacityBtn = v.Btn(
            icon=True,
            children=[v.Icon(children=["mdi-opacity"])],
            class_="ma-2 ml-6 pa-3",
            elevation="3",
        )

        resetOpacityBtn.children = [
            gui_utils.wrap_in_a_tooltip(
                resetOpacityBtn.children[0],
                "Reset the opacity of the points",
            )
        ]

        def resetOpacity(*args):
            with self._leftVSFigure.batch_update():
                self._leftVSFigure.data[0].marker.opacity = 1
            with self._rightESFigure.batch_update():
                self._rightESFigure.data[0].marker.opacity = 1

        resetOpacityBtn.on_event("click", resetOpacity)

        self.update_explanation_select()

        def changeExplanationMethod(widget, event, data):
            # _explanationSelect has been changed

            oldExplanationMethod, oldOrigin = self._explanationES
            newExplanationMethod, newOrigin = None, None

            match data:
                case "SHAP (imported)":
                    newExplanationMethod = ExplanationMethod.SHAP
                    newOrigin = ExplanationDataset.IMPORTED
                case "SHAP (computed)":
                    newExplanationMethod = ExplanationMethod.SHAP
                    newOrigin = ExplanationDataset.COMPUTED
                case "LIME (imported)":
                    newExplanationMethod = ExplanationMethod.LIME
                    newOrigin = ExplanationDataset.IMPORTED
                case "LIME (computed)":
                    newExplanationMethod = ExplanationMethod.LIME
                    newOrigin = ExplanationDataset.COMPUTED

            self._explanationES = (newExplanationMethod, newOrigin)

            self.check_explanation()
            self.redraw_graph(GUI.ES)

        self._aeES.getExplanationSelect().on_event("change", changeExplanationMethod)

        def explanationComputationRequested(widget, event, data):
            logger.debug(
                f"explanationComputationRequested : computation asked for {widget.v_model}"
            )

            # # We set the new value for self._explanationES
            # if widget.v_model == "SHAP":
            #     self._explanationES = (ExplanationMethod.SHAP, ExplanationDataset.COMPUTED)
            # else :
            #     self._explanationES = (ExplanationMethod.LIME, ExplanationDataset.COMPUTED)

            # self.check_explanation()
            # self.update_explanation_select()
            # self.redraw_graph(GUI.ES)

        # TO be connected !!!
        # ourSHAPProgLinearColumn.children[-1].on_event("click", explanationComputationRequested)
        # ourLIMEProgLinearColumn.children[-1].on_event("click", explanationComputationRequested)

        topButtonsHBox = widgets.HBox(
            [
                dimSwitchRow,
                v.Layout(
                    class_="pa-2 ma-2",
                    elevation="3",
                    children=[
                        gui_utils.wrap_in_a_tooltip(
                            v.Icon(
                                children=["mdi-format-color-fill"], class_="mt-n5 mr-4"
                            ),
                            "Color of the points",
                        ),
                        colorChoiceBtnToggle,
                        resetOpacityBtn,
                        self._aeES.getExplanationSelect()
                        # computeMenu,
                    ],
                ),
                v.Layout(
                    class_="mt-3",
                    children=[
                        self._aeVS.getProjectionSelect(),
                        self._aeES.getProjectionSelect(),
                    ],
                ),
            ],
            layout=Layout(
                width="100%",
                display="flex",
                flex_flow="row",
                justify_content="space-around",
            ),
        )

        widgets.VBox(
            [self._aeVS.getFigureWidget(), self._aeES.getFigureWidget()],
            layout=Layout(width="100%"),
        )

        beeSwarmCheck = v.Checkbox(
            v_model=True,
            label="Show Shapley's beeswarm plots",
            class_="ma-1 mr-3",
        )

        def beeSwarmCheckChange(*b):
            if not beeSwarmCheck.v_model:
                for i in range(len(allBeeSwarms_total)):
                    allBeeSwarms_total[i].layout.display = "none"
            else:
                for i in range(len(allBeeSwarms_total)):
                    allBeeSwarms_total[i].layout.display = "block"

        beeSwarmCheck.on_event("change", beeSwarmCheckChange)

        skopeBtns = v.Layout(
            class_="d-flex flex-row",
            children=[
                validateSkopeBtn,
                reinitSkopeBtn,
                v.Spacer(),
                beeSwarmCheck,
            ],
        )

        skopeBtnsGrp = widgets.VBox([skopeBtns, skopeText])

        bouton_magic = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(children=["mdi-creation"], class_="mr-3"),
                "Magic button",
            ],
        )

        magicGUI = v.Layout(
            class_="d-flex flex-row justify-center align-center",
            children=[
                v.Spacer(),
                bouton_magic,
                v.Checkbox(v_model=True, label="Demonstration mode", class_="ma-4"),
                v.TextField(
                    class_="shrink",
                    type="number",
                    label="Time between the steps (ds)",
                    v_model=10,
                ),
                v.Spacer(),
            ],
        )

        def magicCheckChange(*args):
            if magicGUI.children[2].v_model:
                magicGUI.children[3].disabled = False
            else:
                magicGUI.children[3].disabled = True

        magicGUI.children[2].on_event("change", magicCheckChange)

        def find_best_score():
            a = 1000
            for i in range(len(self._subModelsScores)):
                score = self._subModelsScores[i][0]
                if score < a:
                    a = score
                    index = i
            return index

        def magicClustering(*args):
            demo = magicGUI.children[2].v_model
            if demo is False:
                antakiaMethodCard.children[0].v_model = 3
            N_antakiaStepsCard = dynamicClustering(None)
            if demo:
                tempo = int(magicGUI.children[3].v_model) / 10
                if tempo < 0:
                    tempo = 0
            else:
                tempo = 0
            time.sleep(tempo)
            for i in range(N_antakiaStepsCard):
                tabOneSelectionColumn.children[-1].children[0].children[
                    0
                ].v_model = str(i)
                selectedClusterChange(None, None, None)
                time.sleep(tempo)
                if demo:
                    antakiaMethodCard.children[0].v_model = 1
                time.sleep(tempo)
                updateSkopeRules(None)
                time.sleep(tempo)
                if demo:
                    antakiaMethodCard.children[0].v_model = 2
                time.sleep(tempo)
                index = find_best_score()
                subModelslides.children[index].children[0].color = "blue lighten-4"
                select_submodel(None, None, None, False)
                time.sleep(tempo)
                subModelslides.children[index].children[0].color = "white"
                if demo:
                    antakiaMethodCard.children[0].v_model = 3
                time.sleep(tempo)
                newRegionValidated(None)
                time.sleep(tempo)
                if i != N_antakiaStepsCard - 1:
                    if demo:
                        antakiaMethodCard.children[0].v_model = 0
                    time.sleep(tempo)
            colorChoiceBtnToggle.v_model = "Regions"
            changeMarkersColor(None)

        #map plotly
        mapWIdget = go.FigureWidget(
            data=go.Scatter(x=[1], y=[1], mode="markers", marker=ourVSMarkerDict, customdata=ourVSMarkerDict["color"], hovertemplate = '%{customdata:.3f}')
        )

        mapWIdget.update_layout(dragmode="lasso")

        # instanciate the map, with latitude and longitude
        if self._ds.getLatLon() is not None:
            df = self._ds.getFullValues()
            data=go.Scattergeo(
                lat = df[self._ds.getLatLon()[0]],
                lon = df[self._ds.getLatLon()[1]],
                mode = 'markers',
                marker_color = self._ds.getYValues(),
                )
            mapWIdget = go.FigureWidget(
            data=data
            )
            lat_center = max(df[self._ds.getLatLon()[0]]) - (max(df[self._ds.getLatLon()[0]]) - min(df[self._ds.getLatLon()[0]]))/2
            long_center = max(df[self._ds.getLatLon()[1]]) - (max(df[self._ds.getLatLon()[1]]) - min(df[self._ds.getLatLon()[1]]))/2
            mapWIdget.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                #geo_scope="world",
                height=300,
                width=900,
                geo=dict(
                    center=dict(
                        lat=lat_center,
                        lon=long_center
                    ),
                    projection_scale=5,
                    showland = True,
                )
            )

        mapSelectionTxt = v.Card(
            style_="width: 30%",
            class_="ma-5",
            children=[
                v.CardTitle(children=["Selection on the map"]),
                v.CardText(
                    children=[
                        v.Html(
                            tag="div",
                            children=["No selection"],
                        )
                    ]
                ),
            ],
        )

        def changeMapText(trace, points, selector):
            mapSelectionTxt.children[1].children[0].children = ["Number of entries selected: " + str(len(points.point_inds))]
            self._selection._mapIndexes = points.point_inds
            ourVSSkopeCard.children = gui_utils.createRuleCard(self._selection.ruleListToStr())
            update_graph_with_rules()

        mapWIdget.data[0].on_selection(changeMapText)

        mapPartLayout = v.Layout(
            class_="d-none ma-0 pa-0",
            children=[mapWIdget, mapSelectionTxt]
        )

        bouton_magic.on_event("click", magicClustering)

        loadingClustersProgLinear = v.ProgressLinear(
            indeterminate=True, class_="ma-3", style_="width : 100%"
        )

        loadingClustersProgLinear.class_ = "d-none"

        tabOneSelectionColumn = v.Col(
            children=[
                selectionCard,
                out_accordion,
                clusterGrp,
                loadingClustersProgLinear,
                clusterResults,
            ]
        )

        loadingModelsProgLinear = v.ProgressLinear(
            indeterminate=True,
            class_="my-0 mx-15",
            style_="width: 100%;",
            color="primary",
            height="5",
        )

        loadingModelsProgLinear.class_ = "d-none"

        tabTwoSkopeRulesColumn = v.Col(
            children=[skopeBtnsGrp, skopeAccordion, addButtonsGrp]
        )
        tabThreeSubstitutionVBox = widgets.VBox(
            [loadingModelsProgLinear, subModelslides]
        )
        # allRegionUI = widgets.VBox([self._selection, self._regionsTable])
        tabFourRegionListVBox = loadingModelsProgLinear # Just want to go further

        antakiaMethodCard = v.Card(
            class_="w-100 pa-3 ma-3",
            elevation="3",
            children=[
                v.Tabs(
                    class_="w-100",
                    v_model="tabs",
                    children=[
                        v.Tab(value="one", children=["1. Current selection"]),
                        v.Tab(value="two", children=["2. Selection adjustment"]),
                        v.Tab(value="three", children=["3. Choice of the sub-model"]),
                        v.Tab(value="four", children=["4. Overview of the Regions"]),
                    ],
                ),
                v.CardText(
                    class_="w-100",
                    children=[
                        v.Window(
                            class_="w-100",
                            v_model="tabs",
                            children=[
                                v.WindowItem(value=0, children=[tabOneSelectionColumn]),
                                v.WindowItem(
                                    value=1, children=[tabTwoSkopeRulesColumn]
                                ),
                                v.WindowItem(
                                    value=2, children=[tabThreeSubstitutionVBox]
                                ),
                                v.WindowItem(value=3, children=[tabFourRegionListVBox]),
                            ],
                        )
                    ],
                ),
            ],
        )

        widgets.jslink(
            (antakiaMethodCard.children[0], "v_model"),
            (antakiaMethodCard.children[1].children[0], "v_model"),
        )

        ourGUIVBox = widgets.VBox(
            [
                menuAppBar,
                backupsDialog,
                topButtonsHBox,
                self._aeVS.getFigureWidget(),
                self._aeES.getFigureWidget(),
                antakiaMethodCard,
                magicGUI,
            ],
            layout=Layout(width="100%"),
        )
        with self._out:
            display(ourGUIVBox)

    def results(self, number: int = None, item: str = None):
        L_f = []
        if len(self._regions) == 0:
            return "No region has been created !"
        for i in range(len(self._regions)):
            dictio = dict()
            dictio["X"] = (
                self._ds.getXValues()
                .iloc[self._regions[i].getIndexes(), :]
                .reset_index(drop=True)
            )
            dictio["y"] = (
                self._ds.getYValues()
                .iloc[self._regions[i].getIndexes()]
                .reset_index(drop=True)
            )
            dictio["indexs"] = self._regions[i].getIndexes()
            dictio["explain"] = {"Imported": None, "SHAP": None, "LIME": None}

            if (
                self._xds.isExplanationAvailable(ExplanationMethod.SHAP)[1]
                == ExplanationDataset.IMPORTED
            ) or (
                self._xds.isExplanationAvailable(ExplanationMethod.SHAP)[1]
                == ExplanationDataset.BOTH
            ):
                # We have imported SHAP
                dictio["explain"]["Imported"] = (
                    self._xds.getFullValues(ExplanationMethod.SHAP, onlyImported=True)
                    .iloc[self._regions[i].getIndexes(), :]
                    .reset_index(idrop=True)
                )

            if (
                self._xds.isExplanationAvailable(ExplanationMethod.LIME)[1]
                == ExplanationDataset.IMPORTED
            ) or (
                self._xds.isExplanationAvailable(ExplanationMethod.LIME)[1]
                == ExplanationDataset.BOTH
            ):
                # We have imported LIME
                dictio["explain"]["Imported"] = (
                    self._xds.getFullValues(ExplanationMethod.LIME, onlyImported=True)
                    .iloc[self._regions[i].getIndexes(), :]
                    .reset_index(drop=True)
                )

            # Now we're talking of computed explanations
            tempTuple = self._xds.isExplanationAvailable(ExplanationMethod.SHAP)
            if tempTuple[0] and (
                tempTuple[1] == ExplanationDataset.IMPORTED
                or tempTuple[1] == ExplanationDataset.BOTH
            ):
                dictio["explain"]["SHAP"] = (
                    self._xds.getFullValues(ExplanationMethod.SHAP)
                    .iloc[self._regions[i].getIndexes(), :]
                    .reset_index(drop=True)
                )
            tempTuple = self._xds.isExplanationAvailable(ExplanationMethod.LIME)
            if tempTuple[0] and (
                tempTuple[1] is ExplanationDataset.IMPORTED or ExplanationDataset.BOTH
            ):
                dictio["explain"]["LIME"] = (
                    self._xds.getFullValues(ExplanationMethod.LIME)
                    .iloc[self._regions[i].getIndexes(), :]
                    .reset_index(drop=True)
                )

            if self._regions[i].sub_model is None:
                dictio["model name"] = None
                dictio["model score"] = None
                dictio["model"] = None
            else:
                dictio["model name"] = self._regions[i].sub_model["name"].__name__
                dictio["model score"] = self._regions[i].sub_model["score"]
                dictio["model"] = self._regions[i].sub_model["name"]
            dictio["rules"] = self._regions[i].rules
            L_f.append(dictio)
        if number is None or item is None:
            return L_f
        else:
            return L_f[number][item]
