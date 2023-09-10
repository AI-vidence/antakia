# Datascience imports
import pandas as pd
import numpy as np
from skrules import SkopeRules
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# GUI related imports
# import ipywidgets as widgets
from ipywidgets import widgets, Layout
from IPython.display import display, clear_output, HTML
import ipyvuetify as v
import plotly.graph_objects as go
import seaborn as sns

# Others imports
import time
from copy import deepcopy
from importlib.resources import files
from typing import Tuple

# Warnings imports
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore")

# Internal imports
from antakia.antakia import AntakIA
from antakia.utils import _function_models as function_models
from antakia.utils import _conflict_handler as conflict_handler
from antakia.potato import Potato
from antakia.compute import DimReducMethod, ExplainationMethod, SHAPExplaination, LIMExplaination, computeProjections
from antakia.data import Dataset, ExplanationsDataset
from antakia.gui_elements import addTooltip, createLinearProgressBar, createGlobalProgressBar, createMenuBar, colorChooser, SliderParam


class GUI():
    """
    GUI class.

    A GUI instance contains all the data and variables needed to run the interface.
    The interface is built using ipyvuetify and plotly.
    It heavily relies on the IPyWidgets framework.

    Instance Attributes
    ---------------------
    __atk : AntakIA  object
        Parent reference to access data, model, explanations etc.
    __selection : a Potato object
        The `Potato` object containing the current selection.
    __projectionVS : VS current projection as a Tuple (eg. TSNE, 2)
    __projectionES : ES current projection as a Tuple (eg. UMAP, 3)
    __explanationES : current explanation method
    __explanationDataset : __atk's dataset's explanationdataset
    __leftVSFigure2D : the left figure in the VS
    __rightESFigure2D : the right figure in the ES
    __leftVSFigure3D : the left figure in the VS
    __rightESFigure3D : the right figure in the ES

    __calculus : if True, we need to calculate the explanations (SHAP and LIME)
    
    __save_rules useful to keep the initial rules from the skope-rules, in order to be able to reset the rules
    __other_columns : to keep track of the columns that are not used in the rules !
    
    __activate_histograms : to know if the histograms are activated or not (bug ipywidgets !). If they are activated, we have to update the histograms.
    
    
    __result_dyadic_clustering : to keep track  of the entire results from the dyadic-clustering
    __autoClusterRegionColors: the color of the regions created by the automatic dyadic clustering
    __labels_automatic_clustering :  to keep track of the labels from the automatic-clustering, used for the colors !

    __model_index : to know which sub_model is selected by the user. 
    __score_sub_models : to keep track of the scores of the sub-models

    __backupTable : to manipulate the table of the backups

    """
    
    def __init__(self, atk : AntakIA, defaultProjection: Tuple = (DimReducMethod.PacMAP,DimReducMethod.DIM_TWO)):
        """
        GUI Class constructor.

        Parameters
        ----------
        atk : AntakIA object
            Parent object
        projection : int
            The default projection to use. See constants in DimensionalityReduction class
        """
        
        self.__atk = atk
        self.__selection = Potato(self.__atk, [], Potato.SELECTION) # Upon creation of the GUI there is no selection ?
        self.__explanationDataset = atk.getDataset().getExplanations()

        if self.__explanationDataset.isUserProvided(ExplainationMethod.SHAP) :
            self.__explanationES = ExplainationMethod.SHAP
        elif self.__explanationDataset.isUserProvided(ExplainationMethod.LIME) :
            self.__explanationES = ExplainationMethod.LIME
        elif self.__explanationDataset.isUserProvided(ExplainationMethod.OTHER) :
            self.__explanationES = ExplainationMethod.OTHER
        else : 
            self.__explanationES = ExplainationMethod.NONE # That is we need to compute it

        
        if  not DimReducMethod.isValidDimReducType(defaultProjection) :
            raise ValueError("Invalid projection type")
        
        self.__projectionVS = defaultProjection 
        self.__projectionES = defaultProjection 

        self.__leftVSFigure2D = None
        self.__rightESFigure2D = None
        self.__leftVSFigure3D = None
        self.__rightESFigure3D = None

        self.__save_rules = None #useful to keep the initial rules from the skope-rules, in order to be able to reset the rules
        self.__other_columns = None #to keep track of the columns that are not used in the rules !
        self.__activate_histograms = False #to know if the histograms are activated or not (bug ipywidgets !). If they are activated, we have to update the histograms.


        self.__autoClusterRegionColors = [] # the color of the regions created by the automatic dyadic clustering
        self.__labels_automatic_clustering = None #to keep track of the labels from the automatic-clustering, used for the colors !
        self.__result_dyadic_clustering = None #to keep track  of the entire results from the dyadic-clustering

        self.__score_sub_models = None #to keep track of the scores of the sub-models
        self.__model_index = None #to know which sub_model is selected by the user.

        self.__backupTable = None #to manipulate the table of the saves

    def getSelection(self):
        """Function that returns the current selection.

        Returns
        -------
        Potato object
            The current selection.
        """
        return self.__selection

    def __repr__(self):
        return self.display()

    def display(self):
        """Function that renders the interface
        """
        
        # ============  SPLASH SCREEN ==================

        # AntalIA logo
        logoPath = files("antakia.assets").joinpath("logo_antakia.png")
        antakiaLogo = widgets.Image(
            value=widgets.Image._load_file_value(logoPath), 
            layout=Layout(width="230px")
        )

        # Splash screen progress bars for explanations
        explainComputationPB = gui_elements.createLinearProgressBar()

        # Splash screen progress bars for ES explanations
        projComputationPB = gui_elements.createLinearProgressBar()

        # Consolidation of progress bars and progress texts in a single HBox
        explainComputationGPB = gui_elements.createGlobalProgressBar("Computing of explanatory values", explainComputationPB)
        projComputationGPB = gui_elements.createGlobalProgressBar("Computing of dimensions reduction", projComputationPB)

        # Definition of the splash screen which includes all the elements,
        splashScreen = v.Layout(
            class_="d-flex flex-column align-center justify-center",
            children=[antakiaLogo, explainComputationGPB, projComputationGPB],
        )

        # We display the splash screen
        display(splashScreen)


        # -------------- Explanation values ------------

        # If we import the explanatory values, the progress bar of this one is at 100
        if self.__explanationES != ExplainationMethod.NONE :
            explainComputationPB.v_model = 100
            explainComputationGPB.children[2].children[0].v_model = "Explanations values were provided"
        else :
            # We neeed to compute at least one explanation method - We use AntakIA's default
            if AntakIA.DEFAULT_EXPLANATION_METHOD == ExplainationMethod.SHAP :
                
                explainer = SHAPExplaination(
                        self.__atk.getDataset().getXValues(Dataset.CURRENT),
                        self.__atk.getDataset().getXValues(Dataset.ALL),
                        self.__atk.getModel())
                
                self.__explanationDataset.setValues(ExplainationMethod.SHAP, DimReducMethod.NONE, DimReducMethod.DIM_ALL, explainer.compute())
                #TODO : who need to identiify the widget to update with pubsub
            elif AntakIA.DEFAULT_EXPLANATION_METHOD == ExplainationMethod.LIME :
                explainer = LIMExplaination(
                    self.__atk.getDataset().getXValues(Dataset.CURRENT),
                    self.__atk.getDataset().getXValues(Dataset.ALL),
                    self.__atk.getModel())
                self.__explanationDataset.setValues(ExplainationMethod.LIME, DimReducMethod.NONE, DimReducMethod.DIM_ALL, explainer.compute())
                 #TODO : who need to identiify the widget to update with pubsub 
            else :
                raise ValueError("Invalid default explanation method")
                    
        # -------------- Projection values ------------
        
        projComputationGPB.children[2].children[0].v_model = "Values space... "
        # We compute projection values for the VS with the default projection, both in 2D and 3D
        # projValues is a Tuple with both 2D and 3D pd.DataFrame
        projValues = computeProjections(self.__atk.getDataset().getXValues(Dataset.SCALED), self.__defaultProjTuple[0])
        # We store this in our dataset
        self.__atk.getDataset().setXProjValues(self.__projectionVS[0], DimReducMethod.DIM_TWO, projValues[0])
        self.__atk.getDataset().setXProjValues(self.__projectionVS[0], DimReducMethod.DIM_THREE, projValues[1])
        projComputationPB.v_model = +50 # We did half the job

        projComputationGPB.children[2].children[0].v_model = "Values space... Explanatory space..."
        # We compute projection values for the ES with the default projection, bonth in 2D and 3D
        # proValues is a Tuple with both 2D and 3D pd.DataFrame
        projValues = computeProjections(
            self.__explanationDataset.getValues(
                self.__explanationES,
                self.__projectionES[0],
                DimReducMethod.DIM_ALL),
            self.__projectionES[0])
        # We store this in our dataset               
        self.__explanationDataset.setValues(self.__explanationES, self.__projectionES[0], DimReducMethod.DIM_TWO, projValues[0])
        self.__explanationDataset.setValues(self.__explanationES, self.__projectionES[0], DimReducMethod.DIM_THREE, projValues[1])
        projComputationPB.v_model = +50 # End of the job


        # We remove the Splahs screen
        splashScreen.class_ = "d-none"

        # =============== Dim reduction (projection) management ================

        # Below two circular progess bar to tell when a projection is being computed
        circlePB = v.ProgressCircular(
            indeterminate=True, color="blue", width="6", size="35", class_="mx-4 my-3"
        )

        computeIndicatorVS = widgets.HBox([circlePB])
        computeIndicatorES = widgets.HBox([circlePB])
        computeIndicatorVS.layout.visibility = "hidden"
        computeIndicatorES.layout.visibility = "hidden"

        # ---- VS side -----

        # Dropdown allowing to choose the projection in the VS
        dimReducVSDropdown = v.Select(
            label="Projection in the VS:", # TODO : should be localized
            items=DimReducMethod.getDimReducMhdsAsStrList(),
            style_="width: 150px",
        )
        # We set it to the default projection
        dimReducVSDropdown.v_model = dimReducVSDropdown.items[self.__projectionVS[0]]

        # ---- PaCMAP -----
        # TODO create a class for those triple siders ?

        # Here the sliders of the parameters for PaCMAP projection in the VS
        neighboursPaCMAPVSSlider = gui_elements.SliderParam(v_model=10, min=5, max=30, step=1, label="Number of neighbors :") # TODO : should be localized
        mnRatioPaCMAPVSSlider = gui_elements.SliderParam(v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio :")
        fpRatioPacMAPVSSlider = gui_elements.SliderParam(v_model=2, min=0.1, max=5, step=0.1, label="FP ratio :")

        def updatePaCMAPVSSlider(widget, event, data):
            # function that updates the values ​​when there is a change of sliders in the parameters of PaCMAP for the VS
            if widget.label == "Number of neighbors :":  # TODO : should be localized
                neighboursPaCMAPVSSlider.children[1].children = [str(data)]
            elif widget.label == "MN ratio :": # TODO : should be localized
                mnRatioPaCMAPVSSlider.children[1].children = [str(data)]
            elif widget.label == "FP ratio :": # TODO : should be localized
                fpRatioPacMAPVSSlider.children[1].children = [str(data)]

        neighboursPaCMAPVSSlider.children[0].on_event(
            "input", updatePaCMAPVSSlider
        )
        mnRatioPaCMAPVSSlider.children[0].on_event(
            "input", updatePaCMAPVSSlider
        )
        fpRatioPacMAPVSSlider.children[0].on_event(
            "input", updatePaCMAPVSSlider
        )

        allPaCMAPVSSliders = widgets.VBox(
            [
                neighboursPaCMAPVSSlider,
                mnRatioPaCMAPVSSlider,
                fpRatioPacMAPVSSlider,
            ],
            layout=Layout(width="100%"),
        )

        validatePaCMAPParamsVSBtn = v.Btn(
            children=[
                v.Icon(left=True, children=["mdi-check"]),
                "Validate", # TODO : should be localized
            ]
        )

        resetPaCMAPParamsVSBtn = v.Btn(
            class_="ml-4",
            children=[
                v.Icon(left=True, children=["mdi-skip-backward"]),
                "Reset", # TODO : should be localized
            ],
        )

        allPaCMAPParamVSBtn = widgets.HBox(
            [validatePaCMAPParamsVSBtn, resetPaCMAPParamsVSBtn]
        )
        paramsProjVSGroup = widgets.VBox(
            [allPaCMAPVSSliders, allPaCMAPParamVSBtn], layout=Layout(width="100%")
        )

        # Recompute VS PaCMAP projection whenever the user changes the parameters
        def updatePaCMAPVS(*args):
            
            # We get the parameters from the sliders state
            neighbors = neighboursPaCMAPVSSlider.children[0].v_model
            MN_ratio = mnRatioPaCMAPVSSlider.children[0].v_model
            FP_ratio = fpRatioPacMAPVSSlider.children[0].v_model

            computeIndicatorVS.layout.visibility = "visible"

            # We compute :
            projValues = computeProjections(self.__atk.getDataset().getXValues(Dataset.SCALED), DimReducMethod.PaCMAP, neighbors = neighbors, MN_ratio = MN_ratio, FP_ratio =FP_ratio)
            # We store this in our dataset
            self.__atk.getDataset().setXProjValues(DimReducMethod.PaCMAP, DimReducMethod.DIM_TWO, projValues[0])
            self.__atk.getDataset().setXProjValues(DimReducMethod.PaCMAP, DimReducMethod.DIM_THREE, projValues[1])

            computeIndicatorVS.layout.visibility = "hidden"

            compute.update_figures(self, self.__explanation, self.__projectionVS, self.__projectionES)

        validatePaCMAPParamsVSBtn.on_event("click", updatePaCMAPVS)

        def resetPaCMAPVS(*args):
            computeIndicatorVS.layout.visibility = "visible"

            # We compute :
            projValues = computeProjections(self.__atk.getDataset().getXValues(Dataset.SCALED), DimReducMethod.PaCMAP)
            # We store this in our dataset
            self.__atk.getDataset().setXProjValues(DimReducMethod.PaCMAP, DimReducMethod.DIM_TWO, projValues[0])
            self.__atk.getDataset().setXProjValues(DimReducMethod.PaCMAP, DimReducMethod.DIM_THREE, projValues[1])

            computeIndicatorVS.layout.visible = "hidden"
            
            compute.update_figures(self, self.__explanation, self.__projectionVS, self.__projectionES)

        resetPaCMAPParamsVSBtn.on_event("click", resetPaCMAPVS)

        # ----  ES side -----

        # TODO : my guess is that we should make one GUI class for all the stuff present in both spaces
        # Dropdown allowing to choose the projection in the ES
        dimReducESDropdown = v.Select(
            label="Projection in the ES:",
            items=DimReducMethod.getDimReducMhdsAsStrList(),
            style_="width: 150px",
        )
        # We set it to the default projection
        dimReducESDropdown.v_model = dimReducESDropdown.items[self.__projectionES[0]]


        # Here the sliders of the parameters for PaCMAP projection in the ES
        neighboursPaCMAPESSlider = gui_elements.SliderParam(v_model=10, min=5, max=30, step=1, label="Number of neighbors :") # TODO : should be localized
        mnRatioPaCMAPESSlider = gui_elements.SliderParam(v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio :")
        fpRatioPacMAPESSlider = gui_elements.SliderParam(v_model=2, min=0.1, max=5, step=0.1, label="FP ratio :")

        def updatePaCMAPESSlider(widget, event, data):
            if widget.label == "Number of neighbors :": # TODO : should be localized
                neighboursPaCMAPESSlider.children[1].children = [str(data)]
            elif widget.label == "MN ratio :":
                mnRatioPaCMAPESSlider.children[1].children = [str(data)]
            elif widget.label == "FP ratio :":
                fpRatioPacMAPESSlider.children[1].children = [str(data)]

        neighboursPaCMAPESSlider.children[0].on_event(
            "input", updatePaCMAPESSlider
        )
        mnRatioPaCMAPESSlider.children[0].on_event(
            "input", updatePaCMAPESSlider
        )
        fpRatioPacMAPESSlider.children[0].on_event(
            "input", updatePaCMAPESSlider
        )

        allPaCMAPESSliders = widgets.VBox(
            [
                neighboursPaCMAPESSlider,
                mnRatioPaCMAPESSlider,
                fpRatioPacMAPESSlider,
            ],
            layout=Layout(
                width="100%",
            ),
        )

        validatePaCMAPParamsESBtn = v.Btn(
            children=[
                v.Icon(left=True, children=["mdi-check"]),
                "Validate",
            ]
        )

        resetPaCMAPParamsESBtn = v.Btn(
            class_="ml-4",
            children=[
                v.Icon(left=True, children=["mdi-skip-backward"]),
                "Reset",
            ],
        )

        allPaCMAPParamESBtn = widgets.HBox(
            [validatePaCMAPParamsESBtn, resetPaCMAPParamsESBtn]
        )
        paramsProjESGroup = widgets.VBox(
            [allPaCMAPESSliders, allPaCMAPParamESBtn],
            layout=Layout(width="100%", display="flex", align_items="center"),
        )

        # Recompute ES PaCMAP projection whenever the user changes the parameters
        def updatePaCMAPES(*updatePaCMAPES):

            n_neighbors = neighboursPaCMAPESSlider.children[0].v_model
            MN_ratio = mnRatioPaCMAPESSlider.children[0].v_model
            FP_ratio = fpRatioPacMAPESSlider.children[0].v_model

            computeIndicatorES.layout.visibility = "visible"

            # We compute
            projValues = computeProjections(self.__atk.getDataset().getXValues(Dataset.SCALED), DimReducMethod.PaCMAP, neighbors = neighbors, MN_ratio = MN_ratio, FP_ratio =FP_ratio)
            # We store this in our dataset
            self.__explanationDataset.setValues(self.__exxplanationES, DimReducMethod.PaCMAP, DimReducMethod.DIM_TWO, projValues[0])
            self.__explanationDataset.setValues(self.__exxplanationES, DimReducMethod.PaCMAP, DimReducMethod.DIM_THREE, projValues[1])


            atk.getDataset().setXProjValues(DimReducMethod.PaCMAP, DimReducMethod.DIM_TWO, projValues[0])
            self.__atk.getDataset().setXProjValues(DimReducMethod.PaCMAP, DimReducMethod.DIM_THREE, projValues[1])


            computeIndicatorES.layout.visibility = "hidden"
            compute.update_figures(self, self.__explanation, self.__projectionVS, self.__projectionES)

        validatePaCMAPParamsESBtn.on_event("click", updatePaCMAPES)

        def resetPaCMAPVS(*args):
            computeIndicatorES.layout.visibility = "visible"

             # We compute :
            projValues = computeProjections(self.__atk.getDataset().getExplanations(Dataset.SCALED), DimReducMethod.PaCMAP)
            # We store this in our dataset
            self.__explanationDataset.setValues(self.__exxplanationES, DimReducMethod.PaCMAP, DimReducMethod.DIM_TWO, projValues[0])
            self.__explanationDataset.setValues(self.__exxplanationES, DimReducMethod.PaCMAP, DimReducMethod.DIM_THREE, projValues[1])

            computeIndicatorES.layout.visibility = "hidden"

            compute.update_figures(self, self.__explanation, self.__projectionVS, self.__projectionES)

        reset_params_proj_ES.on_event("click", resetPaCMAPVS)


        # =============== Color management ================

        # Allows you to choose the color of the points
        # y, y hat, residuals, current selection, regions, unselected points, automatic clustering
        colorSelectionBtns = gui_elements.colorChooser()

        def changeMarkersColor(*args, opacity: bool = True):
            # Allows you to change the color of the points when you click on the buttons
            color = None
            scale = True
            to_modify = True
            if colorSelectionBtns.v_model == "y":
                color = self.atk.dataset.y
            elif colorSelectionBtns.v_model == "y^":
                color = self.atk.dataset.y_pred
            elif colorSelectionBtns.v_model == "Selec actuelle":
                scale = False
                color = ["grey"] * len(self.atk.dataset.X)
                for i in range(len(self.selection.indexes)):
                    color[self.selection.indexes[i]] = "blue"
            elif colorSelectionBtns.v_model == "Résidus":
                color = self.atk.dataset.y - self.atk.dataset.y_pred
                color = [abs(i) for i in color]
            elif colorSelectionBtns.v_model == "Régions":
                scale = False
                color = [0] * len(self.atk.dataset.X)
                for i in range(len(self.atk.dataset.X)):
                    for j in range(len(self.atk.regions)):
                        if i in self.atk.regions[j].indexes:
                            color[i] = j + 1
            elif colorSelectionBtns.v_model == "Non selec":
                scale = False
                color = ["red"] * len(self.atk.dataset.X)
                if len(self.atk.regions) > 0:
                    for i in range(len(self.atk.dataset.X)):
                        for j in range(len(self.atk.regions)):
                            if i in self.atk.regions[j].indexes:
                                color[i] = "grey"
            elif colorSelectionBtns.v_model == "Clustering auto":
                color = self.__labels_automatic_clustering
                to_modify = False
                scale = False
            with self.fig1.batch_update():
                self.fig1.data[0].marker.color = color
                if color is not None:
                    self.fig1.data[0].customdata= color
                else:
                    self.fig1.data[0].customdata= [None]*len(self.atk.dataset.X)
                if opacity:
                    self.fig1.data[0].marker.opacity = 1
            with self.fig2.batch_update():
                self.fig2.data[0].marker.color = color
                if opacity:
                    self.fig2.data[0].marker.opacity = 1
                if color is not None:
                    self.fig2.data[0].customdata= color
                else:
                    self.fig2.data[0].customdata= [None]*len(self.atk.dataset.X)
            with self.fig1_3D.batch_update():
                self.fig1_3D.data[0].marker.color = color
                if color is not None:
                    self.fig1_3D.data[0].customdata= color
                else:
                    self.fig1_3D.data[0].customdata= [None]*len(self.atk.dataset.X)
            with self.fig2_3D.batch_update():
                self.fig2_3D.data[0].marker.color = color
                if color is not None:
                    self.fig2_3D.data[0].customdata= color
                else:
                    self.fig2_3D.data[0].customdata= [None]*len(self.atk.dataset.X)
            if scale:
                self.fig1.update_traces(marker=dict(showscale=True))
                self.fig1_3D.update_traces(marker=dict(showscale=True))
                self.fig1.data[0].marker.colorscale = "Viridis"
                self.fig1_3D.data[0].marker.colorscale = "Viridis"
                self.fig2.data[0].marker.colorscale = "Viridis"
                self.fig2_3D.data[0].marker.colorscale = "Viridis"
            else:
                self.fig1.update_traces(marker=dict(showscale=False))
                self.fig1_3D.update_traces(marker=dict(showscale=False))
                if to_modify:
                    self.fig1.data[0].marker.colorscale = "Plasma"
                    self.fig1_3D.data[0].marker.colorscale = "Plasma"
                    self.fig2.data[0].marker.colorscale = "Plasma"
                    self.fig2_3D.data[0].marker.colorscale = "Plasma"
                else:
                    self.fig1.data[0].marker.colorscale = "Viridis"
                    self.fig1_3D.data[0].marker.colorscale = "Viridis"
                    self.fig2.data[0].marker.colorscale = "Viridis"
                    self.fig2_3D.data[0].marker.colorscale = "Viridis"

        colorSelectionBtns.on_event("change", changeMarkersColor)

        # Marker for VS
        ourVSMarker = dict(
            color=self.atk.dataset.y,
            colorscale="Viridis",
            colorbar=dict(
                title="y",
                thickness=20,
            ),
        )

        # Marker for EZS
        ourESMarker = dict(color=self.atk.dataset.y, colorscale="Viridis")

        barMenu, FigureSize, saveBtn = gui_elements.createMenuBar()

        # =============== Backup management ================
        
        initialNumBackups = deepcopy(len(self.__atk.__backups))

        def initBackup(bu):
            regionText = "There is no backup"
            if len(bu) > 0:
                regionText = str(len(bu)) + " backup(s) found"
            backupTable = []
            for i in range(len(bu)):
                new_or_not = "Imported"
                if i > initialNumBackups:
                    new_or_not = "Created"
                backupTable.append(
                    [
                        i + 1,
                        bu[i]["name"],
                        new_or_not,
                        len(bu[i]["regions"]),
                    ]
                )
            backupTable = pd.DataFrame(
                backupTable,
                columns=[
                    "Backup #",
                    "Name",
                    "Origin",
                    "Number of regions",
                ],
            )

            columns = [
                {"text": c, "sortable": True, "value": c} for c in backupTable.columns
            ]

            backupTable = v.DataTable(
                v_model=[],
                show_select=True,
                single_select=True,
                headers=columns,
                items=backupTable.to_dict("records"),
                item_value="Backup #",
                item_key="Backup #",
            )
            return [backupTable, regionText]

        # The table that contains the backups
        self.__backupTable = initBackup(self.__atk.__backups)[0] # TODO : use a getter instead of __atk.__backups

        # Initialize the interface for backups
        backupDialogue, backupCard, deleteBackupBtn, backupName, showBackupBtn, newBackup = gui_elements.createBackupsGUI(saveBtn, initBackup(self.atk.saves)[1], self.__backupTable, self.atk)

        # Save a backup
        def deleteBackup(*args):
            if self.__backupTable.v_model == []:
                return
            self.__backupTable = backupCard.children[1]
            index = self.__backupTable.v_model[0]["Save #"] - 1
            self.atk.saves.pop(index)
            self.__backupTable, text = initBackup(self.atk.saves)
            backupCard.children = [text, self.__backupTable] + backupCard.children[
                2:
            ]

        deleteBackupBtn.on_event("click", deleteBackup)

        # View a backup
        def showBackup(*args):
            self.__backupTable = backupCard.children[1]
            if len(self.__backupTable.v_model) == 0:
                return
            index = self.__backupTable.v_model[0]["Save #"] - 1
            self.atk.regions = [element for element in self.atk.saves[index]["regions"]]
            color = deepcopy(self.atk.saves[index]["labels"])
            self.__autoClusterRegionColors = deepcopy(color)
            with self.fig1.batch_update():
                self.fig1.data[0].marker.color = color
                self.fig1.data[0].marker.opacity = 1
            with self.fig2.batch_update():
                self.fig2.data[0].marker.color = color
                self.fig2.data[0].marker.opacity = 1
            with self.fig1_3D.batch_update():
                self.fig1_3D.data[0].marker.color = color
            with self.fig2_3D.batch_update():
                self.fig2_3D.data[0].marker.color = color
            colorSelectionBtns.v_model = "Régions"
            self.fig1.update_traces(marker=dict(showscale=False))
            self.fig2.update_traces(marker=dict(showscale=False))
            function_new_region()

        showBackupBtn.on_event("click", showBackup)

        # Create a new backup with the current regions
        def newBackup(*args):
            if len(backupName.v_model) == 0 or len(backupName.v_model) > 25:
                raise Exception("The name of the backup must be between 1 and 25 characters !")
            # TODO : use a getter instead of __atk.__regions
            backup1 = {"regions": self.__atk.__regions, "labels": self.__regionsColor,"name": backupName.v_model}
            backup = {key: value[:] for key, value in backup1.items()}
            self.__atk.__saves.append(backup)
            self.__backupTable, text_region = initBackup(self.atk.saves)
            backupCard.children = [text_region, self.__backupTable] + backupCard.children[2:]
            

        newBackup.on_event("click", newBackup)


        # ======================  Figures ======================

        # Value Space figure, on the left
        self.__leftVSFigure = go.FigureWidget(
            data=go.Scatter(x=[1], y=[1], mode="markers", marker=ourVSMarker, customdata=ourVSMarker["color"], hovertemplate = '%{customdata:.3f}')
        )

        # To remove the Plotly logo
        self.__leftVSFigure._config = self.__leftVSFigure._config | {"displaylogo": False}

        # Border size
        M = 40

        self.__leftVSFigure.update_layout(margin=dict(l=M, r=M, t=0, b=M), width=int(FigureSize.v_model))
        self.__leftVSFigure.update_layout(dragmode="lasso")

        # Explanation Space figure, on the right
        self.__rightESFigure = go.FigureWidget(
            data=go.Scatter(x=[1], y=[1], mode="markers", marker=ourESMarker, customdata=ourESMarker["color"], hovertemplate = '%{customdata:.3f}')
        )

        self.__rightESFigure.update_layout(margin=dict(l=M, r=M, t=0, b=M), width=int(FigureSize.v_model))
        self.__rightESFigure.update_layout(dragmode="lasso")

        self.__rightESFigure._config = self.__rightESFigure._config | {"displaylogo": False}

        # two checkboxes to choose the projection dimension of figures 1 and 2
        our2D3DSwitch = v.Switch(
            class_="ml-3 mr-2",
            v_model=False,
            label="",
        )

        our2D3DSwitchWithGroup = v.Row(
            class_="ma-3",
            children=[
                v.Icon(children=["mdi-numeric-2-box"]),
                v.Icon(children=["mdi-alpha-d-box"]),
                our2D3DSwitch,
                v.Icon(children=["mdi-numeric-3-box"]),
                v.Icon(children=["mdi-alpha-d-box"]),
            ],
        )

        our2D3DSwitchWithGroup = addTooltip(
            our2D3DSwitchWithGroup, "Dimension of the projection" # TODO : should be localized
        )

        def function_dimension_projection(*args):
            if our2D3DSwitch.v_model:
                currentView.children = [ourESView3D, ourESView3D]
            else:
                currentView.children = [ourVSView2D, ourESView2D]

        our2D3DSwitch.on_event("change", function_dimension_projection)

        # Marker for VS  in 3D
        ourVSMarker3D = dict(
            color=self.atk.dataset.y,
            colorscale="Viridis",
            colorbar=dict(
                thickness=20,
            ),
            size=3,
        )

        # Marker for ES in 3D
        ourESMarker3D = dict(color=self.atk.dataset.y, colorscale="Viridis", size=3)

        self.__leftVSFigure3D = go.FigureWidget(
            data=go.Scatter3d(
                x=[1], y=[1], z=[1], mode="markers", marker=marker_3D,  customdata=marker_3D["color"], hovertemplate = '%{customdata:.3f}'
            )
        )
        self.__leftVSFigure3D.update_layout(
            margin=dict(l=M, r=M, t=0, b=M),
            width=int(FigureSize.v_model),
            scene=dict(aspectmode="cube"),
            template="none",
        )

        self.__leftVSFigure3D._config = self.__leftVSFigure3D._config | {"displaylogo": False}

        self.__rightESFigure3D = go.FigureWidget(
            data=go.Scatter3d(
                x=[1], y=[1], z=[1], mode="markers", marker=marker_3D_2, customdata=marker_3D_2["color"], hovertemplate = '%{customdata:.3f}'
            )
        )
        self.__rightESFigure3D.update_layout(
            margin=dict(l=M, r=M, t=0, b=M),
            width=int(FigureSize.v_model),
            scene=dict(aspectmode="cube"),
            template="none",
        )

        self.fig2_3D._config = self.fig2_3D._config | {"displaylogo": False}

# --------------- update figues ---------------------

    def updateFigures(self):

        projValues = computeProjections(self.__atk.getDataset().getXValues(Dataset.CURRENT), self.__projectionVS[0])

        with self.__leftVSFigure2D.batch_update():
            self.__leftVSFigure2D.data[0].x, self.__leftVSFigure2D.data[0].y  = projValues[0][0], projValues[0][1]
        
        with self.__leftVSFigure3D.batch_update():
            self.__leftVSFigure3D.data[0].x, self.__leftVSFigure3D.data[0].y, self.__leftVSFigure3D.data[0].z = projValues[1][0], projValues[1][1], projValues[1][2]
                
        projValues = computeProjections(self.__explanationDataset().getValues(self.__projectionES[0]))

        with self.__rightESFigure2D.batch_update():
            self.__rightESFigure2D.data[0].x, self.__rightESFigure2D.data[0].y = projValues[0][0], projValues[0][1]
        
        with self.__rightESFigure3D.batch_update():
            self.__rightESFigure3D.data[0].x, self.__rightESFigure3D.data[0].y, self.__rightESFigure3D.data[0].z = projValues[1][0], projValues[1][1], projValues[1][2]


        # We define "Views" = the figures and the text above!
        ourVSView2D = gui_elements.textOverFigure(self.__leftVSFigure2D, widgets.HTML("<h3>Values Space<h3>"))
        ourESView2D = gui_elements.textOverFigure(self.__rightESFigure2D, widgets.HTML("<h3>Explanatory Space<h3>"))
        ourVSView3D = gui_elements.textOverFigure(self.__leftVSFigure3D, widgets.HTML("<h3>Values Space<h3>"))
        ourESView3D = gui_elements.textOverFigure(self.__rightESFigure3D, widgets.HTML("<h3>Explanatory Space<h3>"))

        # Default view is 2D
        currentView = widgets.HBox([ourVSView2D, ourESView2D])

        # Update both figures according to the chosen projection
        def updateScatterPlots(*args):
            self.__projectionVS[0] = deepcopy(dimReducVSDropdown.v_model)
            self.__projectionES[0] = deepcopy(dimReducESDropdown.v_model)

            #TODO : to clarify : there's no mention of the dimension (2 or 3)

            if self.__atk.getDataset().getValues(self.__projectionVS[0], self.__projectionVS[1]) is None:
                # We don't have the projection computed. Let's compute it
                computeIndicatorVS.layout.visibility = "visible"
                projValues = commputeProjections(self.__atk.getDataset().getXValues(Dataset.SCALED), self.__projectionVS[0])            
                self.__atk.getDataset().setXProjValues(self.__projectionVS[0], self.__projectionVS[1], projValues)
                computeIndicatorVS.layout.visibility = "hidden"
            

            if self.__explainationDataset.getValues(self.__projectionES[0], self.__projectionES[1]) is None:
                # We don't have the projection computed. Let's compute it
                computeIndicatorES.layout.visibility = "visible"
                projValues = commputeProjections(self.__explainationDataset.getXValues(Dataset.SCALED), self.__projectionES[0], __projectionES[1])

                self.__explainationDataset.setValues(self.__projectionES[0], self.__projectionES[1], projValues)    
                
                computeIndicatorES.layout.visibility = "hidden"

            updateFigures(self)


            ########################################
            ##
            ## J'en suis là
            ##
            ########################################

            # for the parameters of the projection
            if dimReducVSDropdown.v_model == "PaCMAP":
                param_VS.v_slots[0]["children"].disabled = False
            else:
                param_VS.v_slots[0]["children"].disabled = True
            if dimReducESDropdown.v_model == "PaCMAP":
                param_ES.v_slots[0]["children"].disabled = False
            else:
                param_ES.v_slots[0]["children"].disabled = True

            dimReducVSDropdown.v_model = self.__projectionVS[0]
            dimReducESDropdown.v_model = self.__projectionES[0]

        # we observe the changes in the values ​​of the dropdowns to change the method of reduction
        dimReducVSDropdown.on_event("change", updateScatterPlots)
        dimReducESDropdown.on_event("change", updateScatterPlots)

        self.__regionsColor = [0] * len(self.atk.dataset.X)

        # definition of the table that will show the different results of the regions, with a stat of info about them
        table_regions = widgets.Output()

        # definition of the text that will give information on the selection
        text_base, text_base_debut, text_selec, card_selec, button_validate_skope, button_reset_skope = gui_elements.create_selection_card()

        text_skope, text_skopeES, text_skopeVS, one_card_VS, one_card_ES = gui_elements.create_card_skope()

        # texts that will contain the information on the self.sub_models
        mods = gui_elements.create_slide_sub_models(self)

        def change(widget, event, data, args: bool = True):
            if args == True:
                for i in range(len(mods.children)):
                    mods.children[i].children[0].color = "white"
                widget.color = "blue lighten-4"
            for i in range(len(mods.children)):
                if mods.children[i].children[0].color == "blue lighten-4":
                    self.__model_index = i

        for i in range(len(mods.children)):
            mods.children[i].children[0].on_event("click", change)

        validate_one_region, supprimer_toutes_les_tuiles, selection = gui_elements.create_buttons_regions()

        # we define the sliders used to modify the histogram resulting from the skope
        slider_skope1, button_in_real_time_graph1, slider_text_comb1 = gui_elements.slider_skope()
        slider_skope2, button_in_real_time_graph2, slider_text_comb2 = gui_elements.slider_skope()
        slider_skope3, button_in_real_time_graph3, slider_text_comb3 = gui_elements.slider_skope()

        # if "in real-time" is checked, no need to validate the changes!
        def update_validate1(*args):
            if button_in_real_time_graph1.v_model:
                validate_change_1.disabled = True
            else:
                validate_change_1.disabled = False
        button_in_real_time_graph1.on_event("change", update_validate1)

        def update_validate2(*args):
            if button_in_real_time_graph2.value:
                validate_change_2.disabled = True
            else:
                validate_change_2.disabled = False
        button_in_real_time_graph2.on_event("change", update_validate2)

        def update_validate3(*args):
            if button_in_real_time_graph3.v_model:
                validate_change_3.disabled = True
            else:
                validate_change_3.disabled = False
        button_in_real_time_graph3.on_event("change", update_validate3)

        # valid buttons definition changes:

        def validate_change():
            widget = v.Btn(
                class_="ma-3",
                children=[
                    v.Icon(class_="mr-2", children=["mdi-check"]),
                    "Validate the changes",
                ],
            )
            return widget

        validate_change_1 = validate_change()
        validate_change_2 = validate_change()
        validate_change_3 = validate_change()

        # we wrap the validation button and the checkbox which allows you to view in real time
        two_end1 = widgets.HBox([validate_change_1, button_in_real_time_graph1])
        two_end2 = widgets.HBox([validate_change_2, button_in_real_time_graph2])
        two_end3 = widgets.HBox([validate_change_3, button_in_real_time_graph3])

        # we define the number of bars of the histogram
        number_of_bins_histograms = 50
        # we define the histograms
        [histogram1, histogram2, histogram3] = gui_elements.create_histograms(number_of_bins_histograms, FigureSize.v_model)

        histogram1 = deepcopy(histogram1)
        histogram2 = deepcopy(histogram2)
        histogram3 = deepcopy(histogram3)
        all_histograms = [histogram1, histogram2, histogram3]

        # definitions of the different color choices for the swarm
        [total_beeswarm_1, total_beeswarm_2, total_beeswarm_3] = gui_elements.create_beeswarms(self, self.__explanation, FigureSize.v_model)

        choice_color_beeswarm1 = total_beeswarm_1.children[0]
        choice_color_beeswarm2 = total_beeswarm_2.children[0]
        choice_color_beeswarm3 = total_beeswarm_3.children[0]

        beeswarm1 = total_beeswarm_1.children[1]
        beeswarm2 = total_beeswarm_2.children[1]
        beeswarm3 = total_beeswarm_3.children[1]

        # update the beeswarm plots
        def change_color_beeswarm_shap1(*args):
            if choice_color_beeswarm1.children[1].v_model == False:
                marker = compute.function_beeswarm_shap(self, self.__explanation, self.selection.rules[0][2])[1]
                beeswarm1.data[0].marker = marker
                beeswarm1.update_traces(marker=dict(showscale=True))
            else:
                modifie_all_histograms(
                    slider_skope1.v_model[0]  , slider_skope1.v_model[1]  , 0
                )
                beeswarm1.update_traces(marker=dict(showscale=False))

        choice_color_beeswarm1.children[1].on_event(
            "change", change_color_beeswarm_shap1
        )

        def change_color_beeswarm_shap2(*args):
            if choice_color_beeswarm2.children[1].v_model == False:
                marker = compute.function_beeswarm_shap(self, self.__explanation, self.selection.rules[1][2])[1]
                beeswarm2.data[0].marker = marker
                beeswarm2.update_traces(marker=dict(showscale=True))
            else:
                modifie_all_histograms(
                    slider_skope2.v_model[0]  , slider_skope2.v_model[1]  , 1
                )
            beeswarm2.update_traces(marker=dict(showscale=False))

        choice_color_beeswarm2.children[1].on_event(
            "change", change_color_beeswarm_shap2
        )

        def change_color_beeswarm_shap3(*args):
            if choice_color_beeswarm3.children[1].v_model == False:
                marker = compute.function_beeswarm_shap(self, self.__explanation, self.selection.rules[2][2])[1]
                beeswarm3.data[0].marker = marker
                beeswarm3.update_traces(marker=dict(showscale=True))
            else:
                modifie_all_histograms(
                    slider_skope3.v_model[0]  , slider_skope3.v_model[1]  , 2
                )
                beeswarm3.update_traces(marker=dict(showscale=False))

        choice_color_beeswarm3.children[1].on_event(
            "change", change_color_beeswarm_shap3
        )

        all_beeswarms_total = [total_beeswarm_1, total_beeswarm_2, total_beeswarm_3]

        all_beeswarms = [beeswarm1, beeswarm2, beeswarm3]

        all_color_choosers_beeswarms = [
            choice_color_beeswarm1,
            choice_color_beeswarm2,
            choice_color_beeswarm3,
        ]
        # set of elements that contain histograms and sliders
        all_widgets_slider_histo1 = widgets.VBox([slider_text_comb1, histogram1, two_end1])
        all_widgets_slider_histo2 = widgets.VBox([slider_text_comb2, histogram2, two_end2])
        all_widgets_slider_histo3 = widgets.VBox([slider_text_comb3, histogram3, two_end3])

        # definition of buttons to delete features (disabled for the first 3 for the moment)
        b_delete_skope1 = gui_elements.button_delete_skope()
        b_delete_skope2 = gui_elements.button_delete_skope()
        b_delete_skope3 = gui_elements.button_delete_skope()

        # checkbow to know if the feature is continuous or not
        is_continuous_1 = v.Checkbox(v_model=True, label="is continuous?")
        is_continuous_2 = v.Checkbox(v_model=True, label="is continuous?")
        is_continuous_3 = v.Checkbox(v_model=True, label="is continuous?")

        # the right side of the features : button to delete the feature from the rules + checkbox "is continuous?"
        right_side_1 = v.Col(children=[b_delete_skope1, is_continuous_1], class_="d-flex flex-column align-center justify-center")
        right_side_2 = v.Col(children=[b_delete_skope2, is_continuous_2], class_="d-flex flex-column align-center justify-center")
        right_side_3 = v.Col(children=[b_delete_skope3, is_continuous_3], class_="d-flex flex-column align-center justify-center")

        self.__all_widgets_class_1 = gui_elements.create_class_selector(self, self.atk.dataset.X.columns[0])
        self.__all_widgets_class_2 = gui_elements.create_class_selector(self, self.atk.dataset.X.columns[0])
        self.__all_widgets_class_3 = gui_elements.create_class_selector(self, self.atk.dataset.X.columns[0])

        # when teh user checks the fact that this feature 
        def change_continuous1(widget, event, data):
            if widget.v_model == True and widget == right_side_1.children[1]:
                in_accordion1.children = [all_widgets_slider_histo1] + list(in_accordion1.children[1:])
                count = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i-count][2] == self.selection.rules[0][2] and i-count != 0:
                        self.selection.rules.pop(i-count)
                        count += 1
                self.selection.rules[0][0] = slider_skope1.v_model[0]
                self.selection.rules[0][4] = slider_skope1.v_model[1]
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()
            else:
                in_accordion1.children = [self.__all_widgets_class_1] + list(in_accordion1.children[1:])
                l = []
                for i in range(len(self.__all_widgets_class_1.children[2].children)):
                    if self.__all_widgets_class_1.children[2].children[i].v_model:
                        l.append(int(self.__all_widgets_class_1.children[2].children[i].label))
                if len(l) == 0:
                    widget.v_model = True
                    return
                column = deepcopy(self.selection.rules[0][2])
                count = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i-count][2] == column:
                        self.selection.rules.pop(i-count)
                        count += 1
                croissant = 0
                for ele in l:
                    self.selection.rules.insert(0+croissant, [ele-0.5, '<=', column, '<=', ele+0.5])
                    croissant += 1
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules), is_class=True)
                update_all_graphs()

        def change_continuous2(widget, event, data):
            features = [self.selection.rules[i][2] for i in range(len(self.selection.rules))]
            the_set = []
            for i in range(len(features)):
                if features[i] not in the_set:
                    the_set.append(features[i])
            index = features.index(the_set[1])
            if widget.v_model and widget == right_side_2.children[1]:
                in_accordion2.children = [all_widgets_slider_histo2] + list(in_accordion2.children[1:])
                count = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i-count][2] == self.selection.rules[index][2] and i-count != index:
                        self.selection.rules.pop(i-count)
                        count += 1
                self.selection.rules[index][0] = slider_skope2.v_model[0]
                self.selection.rules[index][4] = slider_skope2.v_model[1]
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()
            else:
                in_accordion2.children = [self.__all_widgets_class_2] + list(in_accordion2.children[1:])
                l = []
                for i in range(len(self.__all_widgets_class_2.children[2].children)):
                    if self.__all_widgets_class_2.children[2].children[i].v_model:
                        l.append(int(self.__all_widgets_class_2.children[2].children[i].label))
                if len(l) == 0:
                    widget.v_model = True
                    return
                column = deepcopy(self.selection.rules[index][2])
                count = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i-count][2] == column:
                        self.selection.rules.pop(i-count)
                        count += 1
                croissant = 0
                for ele in l:
                    self.selection.rules.insert(index+croissant, [ele-0.5, '<=', column, '<=', ele+0.5])
                    croissant += 1
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules), is_class=True)
                update_all_graphs()

        def change_continuous3(widget, event, data):
            features = [self.selection.rules[i][2] for i in range(len(self.selection.rules))]
            the_set = []
            for i in range(len(features)):
                if features[i] not in the_set:
                    the_set.append(features[i])
            index = features.index(the_set[2])
            if widget.v_model and widget == right_side_3.children[1]:
                in_accordion3.children = [all_widgets_slider_histo3] + list(in_accordion3.children[1:])
                count = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i-count][2] == self.selection.rules[index][2] and i-count != index:
                        self.selection.rules.pop(i-count)
                        count += 1
                self.selection.rules[index][0] = slider_skope3.v_model[0]
                self.selection.rules[index][4] = slider_skope3.v_model[1]
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()
            else:
                in_accordion3.children = [self.__all_widgets_class_3] + list(in_accordion3.children[1:])
                l = []
                for i in range(len(self.__all_widgets_class_3.children[2].children)):
                    if self.__all_widgets_class_3.children[2].children[i].v_model:
                        l.append(int(self.__all_widgets_class_3.children[2].children[i].label))
                if len(l) == 0:
                    widget.v_model = True
                    return
                column = deepcopy(self.selection.rules[index][2])
                count = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i-count][2] == column:
                        self.selection.rules.pop(i-count)
                        count += 1
                croissant = 0
                for ele in l:
                    self.selection.rules.insert(index+croissant, [ele-0.5, '<=', column, '<=', ele+0.5])
                    croissant += 1
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules), is_class=True)
                update_all_graphs()

        # the "is continuous" checkbox
        right_side_1.children[1].on_event("change", change_continuous1)
        right_side_2.children[1].on_event("change", change_continuous2)
        right_side_3.children[1].on_event("change", change_continuous3)

        in_accordion1 = widgets.HBox(
            [all_widgets_slider_histo1, total_beeswarm_1, right_side_1],
            layout=Layout(align_items="center"),
        )
        in_accordion2 = widgets.HBox(
            [all_widgets_slider_histo2, total_beeswarm_2, right_side_2],
            layout=Layout(align_items="center"),
        )
        in_accordion3 = widgets.HBox(
            [all_widgets_slider_histo3, total_beeswarm_3, right_side_3],
            layout=Layout(align_items="center"),
        )

        # we define several accordions in this way to be able to open several at the same time
        in_accordion1_n = gui_elements.accordion_skope("X1", in_accordion1)
        in_accordion2_n = gui_elements.accordion_skope("X2", in_accordion2)
        in_accordion3_n = gui_elements.accordion_skope("X3", in_accordion3)

        accordion_skope = widgets.VBox(
            children=[in_accordion1_n, in_accordion2_n, in_accordion3_n],
            layout=Layout(width="100%", height="auto"),
        )

        # allows you to take the set of rules and modify the graph so that it responds to everything!
        def update_all_graphs():
            self.selection.state = Potato.REFINED_SKR
            """
            new_tuile = self.atk.dataset.X[
                (self.atk.dataset.X[self.selection.rules[0][2]] >= self.selection.rules[0][0])
                & (self.atk.dataset.X[self.selection.rules[0][2]] <= self.selection.rules[0][4])
            ].index
            for i in range(1, len(self.selection.rules)):
                X_temp = self.atk.dataset.X[
                    (self.atk.dataset.X[self.selection.rules[i][2]] >= self.selection.rules[i][0])
                    & (self.atk.dataset.X[self.selection.rules[i][2]] <= self.selection.rules[i][4])
                ].index
                new_tuile = [g for g in new_tuile if g in X_temp]
            """
            new_tuile = self.selection.applyRules(to_return=True)
            y_shape_skope = []
            y_color_skope = []
            y_opa_skope = []
            for i in range(len(self.atk.dataset.X)):
                if i in new_tuile:
                    y_shape_skope.append("circle")
                    y_color_skope.append("blue")
                    y_opa_skope.append(0.5)
                else:
                    y_shape_skope.append("cross")
                    y_color_skope.append("grey")
                    y_opa_skope.append(0.5)
            with self.fig1.batch_update():
                self.fig1.data[0].marker.color = y_color_skope
            with self.fig2.batch_update():
                self.fig2.data[0].marker.color = y_color_skope
            with self.fig1_3D.batch_update():
                self.fig1_3D.data[0].marker.color = y_color_skope
            with self.fig2_3D.batch_update():
                self.fig2_3D.data[0].marker.color = y_color_skope

        # allows to modify all the histograms according to the rules
        def modifie_all_histograms(value_min, value_max, index):
            new_list_tout = self.atk.dataset.X.index[
                self.atk.dataset.X[self.selection.rules[index][2]].between(value_min, value_max)
            ].tolist()
            for i in range(len(self.selection.rules)):
                min = self.selection.rules[i][0]
                max = self.selection.rules[i][4]
                if i != index:
                    new_list_temp = self.atk.dataset.X.index[
                        self.atk.dataset.X[self.selection.rules[i][2]].between(min, max)
                    ].tolist()
                    new_list_tout = [g for g in new_list_tout if g in new_list_temp]
            if self.selection.indexes_from_map is not None:
                new_list_tout = [g for g in new_list_tout if g in self.selection.indexes_from_map]
            for i in range(len(self.selection.rules)):
                with all_histograms[i].batch_update():
                    all_histograms[i].data[2].x = self.atk.dataset.X[self.selection.rules[i][2]][new_list_tout]
                if all_color_choosers_beeswarms[i].children[1].v_model:
                    with all_beeswarms[i].batch_update():
                        y_color = [0] * len(self.atk.explain[self.__explanation])
                        if i == index:
                            indexs = self.atk.dataset.X.index[
                                self.atk.dataset.X[self.selection.rules[i][2]].between(value_min, value_max)
                            ].tolist()
                        else:
                            indexs = self.atk.dataset.X.index[
                                self.atk.dataset.X[self.selection.rules[i][2]].between(
                                    self.selection.rules[i][0], self.selection.rules[i][4]
                                )
                            ].tolist()
                        for j in range(len(self.atk.explain[self.__explanation])):
                            if j in new_list_tout:
                                y_color[j] = "blue"
                            elif j in indexs:
                                y_color[j] = "#85afcb"
                            else:
                                y_color[j] = "grey"
                        all_beeswarms[i].data[0].marker.color = y_color

        # when the value of a slider is modified, the histograms and graphs are modified
        def on_value_change_skope1(widget, event, data):
            if widget.__class__.__name__ == "RangeSlider":
                slider_text_comb1.children[0].v_model = slider_skope1.v_model[0]
                slider_text_comb1.children[2].v_model = slider_skope1.v_model[1]
            else :
                if slider_text_comb1.children[0].v_model == '' or slider_text_comb1.children[2].v_model == '':
                    return
                else:
                    slider_skope1.v_model = [float(slider_text_comb1.children[0].v_model), float(slider_text_comb1.children[2].v_model)]
            new_list = [
                g
                for g in list(self.atk.dataset.X[self.selection.rules[0][2]].values)
                if g >= slider_skope1.v_model[0]
                and g <= slider_skope1.v_model[1]
            ]
            with histogram1.batch_update():
                histogram1.data[1].x = new_list
            if self.__activate_histograms:
                modifie_all_histograms(
                    slider_skope1.v_model[0], slider_skope1.v_model[1], 0
                )
            if button_in_real_time_graph1.v_model:
                #self.selection.rules[0][0] = float(deepcopy(slider_skope1.v_model[0]))
                #self.selection.rules[0][4] = float(deepcopy(slider_skope1.v_model[1]))
                self.selection.rules[0][0] = float(deepcopy(slider_skope1.v_model[0]))
                self.selection.rules[0][4] = float(deepcopy(slider_skope1.v_model[1]))
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()

        def on_value_change_skope2(widget, event, data):
            if widget.__class__.__name__ == "RangeSlider":
                slider_text_comb2.children[0].v_model = slider_skope2.v_model[0]  
                slider_text_comb2.children[2].v_model = slider_skope2.v_model[1]  
            new_list = [
                g
                for g in list(self.atk.dataset.X[self.selection.rules[1][2]].values)
                if g >= slider_skope2.v_model[0]  
                and g <= slider_skope2.v_model[1]  
            ]
            with histogram2.batch_update():
                histogram2.data[1].x = new_list
            if self.__activate_histograms:
                modifie_all_histograms(
                    slider_skope2.v_model[0]  , slider_skope2.v_model[1]  , 1
                )
            if button_in_real_time_graph2.v_model:
                self.selection.rules[1][0] = float(deepcopy(slider_skope2.v_model[0]  ))
                self.selection.rules[1][4] = float(deepcopy(slider_skope2.v_model[1]  ))
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()

        def on_value_change_skope3(widget, event, data):
            if widget.__class__.__name__ == "RangeSlider":
                slider_text_comb3.children[0].v_model = slider_skope3.v_model[0]  
                slider_text_comb3.children[2].v_model = slider_skope3.v_model[1]  
            new_list = [
                g
                for g in list(self.atk.dataset.X[self.selection.rules[2][2]].values)
                if g >= slider_skope3.v_model[0]  
                and g <= slider_skope3.v_model[1]  
            ]
            with histogram3.batch_update():
                histogram3.data[1].x = new_list
            if self.__activate_histograms:
                modifie_all_histograms(
                    slider_skope3.v_model[0]  , slider_skope3.v_model[1]  , 2
                )
            if button_in_real_time_graph3.v_model:
                self.selection.rules[2][0] = float(deepcopy(slider_skope3.v_model[0]  ))
                self.selection.rules[2][4] = float(deepcopy(slider_skope3.v_model[1]  ))
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()

        def liste_to_string_skope(liste, is_class=False):
            chaine = ""
            for rule in liste:
                for i in range(len(rule)):
                    if type(rule[i]) == float:
                        chaine += str(np.round(float(rule[i]), 2))
                    elif rule[i] is None:
                        chaine += "None"
                    elif type(rule[i]) == list:
                        chaine+="{"
                        chaine += str(rule[i][0])
                        for j in range(1, len(rule[i])):
                            chaine += "," + str(rule[i][j])
                        chaine+="}"
                    else:
                        chaine += str(rule[i])
                    chaine += " "
            return chaine

        # cwhen the user validates the updates he makes on a rule
        def function_change_validate_1(*change):
            a = deepcopy(float(slider_skope1.v_model[0]))
            b = deepcopy(float(slider_skope1.v_model[1]))
            self.selection.rules[0][0] = a
            self.selection.rules[0][4] = b
            one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
            update_all_graphs()
            function_scores_models(None)

        def function_change_validate_2(*change):
            self.selection.rules[1][0] = float(slider_skope2.v_model[0])
            self.selection.rules[1][4] = float(slider_skope2.v_model[1])
            one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
            update_all_graphs()
            function_scores_models(None)

        def function_change_validate_3(*change):
            self.selection.rules[2][0] = float(slider_skope3.v_model[0])
            self.selection.rules[2][4] = float(slider_skope3.v_model[1])
            one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
            update_all_graphs()
            function_scores_models(None)

        validate_change_1.on_event("click", function_change_validate_1)
        validate_change_2.on_event("click", function_change_validate_2)
        validate_change_3.on_event("click", function_change_validate_3)

        def rules_to_indexs():
            liste_bool = [True] * len(self.atk.dataset.X)
            for i in range(len(self.atk.dataset.X)):
                for j in range(len(self.selection.rules)):
                    column = list(self.atk.dataset.X.columns).index(self.selection.rules[j][2])
                    if (
                        self.selection.rules[j][0] > self.atk.dataset.X.iloc[i, column]
                        or self.atk.dataset.X.iloc[i, column] > self.selection.rules[j][4]
                    ):
                        liste_bool[i] = False
            temp = [i for i in range(len(self.atk.dataset.X)) if liste_bool[i]]
            return temp

        def function_scores_models(temp):
            if type(temp) == type(None):
                temp = rules_to_indexs()
            result_models = function_models(self.atk.dataset.X.iloc[temp, :], self.atk.dataset.y.iloc[temp], self.sub_models)
            score_tot = []
            for i in range(len(self.sub_models)):
                score_tot.append(compute.function_score(self.atk.dataset.y.iloc[temp], result_models[i][-2]))
            score_init = compute.function_score(self.atk.dataset.y.iloc[temp], self.atk.dataset.y_pred[temp])
            if score_init == 0:
                l_compar = ["/"] * len(self.sub_models)
            else:
                l_compar = [
                    round(100 * (score_init - score_tot[i]) / score_init, 1)
                    for i in range(len(self.sub_models))
                ]

            self.__score_sub_models = []
            for i in range(len(self.sub_models)):
                self.__score_sub_models.append(
                    [
                        score_tot[i],
                        score_init,
                        l_compar[i],
                    ]
                )

            # to generate a string for the scores
            # TODO: different score for the classification! Recall/precision!
            def string_for_score(i):
                if score_tot[i] == 0:
                    return (
                        "MSE = "
                        + str(score_tot[i])
                        + " (against "
                        + str(score_init)
                        + ", +"
                        + "∞"
                        + "%)"
                    )
                else:
                    if round(100 * (score_init - score_tot[i]) / score_init, 1) > 0:
                        return (
                            "MSE = "
                            + str(score_tot[i])
                            + " (against "
                            + str(score_init)
                            + ", +"
                            + str(
                                round(100 * (score_init - score_tot[i]) / score_init, 1)
                            )
                            + "%)"
                        )
                    else:
                        return (
                            "MSE = "
                            + str(score_tot[i])
                            + " (against "
                            + str(score_init)
                            + ", "
                            + str(
                                round(100 * (score_init - score_tot[i]) / score_init, 1)
                            )
                            + "%)"
                        )

            for i in range(len(self.sub_models)):
                mods.children[i].children[0].children[1].children = string_for_score(i)

        # when you click on the skope-rules button
        def function_validation_skope(*sender):
            loading_models.class_ = "d-flex"
            self.__activate_histograms = True
            if self.selection.y_train == None:
                text_skopeVS.children[1].children = [
                    widgets.HTML("Please select points")
                ]
                text_skopeES.children[1].children = [
                    widgets.HTML("Please select points")
                ]
            elif 0 not in self.selection.y_train or 1 not in self.selection.y_train:
                text_skopeVS.children[1].children = [
                    widgets.HTML("You can't choose everything/nothing !")
                ]
                text_skopeES.children[1].children = [
                    widgets.HTML("You can't choose everything/nothing !")
                ]
            else:
                # skope calculation for X
                self.selection.applySkope(self.__explanation, 0.2, 0.2)
                print(self.selection.rules)
                # if no rule for one of the two, nothing is displayed
                if self.selection.success == False:
                    text_skopeVS.children[1].children = [
                        widgets.HTML("No rule found")
                    ]
                    text_skopeES.children[1].children = [
                        widgets.HTML("No rule found")
                    ]
                # otherwise we display
                else:
                    #chaine_carac = transform_string(skope_rules_clf.rules_[0])
                    text_skopeVS.children[0].children[3].children = [
                        "p = "
                        + str(self.selection.score[0])
                        + "%"
                        + " r = "
                        + str(self.selection.score[1])
                        + "%"
                        + " ext. of the tree = "
                        + str(self.selection.score[2])
                    ]

                    # there we find the values ​​of the skope to use them for the sliders
                    columns_rules = [self.selection.rules[i][2] for i in range(len(self.selection.rules))]
                    new_columns_rules = []
                    for i in range(len(columns_rules)):
                        if columns_rules[i] not in new_columns_rules:
                            new_columns_rules.append(columns_rules[i])
                    columns_rules = new_columns_rules

                    self.__other_columns = [g for g in self.atk.dataset.X.columns if g not in columns_rules]

                    widget_list_add_skope.items = self.__other_columns
                    widget_list_add_skope.v_model = self.__other_columns[0]

                    self.selection.rules = self.selection.rules

                    one_card_VS.children = gui_elements.generate_rule_card(
                        liste_to_string_skope(self.selection.rules)
                    )

                    [new_y, marker] = compute.function_beeswarm_shap(self, self.__explanation, self.selection.rules[0][2])
                    beeswarm1.data[0].y = deepcopy(new_y)
                    beeswarm1.data[0].x = self.atk.explain[self.__explanation][columns_rules[0] + "_shap"]
                    beeswarm1.data[0].marker = marker

                    all_histograms = [histogram1]
                    if len(set([self.selection.rules[i][2] for i in range(len(self.selection.rules))])) > 1:
                        all_histograms = [histogram1, histogram2]
                        [new_y, marker] = compute.function_beeswarm_shap(self, self.__explanation, self.selection.rules[1][2])
                        beeswarm2.data[0].y = deepcopy(new_y)
                        beeswarm2.data[0].x = self.atk.explain[self.__explanation][columns_rules[1] + "_shap"]
                        beeswarm2.data[0].marker = marker

                    if len(set([self.selection.rules[i][2] for i in range(len(self.selection.rules))])) > 2:
                        all_histograms = [histogram1, histogram2, histogram3]
                        [new_y, marker] = compute.function_beeswarm_shap(self, self.__explanation, self.selection.rules[2][2])
                        beeswarm3.data[0].y = deepcopy(new_y)
                        beeswarm3.data[0].x = self.atk.explain[self.__explanation][columns_rules[2] + "_shap"]
                        beeswarm3.data[0].marker = marker

                    y_shape_skope = []
                    y_color_skope = []
                    y_opa_skope = []
                    for i in range(len(self.atk.dataset.X)):
                        if i in self.selection.indexes:
                            y_shape_skope.append("circle")
                            y_color_skope.append("blue")
                            y_opa_skope.append(0.5)
                        else:
                            y_shape_skope.append("cross")
                            y_color_skope.append("grey")
                            y_opa_skope.append(0.5)
                    colorSelectionBtns.v_model = "Selec actuelle"
                    changeMarkersColor(None)

                    accordion_skope.children = [
                        in_accordion1_n,
                    ]

                    in_accordion1_n.children[0].children[0].children = (
                        "X1 (" + columns_rules[0].replace("_", " ") + ")"
                    )

                    if len(columns_rules) > 1:
                        accordion_skope.children = [
                            in_accordion1_n,
                            in_accordion2_n,
                        ]
                        in_accordion2_n.children[0].children[0].children = (
                            "X2 (" + columns_rules[1].replace("_", " ") + ")"
                        )
                    if len(columns_rules) > 2:
                        accordion_skope.children = [
                            in_accordion1_n,
                            in_accordion2_n,
                            in_accordion3_n,
                        ]
                        in_accordion3_n.children[0].children[0].children = (
                            "X3 (" + columns_rules[2].replace("_", " ") + ")"
                        )

                    self.__all_widgets_class_1 = gui_elements.create_class_selector(self, columns_rules[0], self.selection.rules[0][0], self.selection.rules[0][4], fig_size=FigureSize.v_model)
                    if len(columns_rules) > 1:
                        self.__all_widgets_class_2 = gui_elements.create_class_selector(self, columns_rules[1], self.selection.rules[1][0], self.selection.rules[1][4], fig_size=FigureSize.v_model)
                    if len(columns_rules) > 2:
                        self.__all_widgets_class_3 = gui_elements.create_class_selector(self, columns_rules[2], self.selection.rules[2][0], self.selection.rules[2][4], fig_size=FigureSize.v_model)

                    for ii in range(len(self.__all_widgets_class_1.children[2].children)):
                        self.__all_widgets_class_1.children[2].children[ii].on_event("change", change_continuous1)

                    for ii in range(len(self.__all_widgets_class_2.children[2].children)):
                        self.__all_widgets_class_2.children[2].children[ii].on_event("change", change_continuous2)

                    for ii in range(len(self.__all_widgets_class_3.children[2].children)):
                        self.__all_widgets_class_3.children[2].children[ii].on_event("change", change_continuous3)

                    if self.atk.dataset.lat in columns_rules and self.atk.dataset.long in columns_rules:
                        button_add_map.disabled = False
                    else:
                        button_add_map.disabled = True

                    slider_skope1.min = -10e10
                    slider_skope1.max = 10e10
                    slider_skope2.min = -10e10
                    slider_skope2.max = 10e10
                    slider_skope3.min = -10e10
                    slider_skope3.max = 10e10

                    slider_skope1.max = max(self.atk.dataset.X[columns_rules[0]])
                    slider_skope1.min = min(self.atk.dataset.X[columns_rules[0]])
                    slider_skope1.v_model = [self.selection.rules[0][0], self.selection.rules[0][-1]]
                    [slider_text_comb1.children[0].v_model, slider_text_comb1.children[2].v_model] = [slider_skope1.v_model[0], slider_skope1.v_model[1]]

                    if len(self.selection.rules) > 1 :
                        slider_skope2.max = max(self.atk.dataset.X[columns_rules[1]])
                        slider_skope2.min = min(self.atk.dataset.X[columns_rules[1]])
                        slider_skope2.v_model = [self.selection.rules[1][0], self.selection.rules[1][-1]]
                        [slider_text_comb2.children[0].v_model, slider_text_comb2.children[2].v_model] = [slider_skope2.v_model[0],slider_skope2.v_model[1]]

                    if len(self.selection.rules) > 2:
                        slider_skope3.max = max(self.atk.dataset.X[columns_rules[2]])
                        slider_skope3.min = min(self.atk.dataset.X[columns_rules[2]])
                        slider_skope3.v_model = [self.selection.rules[2][0], self.selection.rules[2][-1]]
                        [
                            slider_text_comb3.children[0].v_model,
                            slider_text_comb3.children[2].v_model,
                        ] = [
                            slider_skope3.v_model[0],
                            slider_skope3.v_model[1],
                        ]

                    with histogram1.batch_update():
                        histogram1.data[0].x = list(self.atk.dataset.X[columns_rules[0]])
                        df_respect1 = self.selection.respectOneRule(0)
                        histogram1.data[1].x = list(df_respect1[columns_rules[0]])
                    if len(set([self.selection.rules[i][2] for i in range(len(self.selection.rules))])) > 1:
                        with histogram2.batch_update():
                            histogram2.data[0].x = list(self.atk.dataset.X[columns_rules[1]])
                            df_respect2 = self.selection.respectOneRule(1)
                            histogram2.data[1].x = list(df_respect2[columns_rules[1]])
                    if len(set([self.selection.rules[i][2] for i in range(len(self.selection.rules))])) > 2:
                        with histogram3.batch_update():
                            histogram3.data[0].x = list(self.atk.dataset.X[columns_rules[2]])
                            df_respect3 = self.selection.respectOneRule(2)
                            histogram3.data[1].x = list(df_respect3[columns_rules[2]])

                    modifie_all_histograms(
                        slider_skope1.v_model[0], slider_skope1.v_model[1], 0
                    )

                    text_skopeES.children[0].children[3].children = [
                        # str(skope_rules_clf.rules_[0])
                        # + "\n"
                        "p = "
                        + str(self.selection.score_exp[0])
                        + "%"
                        + " r = "
                        + str(self.selection.score_exp[1])
                        + "%"
                        + " ext. of the tree ="
                        + str(self.selection.score_exp[2])
                    ]
                    one_card_ES.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules_exp))
                    function_scores_models(self.selection.indexes)

                    in_accordion1_n.children[0].disabled = False
                    in_accordion2_n.children[0].disabled = False
                    in_accordion3_n.children[0].disabled = False

            slider_skope1.on_event("input", on_value_change_skope1)
            slider_skope2.on_event("input", on_value_change_skope2)
            slider_skope3.on_event("input", on_value_change_skope3)

            slider_text_comb1.children[0].on_event("input", on_value_change_skope1)
            slider_text_comb1.children[2].on_event("input", on_value_change_skope1)

            loading_models.class_ = "d-none"

            self.__save_rules = deepcopy(self.selection.rules)

            changeMarkersColor(None)

        def reset_skope(*b):
            self.selection.rules = self.__save_rules
            function_validation_skope(None)
            function_scores_models(None)

        button_reset_skope.on_event("click", reset_skope)

        # here to see the values ​​of the selected points (VS and ES)
        out_selec = v.Layout(style_="min-width: 47%; max-width: 47%", children=[v.Html(tag="h4", children=["Select points on the figure to see their values ​​here"])])
        out_selec_SHAP = v.Layout(style_="min-width: 47%; max-width: 47%", children=[v.Html(tag="h4", children=["Select points on the figure to see their SHAP values ​​here"])])
        out_selec_all = v.Alert(
            max_height="400px",
            style_="overflow: auto",
            elevation="0",
            children=[
                v.Row(class_='d-flex flex-row justify-space-between', children=[out_selec, v.Divider(class_="ma-2", vertical=True), out_selec_SHAP]),
            ],
        )

        # to see the data of the current selection
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

        find_clusters = v.Btn(
            class_="ma-1 mt-2 mb-0",
            elevation="2",
            children=[v.Icon(children=["mdi-magnify"]), "Find clusters"],
        )

        slider_clusters = v.Slider(
            style_="width : 30%",
            class_="ma-3 mb-0",
            min=2,
            max=20,
            step=1,
            v_model=3,
            disabled=True,
        )

        text_slider_cluster = v.Html(
            tag="h3",
            class_="ma-3 mb-0",
            children=["Number of clusters " + str(slider_clusters.v_model)],
        )

        def fonct_text_clusters(*b):
            text_slider_cluster.children = [
                "Number of clusters " + str(slider_clusters.v_model)
            ]

        slider_clusters.on_event("input", fonct_text_clusters)

        check_number_of_clusters = v.Checkbox(
            v_model=True, label="Optimal number of clusters :", class_="ma-3"
        )

        def bool_nb_opti(*b):
            slider_clusters.disabled = check_number_of_clusters.v_model

        check_number_of_clusters.on_event("change", bool_nb_opti)

        part_for_clusters = v.Layout(
            class_="d-flex flex-row",
            children=[
                find_clusters,
                check_number_of_clusters,
                slider_clusters,
                text_slider_cluster,
            ],
        )

        new_df = pd.DataFrame([], columns=["Region #", "Number of points"])
        columns = [{"text": c, "sortable": True, "value": c} for c in new_df.columns]
        results_clusters_table = v.DataTable(
            class_="w-100",
            style_="width : 100%",
            v_model=[],
            show_select=False,
            headers=columns,
            items=new_df.to_dict("records"),
            item_value="Region #",
            item_key="Region #",
            hide_default_footer=True,
        )
        results_clusters = v.Row(
            children=[
                v.Layout(
                    class_="flex-grow-0 flex-shrink-0",
                    children=[v.Btn(class_="d-none", elevation=0, disabled=True)],
                ),
                v.Layout(
                    class_="flex-grow-1 flex-shrink-0",
                    children=[results_clusters_table],
                ),
            ],
        )

        # allows you to make dyadic clustering
        def function_clusters(*b):
            loading_clusters.class_ = "d-flex"
            if check_number_of_clusters.v_model:
                result = function_auto_clustering(self.atk.dataset.X_scaled, self.atk.explain[self.__explanation], 3, True)
            else:
                nb_clusters = slider_clusters.v_model
                result = function_auto_clustering(self.atk.dataset.X_scaled, self.atk.explain[self.__explanation], nb_clusters, False)
            self.__result_dyadic_clustering = result
            labels = result[1]
            self.__labels_automatic_clustering = labels
            with self.fig1.batch_update():
                self.fig1.data[0].marker.color = labels
                self.fig1.update_traces(marker=dict(showscale=False))
            with self.fig2.batch_update():
                self.fig2.data[0].marker.color = labels
            with self.fig1_3D.batch_update():
                self.fig1_3D.data[0].marker.color = labels
                self.fig1_3D.update_traces(marker=dict(showscale=False))
            with self.fig2_3D.batch_update():
                self.fig2_3D.data[0].marker.color = labels
            labels_regions = result[0]
            new_df = []
            for i in range(len(labels_regions)):
                new_df.append(
                    [
                        i + 1,
                        len(labels_regions[i]),
                        str(round(len(labels_regions[i]) / len(self.atk.dataset.X) * 100, 2)) + "%",
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
            results_clusters_table = v.DataTable(
                class_="w-100",
                style_="width : 100%",
                show_select=False,
                single_select=True,
                v_model=[],
                headers=columns,
                items=data,
                item_value="Region #",
                item_key="Region #",
                hide_default_footer=True,
            )
            all_chips = []
            all_radio = []
            N_stages = len(labels_regions)
            Multip = 100
            debut = 0
            fin = (N_stages * Multip - 1) * (1 + 1 / (N_stages - 1))
            pas = (N_stages * Multip - 1) / (N_stages - 1)
            scale_colors = np.arange(debut, fin, pas)
            a = 0
            for i in scale_colors:
                color = sns.color_palette("viridis", N_stages * Multip).as_hex()[
                    round(i)
                ]
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
            results_clusters = v.Row(
                children=[
                    v.Layout(
                        class_="flex-grow-0 flex-shrink-0", children=[part_for_radio]
                    ),
                    v.Layout(
                        class_="flex-grow-0 flex-shrink-0", children=[part_for_chips]
                    ),
                    v.Layout(
                        class_="flex-grow-1 flex-shrink-0",
                        children=[results_clusters_table],
                    ),
                ],
            )
            part_for_selection.children = part_for_selection.children[:-1] + [
                results_clusters
            ]

            colorSelectionBtns.v_model = "Clustering auto"

            part_for_selection.children[-1].children[0].children[0].on_event(
                "change", function_choice_cluster
            )
            loading_clusters.class_ = "d-none"
            return N_stages

        # when we select a region created by the automatic dyadic clustering
        def function_choice_cluster(widget, event, data):
            result = self.__result_dyadic_clustering
            labels = result[1]
            index = part_for_selection.children[-1].children[0].children[0].v_model
            liste = [i for i, d in enumerate(labels) if d == float(index)]
            function_lasso_selection(None, None, None, liste)
            colorSelectionBtns.v_model = "Clustering auto"
            changeMarkersColor(opacity=False)

        find_clusters.on_event("click", function_clusters)

        # function which is called as soon as the points are selected (step 1)
        def function_lasso_selection(trace, points, selector, *args):
            if len(args) > 0:
                liste = args[0]
                les_points = liste
            else:
                les_points = points.point_inds
            self.selection = Potato(self.atk, les_points)
            self.selection.state = Potato.LASSO
            if len(les_points) == 0:
                card_selec.children[0].children[1].children = "0 point !"
                text_selec.value = text_base_debut
                return
            card_selec.children[0].children[1].children = (
                str(len(les_points))
                + " points selected ("
                + str(round(len(les_points) / len(self.atk.dataset.X) * 100, 2))
                + "% of the overall)"
            )
            text_selec.value = (
                text_base
                + str(len(les_points))
                + " points selected ("
                + str(round(len(les_points) / len(self.atk.dataset.X) * 100, 2))
                + "% of the overall)"
            )
            opa = []
            for i in range(len(self.fig2.data[0].x)):
                if i in les_points:
                    opa.append(1)
                else:
                    opa.append(0.1)
            with self.fig2.batch_update():
                self.fig2.data[0].marker.opacity = opa
            with self.fig1.batch_update():
                self.fig1.data[0].marker.opacity = opa

            X_train = self.atk.dataset.X.copy()

            X_mean = (
                pd.DataFrame(
                    X_train.iloc[self.selection.indexes, :].mean(axis=0).values.reshape(1, -1),
                    columns=X_train.columns,
                )
                .round(2)
                .rename(index={0: "Mean of the selection"})
            )
            X_mean_tot = (
                pd.DataFrame(
                    X_train.mean(axis=0).values.reshape(1, -1), columns=X_train.columns
                )
                .round(2)
                .rename(index={0: "Mean of the whole dataset"})
            )
            X_mean = pd.concat([X_mean, X_mean_tot], axis=0)
            SHAP_mean = (
                pd.DataFrame(
                    self.atk.explain[self.__explanation].iloc[self.selection.indexes, :]
                    .mean(axis=0)
                    .values.reshape(1, -1),
                    columns=self.atk.explain[self.__explanation].columns,
                )
                .round(2)
                .rename(index={0: "Mean of the selection"})
            )
            SHAP_mean_tot = (
                pd.DataFrame(
                    self.atk.explain[self.__explanation].mean(axis=0).values.reshape(1, -1),
                    columns=self.atk.explain[self.__explanation].columns,
                )
                .round(2)
                .rename(index={0: "Mean of the whole dataset"})
            )
            SHAP_mean = pd.concat([SHAP_mean, SHAP_mean_tot], axis=0)

            X_mean.insert(loc=0, column=' ', value=["Mean of the selection", "Mean of the whole dataset"])
            data = X_mean.to_dict("records")
            columns = [
                {"text": c, "sortable": True, "value": c} for c in X_mean.columns
            ]

            out_selec_table_means = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                items=data.copy(),
                hide_default_footer=True,
                disable_sort=True,
            )

            data = X_train.iloc[self.selection.indexes, :].round(3).to_dict("records")
            columns = [
                {"text": c, "sortable": True, "value": c} for c in X_train.columns
            ]

            out_selec_table = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                items=data.copy(),
            )

            out_selec.children = [v.Col(class_= "d-flex flex-column justify-center align-center", children=[
                v.Html(tag="h3", children=["Values Space"]),
                out_selec_table_means,
                v.Divider(class_="ma-6"),
                v.Html(tag="h4", children=["Entire dataset:"], class_="mb-2"),
                out_selec_table
            ])]

            SHAP_mean.insert(loc=0, column=' ', value=["Mean of the selection", "Mean of the whole dataset"])
            data = SHAP_mean.to_dict("records")
            columns = [
                {"text": c, "sortable": True, "value": c} for c in SHAP_mean.columns
            ]

            out_selec_table_means = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                items=data.copy(),
                hide_default_footer=True,
                disable_sort=True,
            )

            data = self.atk.explain[self.__explanation].iloc[self.selection.indexes, :].round(3).to_dict("records")
            columns = [
                {"text": c, "sortable": True, "value": c} for c in self.atk.explain[self.__explanation].columns
            ]

            out_selec_table = v.DataTable(
                v_model=[],
                show_select=False,
                headers=columns.copy(),
                items=data.copy(),
            )

            out_selec_SHAP.children = [v.Col(class_="d-flex flex-column justify-center align-center", children=[
                v.Html(tag="h3", children=["Explanatory Space"]),
                out_selec_table_means,
                v.Divider(class_="ma-6"),
                v.Html(tag="h4", children=["Entire dataset:"], class_="mb-2"),
                out_selec_table
            ])]


        # function that is called when validating a tile to add it to the set of regions
        def function_new_region(*args):
            if len(args) == 0:
                pass
            elif self.selection in self.atk.regions:
                print("AntakIA WARNING: this region is already in the set of regions")
            else:
                self.selection.state = Potato.REGION
                if self.__model_index == None:
                    name_model = None
                    score_model = [1,1,1]
                else:
                    name_model = self.sub_models[self.__model_index].__class__.__name__
                    score_model = self.__score_sub_models[self.__model_index]
                if self.selection.rules == None :
                    return

                self.selection.applyRules()
                new_tuile = deepcopy(self.selection.getIndexes())
                self.selection.sub_model["name"], self.selection.sub_model["score"] = name_model, score_model
                # here we will force so that all the points of the new tile belong only to it: we will modify the existing tiles
                self.atk.regions = conflict_handler(self.atk.regions, new_tuile)
                self.selection.setIndexes(new_tuile)
                self.atk.newRegion(self.selection)
            self.__regionsColor=[0]*len(self.atk.dataset.X)
            for i in range(len(self.__regionsColor)):
                for j in range(len(self.atk.regions)):
                    if i in self.atk.regions[j].indexes:
                        self.__regionsColor[i] = j+1
                        break

            toute_somme = 0
            temp = []
            score_tot = 0
            score_tot_glob = 0
            autre_toute_somme = 0
            for i in range(len(self.atk.regions)):
                if self.atk.regions[i].sub_model["score"] == None:
                    temp.append(
                        [
                            i + 1,
                            len(self.atk.regions[i]),
                            np.round(len(self.atk.regions[i]) / len(self.atk.dataset.X) * 100, 2),
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
                            len(self.atk.regions[i]),
                            np.round(len(self.atk.regions[i]) / len(self.atk.dataset.X) * 100, 2),
                            self.atk.regions[i].sub_model["model"],
                            self.atk.regions[i].sub_model["score"][0],
                            self.atk.regions[i].sub_model["score"][1],
                            str(self.atk.regions[i].sub_model["score"][2]) + "%",
                        ]
                    )
                    score_tot += self.atk.regions[i].sub_model["score"][0] * len(self.atk.regions[i])
                    score_tot_glob += self.atk.regions[i].sub_model["score"][1] * len(
                        self.atk.regions[i]
                    )
                    autre_toute_somme += len(self.atk.regions[i])
                toute_somme += len(self.atk.regions[i])
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
                    np.round(toute_somme / len(self.atk.dataset.X) * 100, 2),
                    "/",
                    score_tot,
                    score_tot_glob,
                    percent,
                ]
            )
            new_df = pd.DataFrame(
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
            with table_regions:
                clear_output()
                data = new_df[:-1].to_dict("records")
                total = new_df[-1:].iloc[:, 1:].to_dict("records")
                columns = [
                    {"text": c, "sortable": True, "value": c} for c in new_df.columns
                ]
                columns_total = [
                    {"text": c, "sortable": True, "value": c}
                    for c in new_df.columns[1:]
                ]
                table_donnes = v.DataTable(
                    v_model=[],
                    show_select=True,
                    headers=columns,
                    items=data,
                    item_value="Region #",
                    item_key="Region #",
                )
                table_total = v.DataTable(
                    v_model=[],
                    headers=columns_total,
                    items=total,
                    hide_default_footer=True,
                )
                ensemble_tables = v.Layout(
                    class_="d-flex flex-column",
                    children=[
                        table_donnes,
                        v.Divider(class_="mt-7 mb-4"),
                        v.Html(tag="h2", children=["Review of the regions :"]),
                        table_total,
                    ],
                )

                def function_suppression_tuiles(*b):
                    if table_donnes.v_model == []:
                        return
                    taille = len(table_donnes.v_model)
                    a = 0
                    for i in range(taille):
                        index = table_donnes.v_model[i]["Region #"] - 1
                        self.atk.regions.pop(index - a)
                        function_new_region()
                        a += 1
                    colorSelectionBtns.v_model = "Régions"
                    changeMarkersColor()

                supprimer_toutes_les_tuiles.on_event(
                    "click", function_suppression_tuiles
                )

                display(ensemble_tables)

        validate_one_region.on_event("click", function_new_region)
        button_validate_skope.on_event("click", function_validation_skope)

        self.fig1.data[0].on_selection(function_lasso_selection)
        self.fig2.data[0].on_selection(function_lasso_selection)

        def function_fig_size(*args):
            with self.fig1.batch_update():
                self.fig1.layout.width = int(FigureSize.v_model)
            with self.fig2.batch_update():
                self.fig2.layout.width = int(FigureSize.v_model)
            with self.fig1_3D.batch_update():
                self.fig1_3D.layout.width = int(FigureSize.v_model)
            with self.fig2_3D.batch_update():
                self.fig2_3D.layout.width = int(FigureSize.v_model)
            for i in range(len(all_histograms)):
                with all_histograms[i].batch_update():
                    all_histograms[i].layout.width = 0.9 * int(FigureSize.v_model)
                with all_beeswarms[i].batch_update():
                    all_beeswarms[i].layout.width = 0.9 * int(FigureSize.v_model)

        FigureSize.on_event("input", function_fig_size)

        button_add_skope = v.Btn(
            class_="ma-4 pa-2 mb-1",
            children=[v.Icon(children=["mdi-plus"]), "Add a rule"],
        )

        widget_list_add_skope = v.Select(
            class_="mr-3 mb-0",
            items=["/"],
            v_model="/",
            style_="max-width : 15%",
        )

        button_add_map = v.Btn(
            class_="ma-4 pa-2 mb-1",
            children=[v.Icon(class_="mr-4", children=["mdi-map"]), "Display the map"],
            color="white",
            disabled=True,
        )

        def function_display_map(widget, event, data):
            if widget.color == "white":
                part_map.class_= "d-flex justify-space-around ma-0 pa-0"
                widget.color = "error"
                widget.children =  [widget.children[0]] + ["Hide the map"]
                self.__save_lat_rule = [self.selection.rules[i] for i in range(len(self.selection.rules)) if self.selection.rules[i][2] == self.atk.dataset.lat]
                self.__save_long_rule = [self.selection.rules[i] for i in range(len(self.selection.rules)) if self.selection.rules[i][2] == self.atk.dataset.long]
                count = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i-count][2] == self.atk.dataset.lat or self.selection.rules[i-count][2] == self.atk.dataset.long:
                        self.selection.rules.pop(i-count)
                        count += 1
                for i in range(len(accordion_skope.children)):
                    if accordion_skope.children[i].children[0].children[0].children[0][4:-1] in [self.atk.dataset.lat, self.atk.dataset.long]:
                        accordion_skope.children[i].disabled = True
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()
            else:
                self.selection.setIndexesFromMap(None)
                widget.color = "white"
                part_map.class_= "d-none ma-0 pa-0"
                widget.children =  [widget.children[0]] + ["Display the map"]
                self.selection.rules = self.selection.rules + self.__save_lat_rule + self.__save_long_rule
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()
                for i in range(len(accordion_skope.children)):
                    accordion_skope.children[i].disabled = False
            
            

        button_add_map.on_event("click", function_display_map)

        add_group = v.Row(children=[button_add_skope, widget_list_add_skope, v.Spacer(), button_add_map])

        # function called when we add a feature to the rules. We instanciate the exact same things than from the previous features
        # (beeswarms, histograms, etc...)
        def function_add_skope(*b):
            new_rule = [0] * 5
            column = widget_list_add_skope.v_model
            if self.__other_columns == None:
                return
            self.__other_columns = [a for a in self.__other_columns if a != column]
            new_rule[2] = column
            new_rule[0] = round(min(list(self.atk.dataset.X[column].values)), 1)
            new_rule[1] = "<="
            new_rule[3] = "<="
            new_rule[4] = round(max(list(self.atk.dataset.X[column].values)), 1)
            self.selection.rules.append(new_rule)
            one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))

            new_validate_change, new_slider_skope, new_histogram = gui_elements.create_new_feature_rule(self, new_rule, column, number_of_bins_histograms, FigureSize.v_model)

            all_histograms.append(new_histogram)

            def new_function_change_validate(*change):
                ii = -1
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i][2] == column_2:
                        ii = int(i)
                a = deepcopy(float(new_slider_skope.v_model[0]  ))
                b = deepcopy(float(new_slider_skope.v_model[1]  ))
                self.selection.rules[ii][0] = a
                self.selection.rules[ii][4] = b
                self.selection.rules[ii][0] = a
                self.selection.rules[ii][4] = b
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                update_all_graphs()
                function_scores_models(None)

            new_validate_change.on_event("click", new_function_change_validate)

            new_button_in_real_time_graph = v.Checkbox(
                v_model=False, label="Real-time updates on the graphs", class_="ma-3"
            )

            new_slider_text_comb = v.Layout(
                children=[
                    v.TextField(
                        style_="max-width:100px",
                        v_model=new_slider_skope.v_model[0]  ,
                        hide_details=True,
                        type="number",
                        density="compact",
                    ),
                    new_slider_skope,
                    v.TextField(
                        style_="max-width:100px",
                        v_model=new_slider_skope.v_model[1]  ,
                        hide_details=True,
                        type="number",
                        density="compact",
                    ),
                ],
            )

            def new_update_validate(*args):
                if new_button_in_real_time_graph.v_model:
                    new_validate_change.disabled = True
                else:
                    new_validate_change.disabled = False

            new_button_in_real_time_graph.on_event("change", new_update_validate)

            new_two_end = widgets.HBox([new_validate_change, new_button_in_real_time_graph])

            new_all_widgets_slider_histo = widgets.VBox(
                [new_slider_text_comb, new_histogram, new_two_end]
            )

            column_shap = column + "_shap"
            y_histo_shap = [0] * len(self.atk.explain[self.__explanation])
            new_beeswarm = go.FigureWidget(
                data=[go.Scatter(x=self.atk.explain[self.__explanation][column_shap], y=y_histo_shap, mode="markers")]
            )
            new_beeswarm.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=200,
                width=0.9 * int(FigureSize.v_model),
            )
            new_beeswarm.update_yaxes(visible=False, showticklabels=False)
            [new_y, marker] = compute.function_beeswarm_shap(self, self.__explanation, column)
            new_beeswarm.data[0].y = new_y
            new_beeswarm.data[0].x = self.atk.explain[self.__explanation][column_shap]
            new_beeswarm.data[0].marker = marker

            new_choice_color_beeswarm = v.Row(
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

            def new_change_color_beeswarm_shap(*args):
                if new_choice_color_beeswarm.children[1].v_model == False:
                    marker = compute.function_beeswarm_shap(self, self.__explanation, self.selection.rules[len(self.selection.rules) - 1][2])[1]
                    new_beeswarm.data[0].marker = marker
                    new_beeswarm.update_traces(marker=dict(showscale=True))
                else:
                    modifie_all_histograms(
                        new_slider_skope.v_model[0]  ,
                        new_slider_skope.v_model[1]  ,
                        0,
                    )
                    new_beeswarm.update_traces(marker=dict(showscale=False))

            new_choice_color_beeswarm.children[1].on_event(
                "change", new_change_color_beeswarm_shap
            )

            new_beeswarm_tot = widgets.VBox([new_choice_color_beeswarm, new_beeswarm])
            new_beeswarm_tot.layout.margin = "0px 0px 0px 20px"

            all_beeswarms_total.append(new_beeswarm_tot)

            if not check_beeswarm.v_model:
                new_beeswarm_tot.layout.display = "none"

            all_beeswarms.append(new_beeswarm)

            all_color_choosers_beeswarms.append(new_choice_color_beeswarm)

            widget_list_add_skope.items = self.__other_columns
            widget_list_add_skope.v_model = self.__other_columns[0]

            new_b_delete_skope = v.Btn(
                color="error",
                class_="ma-2 ml-4 pa-1",
                elevation="3",
                icon=True,
                children=[v.Icon(children=["mdi-delete"])],
            )

            def new_delete_skope(*b):
                column_2 = new_slider_skope.label
                ii = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i][2] == column_2:
                        ii = i
                        break
                all_beeswarms_total.pop(ii)
                all_histograms.pop(ii)
                self.selection.rules.pop(ii)
                all_beeswarms.pop(ii)
                all_color_choosers_beeswarms.pop(ii)
                self.__other_columns = [column_2] + self.__other_columns
                one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                widget_list_add_skope.items = self.__other_columns
                widget_list_add_skope.v_model = self.__other_columns[0]
                accordion_skope.children = [
                    a for a in accordion_skope.children if a != new_in_accordion_n
                ]
                for i in range(ii, len([accordion_skope.children[a] for a in range(len(accordion_skope.children)) if accordion_skope.children[a].disabled == False])):
                    col = "X" + str(i + 1) + " (" + self.selection.rules[i][2] + ")"
                    accordion_skope.children[i].children[0].children[0].children = [col]

                if widget_list_add_skope.v_model in [self.atk.dataset.lat, self.atk.dataset.long]:
                    if self.atk.dataset.lat in [self.selection.rules[i][2] for i in range(len(self.selection.rules))] and self.atk.dataset.long in [self.selection.rules[i][2] for i in range(len(self.selection.rules))]:
                        button_add_map.disabled = False
                    else :
                        button_add_map.disabled = True
                update_all_graphs()

            new_b_delete_skope.on_event("click", new_delete_skope)

            is_continuous_new = v.Checkbox(v_model=True, label="is continuous?")

            right_side_new = v.Col(children=[new_b_delete_skope, is_continuous_new], class_="d-flex flex-column align-center justify-center")

            all_widgets_class_new = gui_elements.create_class_selector(self, self.selection.rules[-1][2], min=min(list(self.atk.dataset.X[new_slider_skope.label].values)), max=max(list(self.atk.dataset.X[new_slider_skope.label].values)), fig_size=FigureSize.v_model)

            def change_continuous_new(widget, event, data):
                column_2 = new_slider_skope.label
                index = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i][2] == column_2:
                        index = i
                        break
                if widget.v_model == True and widget == right_side_new.children[1]:
                    new_in_accordion.children = [new_all_widgets_slider_histo] + list(new_in_accordion.children[1:])
                    count = 0
                    for i in range(len(self.selection.rules)):
                        if self.selection.rules[i-count][2] == self.selection.rules[index][2] and i-count != index:
                            self.selection.rules.pop(i-count)
                            count += 1
                    self.selection.rules[index][0] = new_slider_skope.v_model[0]
                    self.selection.rules[index][4] = new_slider_skope.v_model[1]
                    one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
                    update_all_graphs()
                else:
                    new_in_accordion.children = [all_widgets_class_new] + list(new_in_accordion.children[1:])
                    l = []
                    for i in range(len(all_widgets_class_new.children[2].children)):
                        if all_widgets_class_new.children[2].children[i].v_model:
                            l.append(int(all_widgets_class_new.children[2].children[i].label))
                    if len(l) == 0:
                        widget.v_model = True
                        return
                    column = deepcopy(self.selection.rules[index][2])
                    count = 0
                    for i in range(len(self.selection.rules)):
                        if self.selection.rules[i-count][2] == column:
                            self.selection.rules.pop(i-count)
                            count += 1
                    croissant = 0
                    for ele in l:
                        self.selection.rules.insert(index+croissant, [ele-0.5, '<=', column, '<=', ele+0.5])
                        croissant += 1
                    one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules), is_class=True)
                    update_all_graphs()

            right_side_new.children[1].on_event("change", change_continuous_new)

            for ii in range(len(all_widgets_class_new.children[2].children)):
                all_widgets_class_new.children[2].children[ii].on_event("change", change_continuous_new)

            new_in_accordion = widgets.HBox(
                [new_all_widgets_slider_histo, new_beeswarm_tot, right_side_new],
                layout=Layout(align_items="center"),
            )

            new_in_accordion_n = v.ExpansionPanels(
                class_="ma-2 mb-1",
                children=[
                    v.ExpansionPanel(
                        children=[
                            v.ExpansionPanelHeader(children=["Xn"]),
                            v.ExpansionPanelContent(children=[new_in_accordion]),
                        ]
                    )
                ],
            )

            accordion_skope.children = [*accordion_skope.children, new_in_accordion_n]
            name_colcol = "X" + str(len(accordion_skope.children)) + " (" + column + ")"
            accordion_skope.children[-1].children[0].children[0].children = name_colcol

            with new_histogram.batch_update():
                new_list = [
                    g
                    for g in list(self.atk.dataset.X[column].values)
                    if g >= new_slider_skope.v_model[0]  
                    and g <= new_slider_skope.v_model[1]  
                ]
                new_histogram.data[1].x = new_list

                column_2 = new_slider_skope.label
                new_list_rule = self.atk.dataset.X.index[
                    self.atk.dataset.X[column_2].between(
                        new_slider_skope.v_model[0]  ,
                        new_slider_skope.v_model[1]  ,
                    )
                ].tolist()
                new_list_tout = new_list_rule.copy()
                for i in range(1, len(self.selection.rules)):
                    new_list_temp = self.atk.dataset.X.index[
                        self.atk.dataset.X[self.selection.rules[i][2]].between(
                            self.selection.rules[i][0], self.selection.rules[i][4]
                        )
                    ].tolist()
                    new_list_tout = [g for g in new_list_tout if g in new_list_temp]
                new_list_tout_new = self.atk.dataset.X[column_2][new_list_tout]
                new_histogram.data[2].x = new_list_tout_new

            def new_on_value_change_skope(*b1):
                new_slider_text_comb.children[0].v_model = (
                    new_slider_skope.v_model[0]  
                )
                new_slider_text_comb.children[2].v_model = (
                    new_slider_skope.v_model[1]  
                )
                column_2 = new_slider_skope.label
                ii = 0
                for i in range(len(self.selection.rules)):
                    if self.selection.rules[i][2] == column_2:
                        ii = i
                        break
                new_list = [
                    g
                    for g in list(self.atk.dataset.X[column_2].values)
                    if g >= new_slider_skope.v_model[0]  
                    and g <= new_slider_skope.v_model[1]  
                ]
                with new_histogram.batch_update():
                    new_histogram.data[1].x = new_list
                if self.__activate_histograms:
                    modifie_all_histograms(
                        new_slider_skope.v_model[0]  ,
                        new_slider_skope.v_model[1]  ,
                        ii,
                    )
                if new_button_in_real_time_graph.v_model:
                    self.selection.rules[ii - 1][0] = float(
                        deepcopy(new_slider_skope.v_model[0]  )
                    )
                    self.selection.rules[ii - 1][4] = float(
                        deepcopy(new_slider_skope.v_model[1]  )
                    )
                    one_card_VS.children = gui_elements.generate_rule_card(
                        liste_to_string_skope(self.selection.rules)
                    )
                    update_all_graphs()

            new_slider_skope.on_event("input", new_on_value_change_skope)

            if new_slider_skope.label in [self.atk.dataset.lat, self.atk.dataset.long]:
                if self.atk.dataset.lat in [self.selection.rules[i][2] for i in range(len(self.selection.rules))] and self.atk.dataset.long in [self.selection.rules[i][2] for i in range(len(self.selection.rules))]:
                    button_add_map.disabled = False
                else :
                    button_add_map.disabled = True

        function_new_region()

        button_add_skope.on_event("click", function_add_skope)

        param_VS = gui_elements.create_settings_card(paramsProjVSGroup, "Settings of the projection in the Values Space")
        param_ES = gui_elements.create_settings_card(paramsProjESGroup, "Settings of the projection in the Explanatory Space")

        projVS_and_load = widgets.HBox(
            [
                dimReducVSDropdown,
                v.Layout(children=[param_VS]),
                computeIndicatorVS,
            ]
        )
        projES_and_load = widgets.HBox(
            [
                dimReducESDropdown,
                v.Layout(children=[param_ES]),
                computeIndicatorES,
            ]
        )

        bouton_reset_opa = v.Btn(
            icon=True,
            children=[v.Icon(children=["mdi-opacity"])],
            class_="ma-2 ml-6 pa-3",
            elevation="3",
        )

        bouton_reset_opa.children = [
            add_tooltip(
                bouton_reset_opa.children[0],
                "Reset the opacity of the points",
            )
        ]

        def function_reset_opa(*args):
            with self.fig1.batch_update():
                self.fig1.data[0].marker.opacity = 1
            with self.fig2.batch_update():
                self.fig2.data[0].marker.opacity = 1

        bouton_reset_opa.on_event("click", function_reset_opa)

        items = [{'text': "Imported", 'disabled': True},
                {'text': "SHAP", 'disabled': True},
                {'text': "LIME", 'disabled': True}]
        
        for item in items:
            if self.atk.explain[item['text']] is not None:
                item_default = item['text']
                item['disabled'] = False

        choose_explanation = v.Select(
            label="Explainability method",
            items=items,
            v_model=item_default,
            class_="ma-2 mt-1 ml-6",
            style_="width: 150px",
            disabled = False,
        )

        def function_choose_explanation(widget, event, data):
            self.__explanation = data
            exp_val = self.atk.explain[data]
            if self.dim_red['ES'][self.__explanation][self.__projectionES] == None:
                computeIndicatorES.layout.visibility = "visible"
                dim_red = compute.DimensionalityReductionChooser(method=dimReducESDropdown.v_model)
                self.dim_red['ES'][self.__explanation][self.__projectionES] = [dim_red.compute(exp_val, 2), dim_red.compute(exp_val, 3)]
                computeIndicatorES.layout.visibility = "hidden"
            compute.update_figures(self, self.__explanation, self.__projectionVS, self.__projectionES)

        choose_explanation.on_event("change", function_choose_explanation)
        
        new_prog_SHAP = gui_elements.prog_other("SHAP")
        new_prog_LIME = gui_elements.prog_other("LIME")

        if self.__calculus == True:
            if self.__explanation == "SHAP":
                new_prog_SHAP.children[1].v_model = 100
                new_prog_SHAP.children[2].v_model = "Computations already done !"
                new_prog_SHAP.children[-1].disabled = True
            elif self.__explanation == "LIME":
                new_prog_LIME.children[1].v_model = 100
                new_prog_LIME.children[2].v_model = "Computations already done !"
                new_prog_LIME.children[-1].disabled = True

        def function_validation_explanation(widget, event, data):
            if widget.v_model == "SHAP":
                self.__compute_SHAP = compute.computationSHAP(self.atk.dataset.X, self.atk.dataset.X_all, self.atk.dataset.model)
                widgets.jslink((new_prog_SHAP.children[1], "v_model"), (self.__compute_SHAP.progress_widget, "v_model"))
                widgets.jslink((new_prog_SHAP.children[2], "v_model"), (self.__compute_SHAP.text_widget, "v_model"))
                widgets.jslink((new_prog_SHAP.children[-1], "color"), (self.__compute_SHAP.done_widget, "v_model"))
                self.__compute_SHAP.compute_in_thread()
                new_prog_SHAP.children[-1].disabled = True
            if widget.v_model == "LIME":
                self.__compute_LIME = compute.computationLIME(self.atk.dataset.X, self.atk.dataset.X_all, self.atk.dataset.model)
                widgets.jslink((new_prog_LIME.children[1], "v_model"), (self.__compute_LIME.progress_widget, "v_model"))
                widgets.jslink((new_prog_LIME.children[2], "v_model"), (self.__compute_LIME.text_widget, "v_model"))
                widgets.jslink((new_prog_LIME.children[-1], "color"), (self.__compute_LIME.done_widget, "v_model"))
                self.__compute_LIME.compute_in_thread()
                new_prog_LIME.children[-1].disabled = True
                
        def when_SHAP_computation_is_done(*args):
            self.atk.explain["SHAP"] = self.__compute_SHAP.value
            items = choose_explanation.items.copy()
            for item in items:
                if item['text'] == "SHAP":
                    item['disabled'] = False
            choose_explanation.items = items.copy() + [{'text': "Update", 'disabled': False}]
            choose_explanation.items = choose_explanation.items[:-1]

        def when_LIME_computation_is_done(*args):
            self.atk.explain["LIME"] = self.__compute_LIME.value
            items = choose_explanation.items.copy()
            for item in items:
                if item['text'] == "LIME":
                    item['disabled'] = False
            choose_explanation.items = items.copy() + [{'text': "Update", 'disabled': False}]
            choose_explanation.items = choose_explanation.items[:-1]

        new_prog_SHAP.children[-1].observe(when_SHAP_computation_is_done, "color")
        new_prog_LIME.children[-1].observe(when_LIME_computation_is_done, "color")

        new_prog_SHAP.children[-1].on_event("click", function_validation_explanation)
        new_prog_LIME.children[-1].on_event("click", function_validation_explanation)

        time_computing = gui_elements.time_computing(new_prog_SHAP, new_prog_LIME)

        widgets.jslink(
            (time_computing.children[0], "v_model"), (time_computing.children[1].children[0], "v_model")
        )

        choose_computing = v.Menu(
            v_slots=[
                {
                    "name": "activator",
                    "variable": "props",
                    "children": v.Btn(
                        v_on="props.on",
                        icon=True,
                        size="x-large",
                        children=[add_tooltip(v.Icon(children=["mdi-timer-sand"], size="large"), "Time of computing")],
                        class_="ma-2 pa-3",
                        elevation="3",
                    ),
                }
            ],
            children=[
                v.Card(
                    class_="pa-4",
                    rounded=True,
                    children=[time_computing],
                    min_width="500",
                )
            ],
            v_model=False,
            close_on_content_click=False,
            offset_y=True,
        )
            

        top_of_the_UI_all_buttons = widgets.HBox(
            [
                dimension_projection_text,
                v.Layout(
                    class_="pa-2 ma-2",
                    elevation="3",
                    children=[
                        add_tooltip(
                            v.Icon(
                                children=["mdi-format-color-fill"],
                                class_="mt-n5 mr-4"
                            ),
                            "Color of the points",
                        ),
                        colorSelectionBtns,
                        bouton_reset_opa,
                        choose_explanation,
                        choose_computing,
                    ],
                ),
                v.Layout(class_="mt-3", children=[projVS_and_load, projES_and_load]),
            ],
            layout=Layout(
                width="100%",
                display="flex",
                flex_flow="row",
                justify_content="space-around",
            ),
        )
        figures = widgets.VBox([currentView], layout=Layout(width="100%"))

        check_beeswarm = v.Checkbox(
            v_model=True,
            label="Show Shapley's beeswarm plots",
            class_="ma-1 mr-3",
        )

        def function_check_beeswarm(*b):
            if not check_beeswarm.v_model:
                for i in range(len(all_beeswarms_total)):
                    all_beeswarms_total[i].layout.display = "none"
            else:
                for i in range(len(all_beeswarms_total)):
                    all_beeswarms_total[i].layout.display = "block"

        check_beeswarm.on_event("change", function_check_beeswarm)

        buttons_skope = v.Layout(
            class_="d-flex flex-row",
            children=[
                button_validate_skope,
                button_reset_skope,
                v.Spacer(),
                check_beeswarm,
            ],
        )

        two_buttons = widgets.VBox([buttons_skope, text_skope])

        bouton_magic = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(children=["mdi-creation"], class_="mr-3"),
                "Magic button",
            ],
        )

        part_for_magic = v.Layout(
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

        def function_checkbox_magic(*args):
            if part_for_magic.children[2].v_model:
                part_for_magic.children[3].disabled = False
            else:
                part_for_magic.children[3].disabled = True

        part_for_magic.children[2].on_event("change", function_checkbox_magic)

        def find_best_score():
            a = 1000
            for i in range(len(self.__score_sub_models)):
                score = self.__score_sub_models[i][0]
                if score < a:
                    a = score
                    index = i
            return index

        def function_bouton_magic(*args):
            demo = part_for_magic.children[2].v_model
            if demo == False:
                stages.children[0].v_model = 3
            N_stages = function_clusters(None)
            if demo:
                tempo = int(part_for_magic.children[3].v_model) / 10
                if tempo < 0:
                    tempo = 0
            else:
                tempo = 0
            time.sleep(tempo)
            for i in range(N_stages):
                part_for_selection.children[-1].children[0].children[0].v_model = str(i)
                function_choice_cluster(None, None, None)
                time.sleep(tempo)
                if demo:
                    stages.children[0].v_model = 1
                time.sleep(tempo)
                function_validation_skope(None)
                time.sleep(tempo)
                if demo:
                    stages.children[0].v_model = 2
                time.sleep(tempo)
                index = find_best_score()
                mods.children[index].children[0].color = "blue lighten-4"
                change(None, None, None, False)
                time.sleep(tempo)
                mods.children[index].children[0].color = "white"
                if demo:
                    stages.children[0].v_model = 3
                time.sleep(tempo)
                function_new_region(None)
                time.sleep(tempo)
                if i != N_stages - 1:
                    if demo:
                        stages.children[0].v_model = 0
                    time.sleep(tempo)
            colorSelectionBtns.v_model = "Régions"
            changeMarkersColor(None)

        #map plotly
        map_select = go.FigureWidget(
            data=go.Scatter(x=[1], y=[1], mode="markers", marker=marker1, customdata=marker1["color"], hovertemplate = '%{customdata:.3f}')
        )

        map_select.update_layout(dragmode="lasso")

        # instanciate the map, with longitude and latitude
        if self.atk.dataset.lat is not None and self.atk.dataset.long is not None:
            df = self.atk.dataset.X
            data=go.Scattergeo(
                lon = df[self.atk.dataset.long],
                lat = df[self.atk.dataset.lat],
                mode = 'markers',
                marker_color = self.atk.dataset.y,
                )
            map_select = go.FigureWidget(
            data=data
            )
            lat_center = max(df[self.atk.dataset.lat]) - (max(df[self.atk.dataset.lat]) - min(df[self.atk.dataset.lat]))/2
            long_center = max(df[self.atk.dataset.long]) - (max(df[self.atk.dataset.long]) - min(df[self.atk.dataset.long]))/2
            map_select.update_layout(
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
            
        map_text_selection = v.Card(
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

        def change_text_map(trace, points, selector):
            map_text_selection.children[1].children[0].children = ["Number of entries selected: " + str(len(points.point_inds))]
            self.selection.setIndexesFromMap(points.point_inds)
            one_card_VS.children = gui_elements.generate_rule_card(liste_to_string_skope(self.selection.rules))
            update_all_graphs()

        map_select.data[0].on_selection(change_text_map)

        part_map = v.Layout(
            class_="d-none ma-0 pa-0",
            children=[map_select, map_text_selection]
        )

        bouton_magic.on_event("click", function_bouton_magic)

        loading_clusters = v.ProgressLinear(
            indeterminate=True, class_="ma-3", style_="width : 100%"
        )

        loading_clusters.class_ = "d-none"

        part_for_selection = v.Col(
            children=[
                card_selec,
                out_accordion,
                part_for_clusters,
                loading_clusters,
                results_clusters,
            ]
        )

        loading_models = v.ProgressLinear(
            indeterminate=True,
            class_="my-0 mx-15",
            style_="width: 100%;",
            color="primary",
            height="5",
        )

        loading_models.class_ = "d-none"

        part_for_skope = v.Col(children=[two_buttons, accordion_skope, add_group, part_map])
        part_for_modele = widgets.VBox([loading_models, mods])
        part_for_toutes_regions = widgets.VBox([selection, table_regions])

        stages = v.Card(
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
                        v.Tab(value="four", children=["4. Overview of the regions"]),
                    ],
                ),
                v.CardText(
                    class_="w-100",
                    children=[
                        v.Window(
                            class_="w-100",
                            v_model="tabs",
                            children=[
                                v.WindowItem(value=0, children=[part_for_selection]),
                                v.WindowItem(value=1, children=[part_for_skope]),
                                v.WindowItem(value=2, children=[part_for_modele]),
                                v.WindowItem(value=3, children=[part_for_toutes_regions]),
                            ],
                        )
                    ],
                ),
            ],
        )

        widgets.jslink(
            (stages.children[0], "v_model"), (stages.children[1].children[0], "v_model")
        )

        part_for_data = widgets.VBox(
            [
                barMenu,
                backupDialogue,
                top_of_the_UI_all_buttons,
                figures,
                stages,
                part_for_magic,
            ],
            layout=Layout(width="100%"),
        )

        display(part_for_data)
        #return part_for_data

    # --------------- FIN update figues ---------------------

    def results(self, number: int = None, item: str = None):
        L_f = []
        if len(self.atk.regions) == 0:
            return "No region has been created !"
        for i in range(len(self.atk.regions)):
            dictio = dict()
            dictio["X"] = self.atk.dataset.X.iloc[self.atk.regions[i].indexes, :].reset_index(
                drop=True
            )
            dictio["y"] = self.atk.dataset.y.iloc[self.atk.regions[i].indexes].reset_index(drop=True)
            dictio["indexs"] = self.atk.regions[i].indexes
            dictio["explain"] = {"Imported": None, "SHAP": None, "LIME": None}
            if self.atk.explain["Imported"] is not None:
                dictio["explain"]["Imported"] = self.atk.explain["Imported"].iloc[self.atk.regions[i].indexes, :].reset_index(
                    drop=True
                )
            if self.atk.explain["LIME"] is not None:
                dictio["explain"]["LIME"] = self.atk.explain["LIME"].iloc[self.atk.regions[i].indexes, :].reset_index(
                    drop=True
                )
            if self.atk.explain["SHAP"] is not None:
                dictio["explain"]["SHAP"] = self.atk.explain["SHAP"].iloc[self.atk.regions[i].indexes, :].reset_index(
                    drop=True
                )
            if self.atk.regions[i].sub_model == None:
                dictio["model name"] = None
                dictio["model score"] = None
                dictio["model"] = None
            else:
                dictio["model name"] =  self.atk.regions[i].sub_model["name"].__name__
                dictio["model score"] = self.atk.regions[i].sub_model["score"]
                dictio["model"] = self.atk.regions[i].sub_model["name"]
            dictio["rules"] = self.atk.regions[i].rules
            L_f.append(dictio)
        if number == None or item == None:
            return L_f
        else:
            return L_f[number][item]
