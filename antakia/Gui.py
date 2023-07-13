#Importations

# Imports Python
import time
from copy import deepcopy
from importlib.resources import files

# Imports warnings
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore")

# Imports data science
import pandas as pd
import numpy as np
from skrules import SkopeRules

# Imports sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Imports pour le GUI
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, clear_output, HTML
import plotly.graph_objects as go
import ipyvuetify as v
import seaborn as sns

# Import internes
from antakia.utils import from_rules
from antakia.utils import create_save
from antakia.utils import load_save
from antakia.utils import fonction_auto_clustering
from antakia.utils import _conflict_handler
from antakia.utils import _add_tooltip as add_tooltip
from antakia.utils import _fonction_models as fonction_models

import antakia._compute as compute

from antakia import LongTask

from antakia import Potato

import antakia._gui_elements as gui_elements

class GUI():
    """
    Gui object.

    Attributes
    -------
    atk : atk object
        The atk object containing the data to explain.
    explanation_values : pandas dataframe
        The dataframe containing the explanations of the model. Can be given by the user (in explanation) or computed by the Gui.
    selection : list
        The list of the indices of the data currently selected by the user (lasso, skope-rules, or else).
    """
    def __init__(
        self,
        atk,
        explanation: str = None,
        projection: str = "PaCMAP",
        sub_models: list = None,
    ):
        """Function that creates the interface.

        Parameters
        ----------
        explanation : str or pandas dataframe
            The type of explanation to display.
            If not computed : string ("SHAP" or "LIME", default : "SHAP")
            If already computed : pandas dataframe containing the explanations
            TODO : les calculs de SHAP ou LIME sont lancés depuis AntakIA en local ou envoyés à un serveur (avec GPU) distant. Je suggère que les fonctions de calcul long implémentent l'interface LongTask avec les métodes (start, update etc.)
        atk.dataset.X_all : pandas dataframe
            The dataframe containing the entire data. It is used to compute the explanations if they are not already computed.
        dprojection : str
            The default projection to display. It can be "PaCMAP", "PCA", "t-SNE" or "UMAP".
            TODO : ça mériterait une interface "Projection" où chaque implémentation fournit son nom via un getProjType par ex. Ces types pourraient être à choisir parmi une liste de constantes (PCA, TSNE, UMAP ...) définies dans l'interface
        self.sub_models : list
            The list of sub-models to choose from for each region created by the user.
        """
        self.atk = atk

        if sub_models is None :
            sub_models = [
            linear_model.LinearRegression(),
            RandomForestRegressor(random_state=9),
            ensemble.GradientBoostingRegressor(random_state=9),
        ]
        
        self.sub_models = sub_models

        # Publique :
        self.selection = Potato(self.atk.X)

        # Privé :
        if explanation is None :
            if self.atk.dataset.explainability["Imported"] is not None:
                explanation = "Imported"
            else :
                explanation = "SHAP"

        self.__projection = projection #string
        self.__explanation = explanation #string

        self.dim_red = {}
        self.dim_red["EV"] = {"PCA": None, "t-SNE": None, "UMAP": None, "PaCMAP": None}
        self.dim_red["EE"] = {}
        self.dim_red["EE"]["Imported"] = {"PCA": None, "t-SNE": None, "UMAP": None, "PaCMAP": None}
        self.dim_red["EE"]["SHAP"] = {"PCA": None, "t-SNE": None, "UMAP": None, "PaCMAP": None}
        self.dim_red["EE"]["LIME"] = {"PCA": None, "t-SNE": None, "UMAP": None, "PaCMAP": None}

        if self.__explanation == "SHAP" and self.atk.dataset.explain["SHAP"] == None :
            self.__calculus = True
        elif self.__explanation == "LIME" and self.atk.dataset.explain["LIME"] == None :
            self.__calculus = True
        else:
            self.__calculus = False

        self.__list_of_regions = []
        self.__list_of_sub_models = []
        self.__color_regions = []
        self.atk.dataset.X.columns = None
        self.__save_rules = None
        self.__other_columns = None
        self.__valider_bool = False
        self.__SHAP_train = None # class Dataset ? Intêret ? À discuter !
        a = [0] * 10
        self.__all_rules = [a, a, a, a, a, a, a, a, a, a]
        self.atk.dataset.X = None
        y = None 
        self.__model_choice = None
        self.__Y_auto = None
        self.__all_tiles_rules = []
        self.__result_dyadic_clustering = None
        self.__score_models = []
        self.__explanatory_values = None
        self.__table_save = None

    def display(self):
        """Function that displays the interface.

        Returns
        -------
        An interface
        """
        liste_red = ["PCA", "t-SNE", "UMAP", "PaCMAP"]

        if self.sub_models != None and len(self.sub_models) > 9:
            raise ValueError("You can enter up to 9 sub-models maximum ! (changes to come)")
        
        # wait screen definition
        data_path = files("antakia.assets").joinpath("logo_antakia.png")

        logo_antakia = widgets.Image(
            value=open(data_path, "rb").read(), layout=Layout(width="230px")
        )

        # waiting screen progress bars definition
        progress_shap = gui_elements.ProgressLinear()

        # EV dimension reduction progress bar
        progress_red = gui_elements.ProgressLinear()

        # consolidation of progress bars and progress texts in a single HBox
        prog_shap = gui_elements.TotalProgress("Computing of explanatory values", progress_shap)
        prog_red = gui_elements.TotalProgress("Computing of dimensions reduction", progress_red)

        # definition of the splash screen which includes all the elements,
        splash = v.Layout(
            class_="d-flex flex-column align-center justify-center",
            children=[logo_antakia, prog_shap, prog_red],
        )

        # we send the splash screen
        display(splash)

        # if we import the explanatory values, the progress bar of this one is at 100
        if not self.__calculus:
            progress_shap.v_model = 100
            prog_shap.children[2].children[
                0
            ].v_model = "Imported explanatory values"
        else :
            if self.__explanation == "SHAP":
                compute_SHAP = LongTask.compute_SHAP(self.atk.dataset.X, self.atk.dataset.X_all, self.atk.dataset.model)
                widgets.jslink((progress_shap, "v_model"), (compute_SHAP.progress_widget, "v_model"))
                widgets.jslink((prog_shap.children[2].children[0], "v_model"), (compute_SHAP.text_widget, "v_model"))
                self.atk.dataset.explain["SHAP"] = compute_SHAP.compute()
            elif self.__explanation == "LIME":
                compute_LIME = LongTask.compute_LIME(self.atk.dataset.X, self.atk.dataset.X_all, self.atk.dataset.model)
                widgets.jslink((progress_shap, "v_model"), (compute_LIME.progress_widget, "v_model"))
                widgets.jslink((prog_shap.children[2].children[0], "v_model"), (compute_LIME.text_widget, "v_model"))
                self.atk.dataset.explain["LIME"] = compute_LIME.compute()

        # definition of the default projection
        # base, we take the PaCMAP projection
        
        choix_init_proj = ["PCA", "t-SNE", "UMAP", "PaCMAP"].index(self.__projection)
        prog_red.children[2].children[0].v_model = "Values space... "
        self.dim_red["EV"][self.__reduction] = compute.initialize_dim_red_EV(self.atk.dataset.X, self.__projection)
        progress_red.v_model = +50
        prog_red.children[2].children[0].v_model = "Values space... Explanatory space..."
        self.dim_red["EE"][self.__explanation][self.__reduction] = compute.initialize_dim_red_EE(self.atk.dataset.explain[self.__explanation], self.__projection)
        progress_red.v_model = +50

        # once all this is done, the splash screen is removed
        splash.class_ = "d-none"

        # loading_bar = widgets.Image(value=img, width=30, height=20)
        loading_bar = v.ProgressCircular(
            indeterminate=True, color="blue", width="6", size="35", class_="mx-4 my-3"
        )
        out_loading1 = widgets.HBox([loading_bar])
        out_loading2 = widgets.HBox([loading_bar])
        out_loading1.layout.visibility = "hidden"
        out_loading2.layout.visibility = "hidden"

        # dropdown allowing to choose the projection in the value space
        EV_proj = v.Select(
            label="Projection in the VS:",
            items=["PCA", "t-SNE", "UMAP", "PaCMAP"],
            style_="width: 150px",
        )

        EV_proj.v_model = EV_proj.items[choix_init_proj]

        # dropdown allowing to choose the projection in the space of the explanations
        EE_proj = v.Select(
            label="Projection in the ES:",
            items=["PCA", "t-SNE", "UMAP", "PaCMAP"],
            style_="width: 150px",
        )

        EE_proj.v_model = EE_proj.items[choix_init_proj]

        # here the sliders of the parameters for the EV!
        slider_param_PaCMAP_voisins_EV = gui_elements.SliderParam(v_model=10, min=5, max=30, step=1, label="Number of neighbors")
        slider_param_PaCMAP_mn_ratio_EV = gui_elements.SliderParam(v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio :")
        slider_param_PaCMAP_fp_ratio_EV = gui_elements.SliderParam(v_model=2, min=0.1, max=5, step=0.1, label="FP ratio :")

        def fonction_update_sliderEV(widget, event, data):
            # function that updates the values ​​when there is a change of sliders in the parameters of PaCMAP for the EV
            if widget.label == "Number of neighbors :":
                slider_param_PaCMAP_voisins_EV.children[1].children = [str(data)]
            elif widget.label == "MN ratio :":
                slider_param_PaCMAP_mn_ratio_EV.children[1].children = [str(data)]
            elif widget.label == "FP ratio :":
                slider_param_PaCMAP_fp_ratio_EV.children[1].children = [str(data)]

        slider_param_PaCMAP_voisins_EV.children[0].on_event(
            "input", fonction_update_sliderEV
        )
        slider_param_PaCMAP_mn_ratio_EV.children[0].on_event(
            "input", fonction_update_sliderEV
        )
        slider_param_PaCMAP_fp_ratio_EV.children[0].on_event(
            "input", fonction_update_sliderEV
        )

        # sliders parametres EV
        tous_sliders_EV = widgets.VBox(
            [
                slider_param_PaCMAP_voisins_EV,
                slider_param_PaCMAP_mn_ratio_EV,
                slider_param_PaCMAP_fp_ratio_EV,
            ],
            layout=Layout(width="100%"),
        )

        valider_params_proj_EV = v.Btn(
            children=[
                v.Icon(left=True, children=["mdi-check"]),
                "Validate",
            ]
        )

        reinit_params_proj_EV = v.Btn(
            class_="ml-4",
            children=[
                v.Icon(left=True, children=["mdi-skip-backward"]),
                "Reset",
            ],
        )

        deux_boutons_params = widgets.HBox(
            [valider_params_proj_EV, reinit_params_proj_EV]
        )
        params_proj_EV = widgets.VBox(
            [tous_sliders_EV, deux_boutons_params], layout=Layout(width="100%")
        )

        # TODO : quitte à séparer GUI en plusieurs modules, on ne pourrait pas organiser le code selon les 4 étapes / onglets ? On gagnerait énormément en lisibilité

        def changement_params_EV(*b):
            # function that updates the projections when changing the parameters of the projection
            n_neighbors = slider_param_PaCMAP_voisins_EV.children[0].v_model
            MN_ratio = slider_param_PaCMAP_mn_ratio_EV.children[0].v_model
            FP_ratio = slider_param_PaCMAP_fp_ratio_EV.children[0].v_model
            if EV_proj.v_model == "PaCMAP":
                out_loading1.layout.visibility = "visible"
                dim_red = compute.DimensionalityReduction("PaCMAP", False, n_neighbors, MN_ratio, FP_ratio)
                self.__Espace_valeurs[3] = dim_red.compute(X, 2)
                self.__Espace_valeurs_3D[3] = dim_red.compute(X, 3)
                out_loading1.layout.visibility = "hidden"
            with fig1.batch_update():
                fig1.data[0].x = self.__Espace_valeurs[liste_red.index(EV_proj.v_model)][0]
                fig1.data[0].y = self.__Espace_valeurs[liste_red.index(EV_proj.v_model)][1]
            with fig1_3D.batch_update():
                fig1_3D.data[0].x = self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    0
                ]
                fig1_3D.data[0].y = self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    1
                ]
                fig1_3D.data[0].z = self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    2
                ]

        valider_params_proj_EV.on_event("click", changement_params_EV)

        def reinit_param_EV(*b):
            # reset projection settings
            if liste_red.index(EV_proj.v_model) == 3:
                out_loading1.layout.visibility = "visible"
                dim_red = compute.DimensionalityReduction("PaCMAP", True)
                self.__Espace_valeurs[3] = dim_red.compute(X, 2)
                self.__Espace_valeurs_3D[3] = dim_red.compute(X, 3)
                out_loading1.layout.visibility = "hidden"

            with fig1.batch_update():
                fig1.data[0].x = self.__Espace_valeurs[liste_red.index(EV_proj.v_model)][0]
                fig1.data[0].y = self.__Espace_valeurs[liste_red.index(EV_proj.v_model)][1]
            with fig1_3D.batch_update():
                fig1_3D.data[0].x = self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    0
                ]
                fig1_3D.data[0].y = self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    1
                ]
                fig1_3D.data[0].z = self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    2
                ]

        reinit_params_proj_EV.on_event("click", reinit_param_EV)

        # here the sliders of the parameters for the EE!
        slider_param_PaCMAP_voisins_EE = gui_elements.SliderParam(v_model=10, min=5, max=30, step=1, label="Number of neighbors")
        slider_param_PaCMAP_mn_ratio_EE = gui_elements.SliderParam(v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio :")
        slider_param_PaCMAP_fp_ratio_EE = gui_elements.SliderParam(v_model=2, min=0.1, max=5, step=0.1, label="FP ratio :")

        def fonction_update_sliderEE(widget, event, data):
            if widget.label == "Number of neighbors :":
                slider_param_PaCMAP_voisins_EE.children[1].children = [str(data)]
            elif widget.label == "MN ratio :":
                slider_param_PaCMAP_mn_ratio_EE.children[1].children = [str(data)]
            elif widget.label == "FP ratio :":
                slider_param_PaCMAP_fp_ratio_EE.children[1].children = [str(data)]

        slider_param_PaCMAP_voisins_EE.children[0].on_event(
            "input", fonction_update_sliderEE
        )
        slider_param_PaCMAP_mn_ratio_EE.children[0].on_event(
            "input", fonction_update_sliderEE
        )
        slider_param_PaCMAP_fp_ratio_EE.children[0].on_event(
            "input", fonction_update_sliderEE
        )

        tous_sliders_EE = widgets.VBox(
            [
                slider_param_PaCMAP_voisins_EE,
                slider_param_PaCMAP_mn_ratio_EE,
                slider_param_PaCMAP_fp_ratio_EE,
            ],
            layout=Layout(
                width="100%",
            ),
        )

        valider_params_proj_EE = v.Btn(
            children=[
                v.Icon(left=True, children=["mdi-check"]),
                "Valider",
            ]
        )

        reinit_params_proj_EE = v.Btn(
            class_="ml-4",
            children=[
                v.Icon(left=True, children=["mdi-skip-backward"]),
                "Réinitialiser",
            ],
        )

        deux_boutons_params_EE = widgets.HBox(
            [valider_params_proj_EE, reinit_params_proj_EE]
        )
        params_proj_EE = widgets.VBox(
            [tous_sliders_EE, deux_boutons_params_EE],
            layout=Layout(width="100%", display="flex", align_items="center"),
        )

        def changement_params_EE(*b):
            liste_red = ["PCA", "t-SNE", "UMAP", "PaCMAP"]
            n_neighbors = slider_param_PaCMAP_voisins_EE.children[0].v_model
            MN_ratio = slider_param_PaCMAP_mn_ratio_EE.children[0].v_model
            FP_ratio = slider_param_PaCMAP_fp_ratio_EE.children[0].v_model
            if liste_red.index(EV_proj.v_model) == 3:
                out_loading2.layout.visibility = "visible"
                dim_red = compute.DimensionalityReduction("PaCMAP", False, n_neighbors, MN_ratio, FP_ratio)
                self.__Espace_explications[3] = dim_red.compute(self.__explanatory_values, 2)
                self.__Espace_explications_3D[3] = dim_red.compute(self.__explanatory_values, 3)
                out_loading2.layout.visibility = "hidden"
            with fig2.batch_update():
                fig2.data[0].x = self.__Espace_explications[liste_red.index(EE_proj.v_model)][
                    0
                ]
                fig2.data[0].y = self.__Espace_explications[liste_red.index(EE_proj.v_model)][
                    1
                ]
            with fig2_3D.batch_update():
                fig2_3D.data[0].x = self.__Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][0]
                fig2_3D.data[0].y = self.__Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][1]
                fig2_3D.data[0].z = self.__Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][2]

        valider_params_proj_EE.on_event("click", changement_params_EE)

        def reinit_param_EE(*b):
            if EE_proj.v_model == "PaCMAP":
                out_loading2.layout.visibility = "visible"
                dim_red = compute.DimensionalityReduction("PaCMAP", True)
                self.__Espace_explications[3] = dim_red.compute(self.__explanatory_values, 2)
                self.__Espace_explications_3D[3] = dim_red.compute(self.__explanatory_values, 3)
                out_loading2.layout.visibility = "hidden"
            with fig2.batch_update():
                fig2.data[0].x = self.__Espace_explications[liste_red.index(EE_proj.v_model)][
                    0
                ]
                fig2.data[0].y = self.__Espace_explications[liste_red.index(EE_proj.v_model)][
                    1
                ]
            with fig2_3D.batch_update():
                fig2_3D.data[0].x = self.__Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][0]
                fig2_3D.data[0].y = self.__Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][1]
                fig2_3D.data[0].z = self.__Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][2]

        reinit_params_proj_EE.on_event("click", reinit_param_EE)

        # allows you to choose the color of the points
        # y, y hat, residuals, current selection, regions, unselected points, automatic clustering
        couleur_radio = gui_elements.color_choice()

        def fonction_changement_couleur(*args, opacity: bool = True):
            # allows you to change the color of the points when you click on the buttons
            couleur = None
            scale = True
            a_modifier = True
            if couleur_radio.v_model == "y":
                couleur = y
            elif couleur_radio.v_model == "y^":
                couleur = self.atk.dataset.y_pred
            elif couleur_radio.v_model == "Selec actuelle":
                scale = False
                couleur = ["grey"] * len(self.atk.dataset.X)
                for i in range(len(self.selection)):
                    couleur[self.selection[i]] = "blue"
            elif couleur_radio.v_model == "Résidus":
                couleur = y - self.atk.dataset.y_pred
                couleur = [abs(i) for i in couleur]
            elif couleur_radio.v_model == "Régions":
                scale = False
                couleur = [0] * len(self.atk.dataset.X)
                for i in range(len(self.atk.dataset.X)):
                    for j in range(len(self.__list_of_regions)):
                        if i in self.__list_of_regions[j]:
                            couleur[i] = j + 1
            elif couleur_radio.v_model == "Non selec":
                scale = False
                couleur = ["red"] * len(self.atk.dataset.X)
                if len(self.__list_of_regions) > 0:
                    for i in range(len(self.atk.dataset.X)):
                        for j in range(len(self.__list_of_regions)):
                            if i in self.__list_of_regions[j]:
                                couleur[i] = "grey"
            elif couleur_radio.v_model == "Clustering auto":
                couleur = self.__Y_auto
                a_modifier = False
                scale = False
            with fig1.batch_update():
                fig1.data[0].marker.color = couleur
                if opacity:
                    fig1.data[0].marker.opacity = 1
            with fig2.batch_update():
                fig2.data[0].marker.color = couleur
                if opacity:
                    fig2.data[0].marker.opacity = 1
            with fig1_3D.batch_update():
                fig1_3D.data[0].marker.color = couleur
            with fig2_3D.batch_update():
                fig2_3D.data[0].marker.color = couleur
            if scale:
                fig1.update_traces(marker=dict(showscale=True))
                fig1_3D.update_traces(marker=dict(showscale=True))
                fig1.data[0].marker.colorscale = "Viridis"
                fig1_3D.data[0].marker.colorscale = "Viridis"
                fig2.data[0].marker.colorscale = "Viridis"
                fig2_3D.data[0].marker.colorscale = "Viridis"
            else:
                fig1.update_traces(marker=dict(showscale=False))
                fig1_3D.update_traces(marker=dict(showscale=False))
                if a_modifier:
                    fig1.data[0].marker.colorscale = "Plasma"
                    fig1_3D.data[0].marker.colorscale = "Plasma"
                    fig2.data[0].marker.colorscale = "Plasma"
                    fig2_3D.data[0].marker.colorscale = "Plasma"
                else:
                    fig1.data[0].marker.colorscale = "Viridis"
                    fig1_3D.data[0].marker.colorscale = "Viridis"
                    fig2.data[0].marker.colorscale = "Viridis"
                    fig2_3D.data[0].marker.colorscale = "Viridis"

        couleur_radio.on_event("change", fonction_changement_couleur)

        # EVX and EVY are the coordinates of the points in the value space
        EVX = np.array(self.__Espace_valeurs[choix_init_proj][0])
        EVY = np.array(self.__Espace_valeurs[choix_init_proj][1])

        # marker 1 is the marker of figure 1
        marker1 = dict(
            color=y,
            colorscale="Viridis",
            colorbar=dict(
                title="y",
                thickness=20,
            ),
        )

        # marker 2 is the marker of figure 2 (without colorbar therefore)
        marker2 = dict(color=y, colorscale="Viridis")

        barre_menu, fig_size, bouton_save = gui_elements.create_menu_bar()

        if self.atk.saves == None:
            self.atk.saves = []

        len_init_regions = len(self.atk.saves)

        # for the part on backups
        def init_save(new: bool = False):
            texte_regions = "There is no backup"
            for i in range(len(self.atk.saves)):
                if len(self.atk.saves[i]["liste"]) != len(atk.dataset.X_all):
                    raise Exception("Your save is not the right size !")
            if len(self.atk.saves) > 0:
                texte_regions = str(len(self.atk.saves)) + " save(s) found"
            self.__table_save = []
            for i in range(len(self.atk.saves)):
                sous_mod_bool = "No"
                new_or_not = "Imported"
                if i > len_init_regions:
                    new_or_not = "Created"
                if (
                    len(self.atk.saves[i]["self.sub_models"])
                    == max(self.atk.saves[i]["liste"]) + 1
                ):
                    sous_mod_bool = "Yes"
                self.__table_save.append(
                    [
                        i + 1,
                        self.atk.saves[i]["nom"],
                        new_or_not,
                        max(self.atk.saves[i]["liste"]) + 1,
                        sous_mod_bool,
                    ]
                )
            self.__table_save = pd.DataFrame(
                self.__table_save,
                columns=[
                    "Save #",
                    "Name",
                    "Origin",
                    "Number of regions",
                    "Sub-models ?",
                ],
            )

            colonnes = [
                {"text": c, "sortable": True, "value": c} for c in self.__table_save.columns
            ]

            self.__table_save = v.DataTable(
                v_model=[],
                show_select=True,
                single_select=True,
                headers=colonnes,
                items=self.__table_save.to_dict("records"),
                item_value="Save #",
                item_key="Save #",
            )
            return [self.__table_save, texte_regions]

        # the table that contains the backups
        self.__table_save = init_save()[0]

        dialogue_save, carte_save, delete_save, nom_sauvegarde, visu_save, new_save = gui_elements.dialog_save(bouton_save, init_save()[1], self.__table_save, self.atk.saves)

        # save a backup
        def delete_save_fonction(*args):
            if self.__table_save.v_model == []:
                return
            self.__table_save = carte_save.children[1]
            indice = self.__table_save.v_model[0]["Save #"] - 1
            self.atk.saves.pop(indice)
            self.__table_save, texte = init_save(True)
            carte_save.children = [texte, self.__table_save] + carte_save.children[
                2:
            ]

        delete_save.on_event("click", delete_save_fonction)

        # to view a backup
        def fonction_visu_save(*args):
            self.__table_save = carte_save.children[1]
            if len(self.__table_save.v_model) == 0:
                return
            indice = self.__table_save.v_model[0]["Save #"] - 1
            n = []
            for i in range(int(max(self.atk.saves[indice]["liste"])) + 1):
                temp = []
                for j in range(len(self.atk.saves[indice]["liste"])):
                    if self.atk.saves[indice]["liste"][j] == i:
                        temp.append(j)
                if len(temp) > 0:
                    n.append(temp)
            self.__list_of_regions = n
            couleur = deepcopy(self.atk.saves[indice]["liste"])
            self.__color_regions = deepcopy(couleur)
            with fig1.batch_update():
                fig1.data[0].marker.color = couleur
                fig1.data[0].marker.opacity = 1
            with fig2.batch_update():
                fig2.data[0].marker.color = couleur
                fig2.data[0].marker.opacity = 1
            with fig1_3D.batch_update():
                fig1_3D.data[0].marker.color = couleur
            with fig2_3D.batch_update():
                fig2_3D.data[0].marker.color = couleur
            couleur_radio.v_model = "Régions"
            fig1.update_traces(marker=dict(showscale=False))
            fig2.update_traces(marker=dict(showscale=False))
            if len(self.atk.saves[indice]["self.sub_models"]) != len(self.__list_of_regions):
                self.__list_of_sub_models = [[None, None, None]] * len(self.__list_of_regions)
            else:
                self.__list_of_sub_models = []
                for i in range(len(self.__list_of_regions)):
                    nom = self.atk.saves[indice]["self.sub_models"][i].__class__.__name__
                    indices_respectent = self.__list_of_regions[i]
                    score_init = compute.fonction_score(
                        y.iloc[indices_respectent], self.atk.dataset.y_pred[indices_respectent]
                    )
                    self.atk.saves[indice]["self.sub_models"][i].fit(
                        X.iloc[indices_respectent], y.iloc[indices_respectent]
                    )
                    score_reg = compute.fonction_score(
                        y.iloc[indices_respectent],
                        self.atk.saves[indice]["self.sub_models"][i].predict(
                            X.iloc[indices_respectent]
                        ),
                    )
                    if score_init == 0:
                        l_compar = "inf"
                    else:
                        l_compar = round(100 * (score_init - score_reg) / score_init, 1)
                    score = [score_reg, score_init, l_compar]
                    self.__list_of_sub_models.append([nom, score, -1])
            fonction_validation_une_tuile()

        visu_save.on_event("click", fonction_visu_save)

        # create a new savegame with the current regions
        def fonction_new_save(*args):
            if len(nom_sauvegarde.v_model) == 0 or len(nom_sauvegarde.v_model) > 25:
                return
            l_m = []
            if len(self.__list_of_sub_models) == 0:
                return
            for i in range(len(self.__list_of_sub_models)):
                if self.__list_of_sub_models[i][-1] == None:
                    l_m.append(None)
                else:
                    l_m.append(self.sub_models[self.__list_of_sub_models[i][-1]])
            save = create_save(self.__color_regions, nom_sauvegarde.v_model, l_m)
            self.atk.saves.append(save)
            self.__table_save, texte = init_save(True)
            carte_save.children = [texte, self.__table_save] + carte_save.children[
                2:
            ]

        new_save.on_event("click", fonction_new_save)

        # value space graph
        fig1 = go.FigureWidget(
            data=go.Scatter(x=EVX, y=EVY, mode="markers", marker=marker1)
        )

        # to remove the plotly logo
        fig1._config = fig1._config | {"displaylogo": False}

        # border size
        M = 40

        fig1.update_layout(margin=dict(l=M, r=M, t=0, b=M), width=int(fig_size.v_model))
        fig1.update_layout(dragmode="lasso")

        # coordonnées des points de l'espace des explications
        EEX = np.array(self.__Espace_explications[choix_init_proj][0])
        EEY = np.array(self.__Espace_explications[choix_init_proj][1])

        # grapbique de l'espace des explications
        fig2 = go.FigureWidget(
            data=go.Scatter(x=EEX, y=EEY, mode="markers", marker=marker2)
        )

        fig2.update_layout(margin=dict(l=M, r=M, t=0, b=M), width=int(fig_size.v_model))
        fig2.update_layout(dragmode="lasso")

        fig2._config = fig2._config | {"displaylogo": False}

        # two checkboxes to choose the projection dimension of figures 1 and 2
        dimension_projection = v.Switch(
            class_="ml-3 mr-2",
            v_model=False,
            label="",
        )

        dimension_projection_text = v.Row(
            class_="ma-3",
            children=[
                v.Icon(children=["mdi-numeric-2-box"]),
                v.Icon(children=["mdi-alpha-d-box"]),
                dimension_projection,
                v.Icon(children=["mdi-numeric-3-box"]),
                v.Icon(children=["mdi-alpha-d-box"]),
            ],
        )

        dimension_projection_text = add_tooltip(
            dimension_projection_text, "Dimension of the projection"
        )

        def fonction_dimension_projection(*args):
            if dimension_projection.v_model:
                fig_2D_ou_3D.children = [fig1_3D_et_texte, fig2_3D_et_texte]
            else:
                fig_2D_ou_3D.children = [fig1_et_texte, fig2_et_texte]

        dimension_projection.on_event("change", fonction_dimension_projection)

        # coordinates of points in the 3D value space
        EVX_3D = np.array(self.__Espace_valeurs_3D[choix_init_proj][0])
        EVY_3D = np.array(self.__Espace_valeurs_3D[choix_init_proj][1])
        EVZ_3D = np.array(self.__Espace_valeurs_3D[choix_init_proj][2])

        # coordinates of the points of the space of the explanations in 3D
        EEX_3D = np.array(self.__Espace_explications_3D[choix_init_proj][0])
        EEY_3D = np.array(self.__Espace_explications_3D[choix_init_proj][1])
        EEZ_3D = np.array(self.__Espace_explications_3D[choix_init_proj][2])

        # marker 3D is the marker of figure 1 in 3D
        marker_3D = dict(
            color=y,
            colorscale="Viridis",
            colorbar=dict(
                thickness=20,
            ),
            size=3,
        )

        # marker 3D_2 is the marker of figure 2 in 3D (without the colorbar therefore!)
        marker_3D_2 = dict(color=y, colorscale="Viridis", size=3)

        fig1_3D = go.FigureWidget(
            data=go.Scatter3d(
                x=EVX_3D, y=EVY_3D, z=EVZ_3D, mode="markers", marker=marker_3D
            )
        )
        fig1_3D.update_layout(
            margin=dict(l=M, r=M, t=0, b=M),
            width=int(fig_size.v_model),
            scene=dict(aspectmode="cube"),
            template="none",
        )

        fig1_3D._config = fig1_3D._config | {"displaylogo": False}

        fig2_3D = go.FigureWidget(
            data=go.Scatter3d(
                x=EEX_3D, y=EEY_3D, z=EEZ_3D, mode="markers", marker=marker_3D_2
            )
        )
        fig2_3D.update_layout(
            margin=dict(l=M, r=M, t=0, b=M),
            width=int(fig_size.v_model),
            scene=dict(aspectmode="cube"),
            template="none",
        )

        fig2_3D._config = fig2_3D._config | {"displaylogo": False}

        # text that indicate spaces for better understanding
        texteEV = widgets.HTML("<h3>Values Space<h3>")
        texteEE = widgets.HTML("<h3>Explanatory Space<h3>")

        # we display the figures and the text above!
        fig1_et_texte = gui_elements.figure_and_text(fig1, texteEV)
        fig2_et_texte = gui_elements.figure_and_text(fig2, texteEE)
        fig1_3D_et_texte = gui_elements.figure_and_text(fig1_3D, texteEV)
        fig2_3D_et_texte = gui_elements.figure_and_text(fig2_3D, texteEE)

        # HBox which allows you to choose between 2D and 3D figures by changing its children parameter!
        fig_2D_ou_3D = widgets.HBox([fig1_et_texte, fig2_et_texte])

        # allows to update graphs 1 & 2 according to the chosen projection
        def update_scatter(*args):
            val_act_EV = deepcopy(liste_red.index(EV_proj.v_model))
            val_act_EE = deepcopy(liste_red.index(EE_proj.v_model))
            param_EV.v_slots[0]["children"].disabled = True
            param_EE.v_slots[0]["children"].disabled = True
            if str(self.__Espace_valeurs[liste_red.index(EV_proj.v_model)]) == "None":
                out_loading1.layout.visibility = "visible"
                dim_red = compute.DimensionalityReduction(EV_proj.v_model, True)
                self.__Espace_valeurs[liste_red.index(EV_proj.v_model)] = dim_red.compute(X, 2)
                self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)] = dim_red.compute(X, 3)
                out_loading1.layout.visibility = "hidden"
            if str(self.__Espace_explications[liste_red.index(EE_proj.v_model)]) == "None":
                out_loading2.layout.visibility = "visible"
                dim_red = compute.DimensionalityReduction(EE_proj.v_model, True)
                self.__Espace_explications[liste_red.index(EE_proj.v_model)] = dim_red.compute(X, 2)
                self.__Espace_explications_3D[liste_red.index(EE_proj.v_model)] = dim_red.compute(X, 3)
                out_loading2.layout.visibility = "hidden"
            if liste_red.index(EE_proj.v_model) == 3:
                param_EE.v_slots[0]["children"].disabled = False
            if liste_red.index(EV_proj.v_model) == 3:
                param_EV.v_slots[0]["children"].disabled = False
            with fig1.batch_update():
                fig1.data[0].x = np.array(
                    self.__Espace_valeurs[liste_red.index(EV_proj.v_model)][0]
                )
                fig1.data[0].y = np.array(
                    self.__Espace_valeurs[liste_red.index(EV_proj.v_model)][1]
                )
            with fig2.batch_update():
                fig2.data[0].x = np.array(
                    self.__Espace_explications[liste_red.index(EE_proj.v_model)][0]
                )
                fig2.data[0].y = np.array(
                    self.__Espace_explications[liste_red.index(EE_proj.v_model)][1]
                )
            with fig1_3D.batch_update():
                fig1_3D.data[0].x = np.array(
                    self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][0]
                )
                fig1_3D.data[0].y = np.array(
                    self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][1]
                )
                fig1_3D.data[0].z = np.array(
                    self.__Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][2]
                )
            with fig2_3D.batch_update():
                fig2_3D.data[0].x = np.array(
                    self.__Espace_explications_3D[liste_red.index(EE_proj.v_model)][0]
                )
                fig2_3D.data[0].y = np.array(
                    self.__Espace_explications_3D[liste_red.index(EE_proj.v_model)][1]
                )
                fig2_3D.data[0].z = np.array(
                    self.__Espace_explications_3D[liste_red.index(EE_proj.v_model)][2]
                )

            EV_proj.v_model = liste_red[val_act_EV]
            EE_proj.v_model = liste_red[val_act_EE]

        # we observe the changes in the values ​​of the dropdowns to change the method of reduction
        EV_proj.on_event("change", update_scatter)
        EE_proj.on_event("change", update_scatter)

        self.__color_regions = [0] * len(self.atk.dataset.X)

        # definition of the table that will show the different results of the regions, with a stat of info about them
        table_regions = widgets.Output()

        # definition of the text that will give information on the selection
        texte_base = "About the current selection : \n"
        texte_base_debut = (
            "About the current selection : \n0 pont selected (0% of the overall data)"
        )
        # text that will take the value of text_base + information on the selection
        texte_selec = widgets.Textarea(
            value=texte_base_debut,
            placeholder="Infos",
            description="",
            disabled=True,
            layout=Layout(width="100%"),
        )

        card_selec = v.Card(
            class_="ma-2",
            elevation=0,
            children=[
                v.Layout(
                    children=[
                        v.Icon(children=["mdi-lasso"]),
                        v.Html(
                            class_="mt-2 ml-4",
                            tag="h4",
                            children=[
                                "0 point selected : use the lasso tool on the figures above or use the auto-selection tool below"
                            ],
                        ),
                    ]
                ),
            ],
        )

        button_valider_skope = v.Btn(
            class_="ma-1",
            children=[
                v.Icon(class_="mr-2", children=["mdi-auto-fix"]),
                "Skope-Rules",
            ],
        )

        # button that allows you to return to the initial rules in part 2. Skope-rules

        boutton_reinit_skope = v.Btn(
            class_="ma-1",
            children=[
                v.Icon(class_="mr-2", children=["mdi-skip-backward"]),
                "Come back to the initial rules",
            ],
        )

        # text that will contain the skope info on the EV
        une_carte_EV = v.Card(
            class_="mx-4 mt-0",
            elevation=0,
            children=[
                v.CardText(
                    children=[
                        v.Row(
                            class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                            children=[
                                "Waiting for the skope-rules to be applied...",
                            ],
                        )
                    ]
                )
            ],
        )

        texte_skopeEV = v.Card(
            style_="width: 50%;",
            class_="ma-3",
            children=[
                v.Row(
                    class_="ml-4",
                    children=[
                        v.Icon(children=["mdi-target"]),
                        v.CardTitle(children=["Rules applied to the Values Space"]),
                        v.Spacer(),
                        v.Html(
                            class_="mr-5 mt-5 font-italic",
                            tag="p",
                            children=["precision = /"],
                        ),
                    ],
                ),
                une_carte_EV,
            ],
        )

        # text that will contain the skope info on the EE

        une_carte_EE = v.Card(
            class_="mx-4 mt-0",
            elevation=0,
            # style_="width: 100%;",
            children=[
                v.CardText(
                    children=[
                        v.Row(
                            class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                            children=[
                                "Waiting for the skope-rules to be applied...",
                            ],
                        )
                    ]
                ),
            ],
        )

        texte_skopeEE = v.Card(
            style_="width: 50%;",
            class_="ma-3",
            children=[
                v.Row(
                    class_="ml-4",
                    children=[
                        v.Icon(children=["mdi-target"]),
                        v.CardTitle(children=["Rules applied on the Explanatory Space"]),
                        v.Spacer(),
                        v.Html(
                            class_="mr-5 mt-5 font-italic",
                            tag="p",
                            children=["precision = /"],
                        ),
                    ],
                ),
                une_carte_EE,
            ],
        )

        # text that will contain the skope info on the EV and EE
        texte_skope = v.Layout(
            class_="d-flex flex-row", children=[texte_skopeEV, texte_skopeEE]
        )

        # texts that will contain the information on the self.sub_models
        liste_mods = []
        for i in range(len(self.sub_models)):
            nom_mdi = "mdi-numeric-" + str(i + 1) + "-box"
            mod = v.SlideItem(
                # style_="width: 30%",
                children=[
                    v.Card(
                        class_="grow ma-2",
                        children=[
                            v.Row(
                                class_="ml-5 mr-4",
                                children=[
                                    v.Icon(children=[nom_mdi]),
                                    v.CardTitle(
                                        children=[self.sub_models[i].__class__.__name__]
                                    ),
                                ],
                            ),
                            v.CardText(
                                class_="mt-0 pt-0",
                                children=["Model's score"],
                            ),
                        ],
                    )
                ],
            )
            liste_mods.append(mod)

        mods = v.SlideGroup(
            v_model=None,
            class_="ma-3 pa-3",
            elevation=4,
            center_active=True,
            show_arrows=True,
            children=liste_mods,
        )

        def changement(widget, event, data, args: bool = True):
            if args == True:
                for i in range(len(mods.children)):
                    mods.children[i].children[0].color = "white"
                widget.color = "blue lighten-4"
            for i in range(len(mods.children)):
                if mods.children[i].children[0].color == "blue lighten-4":
                    self.__model_choice = i

        for i in range(len(mods.children)):
            mods.children[i].children[0].on_event("click", changement)

        # selection validation button to create a region
        valider_une_region = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-3", children=["mdi-check"]),
                "Validate the selection",
            ],
        )
        # button to delete all regions
        supprimer_toutes_les_tuiles = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-3", children=["mdi-trash-can-outline"]),
                "Delete the selected regions",
            ],
        )

        # we wrap the sets of buttons in Boxes to put them online
        en_deux_b = v.Layout(
            class_="ma-3 d-flex flex-row",
            children=[valider_une_region, v.Spacer(), supprimer_toutes_les_tuiles],
        )
        selection = widgets.VBox([en_deux_b])

        # we define this to be able to initialize the histograms
        x = np.linspace(0, 20, 20)

        # we define the sliders used to modify the histogram resulting from the skope
        slider_skope1 = v.RangeSlider(
            class_="ma-3",
            v_model=[-1, 1],
            min=-10e10,
            max=10e10,
            step=1,
        )

        bout_temps_reel_graph1 = v.Checkbox(
            v_model=False, label="Real-time updates on the figures", class_="ma-3"
        )

        slider_text_comb1 = v.Layout(
            children=[
                v.TextField(
                    style_="max-width:100px",
                    v_model=slider_skope1.v_model[0] / 100,
                    hide_details=True,
                    type="number",
                    density="compact",
                ),
                slider_skope1,
                v.TextField(
                    style_="max-width:100px",
                    v_model=slider_skope1.v_model[1] / 100,
                    hide_details=True,
                    type="number",
                    density="compact",
                ),
            ],
        )

        def update_validate1(*args):
            if bout_temps_reel_graph1.v_model:
                valider_change_1.disabled = True
            else:
                valider_change_1.disabled = False

        bout_temps_reel_graph1.on_event("change", update_validate1)

        slider_skope2 = v.RangeSlider(
            class_="ma-3",
            v_model=[-1, 1],
            min=-10e10,
            max=10e10,
            step=1,
        )

        bout_temps_reel_graph2 = v.Checkbox(
            v_model=False, label="Real-time updates on the figures", class_="ma-3"
        )

        slider_text_comb2 = v.Layout(
            children=[
                v.TextField(
                    style_="max-width:100px",
                    v_model=slider_skope2.v_model[0],
                    hide_details=True,
                    type="number",
                    density="compact",
                ),
                slider_skope2,
                v.TextField(
                    style_="max-width:100px",
                    v_model=slider_skope2.v_model[1],
                    hide_details=True,
                    type="number",
                    density="compact",
                ),
            ],
        )

        def update_validate2(*args):
            if bout_temps_reel_graph2.value:
                valider_change_2.disabled = True
            else:
                valider_change_2.disabled = False

        bout_temps_reel_graph2.observe(update_validate2)

        slider_skope3 = v.RangeSlider(
            class_="ma-3",
            v_model=[-1, 1],
            min=-10e10,
            max=10e10,
            step=1,
        )

        bout_temps_reel_graph3 = v.Checkbox(
            v_model=False, label="Real-time updates on the figures", class_="ma-3"
        )

        slider_text_comb3 = v.Layout(
            children=[
                v.TextField(
                    style_="max-width:60px",
                    v_model=slider_skope3.v_model[0],
                    hide_details=True,
                    type="number",
                    density="compact",
                ),
                slider_skope3,
                v.TextField(
                    style_="max-width:60px",
                    v_model=slider_skope3.v_model[1],
                    hide_details=True,
                    type="number",
                    density="compact",
                ),
            ],
        )

        def update_validate3(*args):
            if bout_temps_reel_graph3.v_model:
                valider_change_3.disabled = True
            else:
                valider_change_3.disabled = False

        bout_temps_reel_graph3.on_event("change", update_validate3)

        # valid buttons definition changes:

        valider_change_1 = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-2", children=["mdi-check"]),
                "Validate the changes",
            ],
        )
        valider_change_2 = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-2", children=["mdi-check"]),
                "Validate the changes",
            ],
        )
        valider_change_3 = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-2", children=["mdi-check"]),
                "Validate the changes",
            ],
        )

        # we wrap the validation button and the checkbox which allows you to view in real time
        deux_fin1 = widgets.HBox([valider_change_1, bout_temps_reel_graph1])
        deux_fin2 = widgets.HBox([valider_change_2, bout_temps_reel_graph2])
        deux_fin3 = widgets.HBox([valider_change_3, bout_temps_reel_graph3])

        # we define the number of bars of the histogram
        nombre_bins = 50
        # we define the histograms
        histogram1 = go.FigureWidget(
            data=[
                go.Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="grey")
            ]
        )
        histogram1.update_layout(
            barmode="overlay",
            bargap=0.1,
            width=0.9 * int(fig_size.v_model),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=150,
        )
        histogram1.add_trace(
            go.Histogram(
                x=x,
                bingroup=1,
                nbinsx=nombre_bins,
                marker_color="LightSkyBlue",
                opacity=0.6,
            )
        )
        histogram1.add_trace(
            go.Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="blue")
        )

        histogram2 = go.FigureWidget(
            data=[
                go.Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="grey")
            ]
        )
        histogram2.update_layout(
            barmode="overlay",
            bargap=0.1,
            width=0.9 * int(fig_size.v_model),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=150,
        )
        histogram2.add_trace(
            go.Histogram(
                x=x,
                bingroup=1,
                nbinsx=nombre_bins,
                marker_color="LightSkyBlue",
                opacity=0.6,
            )
        )
        histogram2.add_trace(
            go.Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="blue")
        )

        histogram3 = go.FigureWidget(
            data=[
                go.Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="grey")
            ]
        )
        histogram3.update_layout(
            barmode="overlay",
            bargap=0.1,
            width=0.9 * int(fig_size.v_model),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=150,
        )
        histogram3.add_trace(
            go.Histogram(
                x=x,
                bingroup=1,
                nbinsx=nombre_bins,
                marker_color="LightSkyBlue",
                opacity=0.6,
            )
        )
        histogram3.add_trace(
            go.Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="blue")
        )

        all_histograms = [histogram1, histogram2, histogram3]

        def fonction_beeswarm_shap(nom_colonne):
            # redefinition de la figure beeswarm de shap
            def positions_ordre_croissant(lst):
                positions = list(range(len(lst)))  # Create a list of initial positions
                positions.sort(key=lambda x: lst[x])
                l = []
                for i in range(len(positions)):
                    l.append(positions.index(i))  # Sort positions by list items
                return l

            nom_colonne_shap = nom_colonne + "_shap"
            y_histo_shap = [0] * len(self.__explanatory_values)
            nombre_div = 60
            garde_indice = []
            garde_valeur_y = []
            for i in range(nombre_div):
                garde_indice.append([])
                garde_valeur_y.append([])
            liste_scale = np.linspace(
                min(self.__explanatory_values[nom_colonne_shap]), max(self.__explanatory_values[nom_colonne_shap]), nombre_div + 1
            )
            for i in range(len(self.__explanatory_values)):
                for j in range(nombre_div):
                    if (
                        self.__explanatory_values[nom_colonne_shap][i] >= liste_scale[j]
                        and self.__explanatory_values[nom_colonne_shap][i] <= liste_scale[j + 1]
                    ):
                        garde_indice[j].append(i)
                        garde_valeur_y[j].append(y[i])
                        break
            for i in range(nombre_div):
                l = positions_ordre_croissant(garde_valeur_y[i])
                for j in range(len(garde_indice[i])):
                    ii = garde_indice[i][j]
                    if l[j] % 2 == 0:
                        y_histo_shap[ii] = l[j]
                    else:
                        y_histo_shap[ii] = -l[j]
            marker_shap = dict(
                size=4,
                opacity=0.6,
                color=self.atk.dataset.X[nom_colonne],
                colorscale="Bluered_r",
                colorbar=dict(thickness=20, title=nom_colonne),
            )
            return [y_histo_shap, marker_shap]

        # definitions of the different color choices for the swarm

        choix_couleur_essaim1 = v.Row(
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

        def changement_couleur_essaim_shap1(*args):
            if choix_couleur_essaim1.children[1].v_model == False:
                marker = fonction_beeswarm_shap(self.__all_rules[0][2])[1]
                essaim1.data[0].marker = marker
                essaim1.update_traces(marker=dict(showscale=True))
            else:
                modifier_tous_histograms(
                    slider_skope1.v_model[0] / 100, slider_skope1.v_model[1] / 100, 0
                )
                essaim1.update_traces(marker=dict(showscale=False))

        choix_couleur_essaim1.children[1].on_event(
            "change", changement_couleur_essaim_shap1
        )

        y_histo_shap = [0] * len(self.__explanatory_values)
        nom_col_shap = str(X.columns[0]) + "_shap"
        essaim1 = go.FigureWidget(
            data=[go.Scatter(x=self.__explanatory_values[nom_col_shap], y=y_histo_shap, mode="markers")]
        )
        essaim1.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            width=0.9 * int(fig_size.v_model),
        )
        essaim1.update_yaxes(visible=False, showticklabels=False)

        total_essaim_1 = widgets.VBox([choix_couleur_essaim1, essaim1])
        total_essaim_1.layout.margin = "0px 0px 0px 20px"

        choix_couleur_essaim2 = v.Row(
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

        def changement_couleur_essaim_shap2(*args):
            if choix_couleur_essaim2.children[1].v_model == False:
                marker = fonction_beeswarm_shap(self.__all_rules[1][2])[1]
                essaim2.data[0].marker = marker
                essaim2.update_traces(marker=dict(showscale=True))
            else:
                modifier_tous_histograms(
                    slider_skope2.v_model[0] / 100, slider_skope2.v_model[1] / 100, 1
                )
            essaim2.update_traces(marker=dict(showscale=False))

        choix_couleur_essaim2.children[1].on_event(
            "change", changement_couleur_essaim_shap2
        )

        essaim2 = go.FigureWidget(
            data=[go.Scatter(x=self.__explanatory_values[nom_col_shap], y=y_histo_shap, mode="markers")]
        )
        essaim2.update_layout(
            margin=dict(l=20, r=0, t=0, b=0),
            height=200,
            width=0.9 * int(fig_size.v_model),
        )
        essaim2.update_yaxes(visible=False, showticklabels=False)

        total_essaim_2 = widgets.VBox([choix_couleur_essaim2, essaim2])
        total_essaim_2.layout.margin = "0px 0px 0px 20px"

        essaim3 = go.FigureWidget(
            data=[go.Scatter(x=self.__explanatory_values[nom_col_shap], y=y_histo_shap, mode="markers")]
        )
        essaim3.update_layout(
            margin=dict(l=20, r=0, t=0, b=0),
            height=200,
            width=0.9 * int(fig_size.v_model),
        )
        essaim3.update_yaxes(visible=False, showticklabels=False)

        choix_couleur_essaim3 = v.Row(
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

        def changement_couleur_essaim_shap3(*args):
            if choix_couleur_essaim3.children[1].v_model == False:
                marker = fonction_beeswarm_shap(self.__all_rules[2][2])[1]
                essaim3.data[0].marker = marker
                essaim3.update_traces(marker=dict(showscale=True))
            else:
                modifier_tous_histograms(
                    slider_skope3.v_model[0] / 100, slider_skope3.v_model[1] / 100, 2
                )
                essaim3.update_traces(marker=dict(showscale=False))

        choix_couleur_essaim3.children[1].on_event(
            "change", changement_couleur_essaim_shap3
        )

        total_essaim_3 = widgets.VBox([choix_couleur_essaim3, essaim3])
        total_essaim_3.layout.margin = "0px 0px 0px 20px"

        all_beeswarms_total = [total_essaim_1, total_essaim_2, total_essaim_3]

        all_beeswarms = [essaim1, essaim2, essaim3]

        all_color_choosers_beeswarms = [
            choix_couleur_essaim1,
            choix_couleur_essaim2,
            choix_couleur_essaim3,
        ]
        # set of elements that contain histograms and sliders
        ens_slider_histo1 = widgets.VBox([slider_text_comb1, histogram1, deux_fin1])
        ens_slider_histo2 = widgets.VBox([slider_text_comb2, histogram2, deux_fin2])
        ens_slider_histo3 = widgets.VBox([slider_text_comb3, histogram3, deux_fin3])

        # definition of buttons to delete features (disabled for the first 3 for the moment)
        b_delete_skope1 = v.Btn(
            class_="ma-2 ml-4 pa-1",
            elevation="3",
            icon=True,
            children=[v.Icon(children=["mdi-delete"])],
            disabled=True,
        )
        b_delete_skope2 = v.Btn(
            class_="ma-2 ml-4 pa-1",
            elevation="3",
            icon=True,
            children=[v.Icon(children=["mdi-delete"])],
            disabled=True,
        )
        b_delete_skope3 = v.Btn(
            class_="ma-2 ml-4 pa-1",
            elevation="3",
            icon=True,
            children=[v.Icon(children=["mdi-delete"])],
            disabled=True,
        )

        dans_accordion1 = widgets.HBox(
            [ens_slider_histo1, total_essaim_1, b_delete_skope1],
            layout=Layout(align_items="center"),
        )
        dans_accordion2 = widgets.HBox(
            [ens_slider_histo2, total_essaim_2, b_delete_skope2],
            layout=Layout(align_items="center"),
        )
        dans_accordion3 = widgets.HBox(
            [ens_slider_histo3, total_essaim_3, b_delete_skope3],
            layout=Layout(align_items="center"),
        )

        elements_final_accordion = [dans_accordion1, dans_accordion2, dans_accordion3]

        # we define several accordions in this way to be able to open several at the same time
        dans_accordion1_n = v.ExpansionPanels(
            class_="ma-2 mb-1",
            children=[
                v.ExpansionPanel(
                    children=[
                        v.ExpansionPanelHeader(children=["X1"]),
                        v.ExpansionPanelContent(children=[dans_accordion1]),
                    ]
                )
            ],
        )

        dans_accordion2_n = v.ExpansionPanels(
            class_="ma-2 mt-0 mb-1",
            children=[
                v.ExpansionPanel(
                    children=[
                        v.ExpansionPanelHeader(children=["X2"]),
                        v.ExpansionPanelContent(children=[dans_accordion2]),
                    ]
                )
            ],
        )

        dans_accordion3_n = v.ExpansionPanels(
            class_="ma-2 mt-0",
            children=[
                v.ExpansionPanel(
                    children=[
                        v.ExpansionPanelHeader(children=["X3"]),
                        v.ExpansionPanelContent(children=[dans_accordion3]),
                    ]
                )
            ],
        )

        accordion_skope = widgets.VBox(
            children=[dans_accordion1_n, dans_accordion2_n, dans_accordion3_n],
            layout=Layout(width="100%", height="auto"),
        )

        # allows you to take the set of rules and modify the graph so that it responds to everything!
        def tout_modifier_graphique():
            nouvelle_tuile = self.atk.dataset.X[
                (self.atk.dataset.X[self.__all_rules[0][2]] >= self.__all_rules[0][0])
                & (self.atk.dataset.X[self.__all_rules[0][2]] <= self.__all_rules[0][4])
            ].index
            for i in range(1, len(self.__all_rules)):
                X_temp = self.atk.dataset.X[
                    (self.atk.dataset.X[self.__all_rules[i][2]] >= self.__all_rules[i][0])
                    & (self.atk.dataset.X[self.__all_rules[i][2]] <= self.__all_rules[i][4])
                ].index
                nouvelle_tuile = [g for g in nouvelle_tuile if g in X_temp]
            y_shape_skope = []
            y_color_skope = []
            y_opa_skope = []
            self.selection = nouvelle_tuile
            for i in range(len(self.atk.dataset.X)):
                if i in nouvelle_tuile:
                    y_shape_skope.append("circle")
                    y_color_skope.append("blue")
                    y_opa_skope.append(0.5)
                else:
                    y_shape_skope.append("cross")
                    y_color_skope.append("grey")
                    y_opa_skope.append(0.5)
            with fig1.batch_update():
                # fig1.data[0].marker.symbol = y_shape_skope
                fig1.data[0].marker.color = y_color_skope
                # fig1.data[0].marker.opacity = y_opa_skope
            with fig2.batch_update():
                # fig2.data[0].marker.symbol = y_shape_skope
                fig2.data[0].marker.color = y_color_skope
                # fig2.data[0].marker.opacity = y_opa_skope
            with fig1_3D.batch_update():
                # fig1_3D.data[0].marker.symbol = y_shape_skope
                fig1_3D.data[0].marker.color = y_color_skope
            with fig2_3D.batch_update():
                # fig2_3D.data[0].marker.symbol = y_shape_skope
                fig2_3D.data[0].marker.color = y_color_skope

        # allows to modify all the histograms according to the rules
        def modifier_tous_histograms(value_min, value_max, indice):
            new_list_tout = self.atk.dataset.X.index[
                self.atk.dataset.X[self.__all_rules[indice][2]].between(value_min, value_max)
            ].tolist()
            for i in range(len(self.__all_rules)):
                min = self.__all_rules[i][0]
                max = self.__all_rules[i][4]
                if i != indice:
                    new_list_temp = self.atk.dataset.X.index[
                        self.atk.dataset.X[self.__all_rules[i][2]].between(min, max)
                    ].tolist()
                    new_list_tout = [g for g in new_list_tout if g in new_list_temp]
            for i in range(len(self.__all_rules)):
                with all_histograms[i].batch_update():
                    all_histograms[i].data[2].x = self.atk.dataset.X[self.__all_rules[i][2]][new_list_tout]
                if all_color_choosers_beeswarms[i].children[1].v_model:
                    with all_beeswarms[i].batch_update():
                        y_color = [0] * len(self.__explanatory_values)
                        if i == indice:
                            indices = self.atk.dataset.X.index[
                                self.atk.dataset.X[self.__all_rules[i][2]].between(value_min, value_max)
                            ].tolist()
                        else:
                            indices = self.atk.dataset.X.index[
                                self.atk.dataset.X[self.__all_rules[i][2]].between(
                                    self.__all_rules[i][0], self.__all_rules[i][4]
                                )
                            ].tolist()
                        for j in range(len(self.__explanatory_values)):
                            if j in new_list_tout:
                                y_color[j] = "blue"
                            elif j in indices:
                                y_color[j] = "#85afcb"
                            else:
                                y_color[j] = "grey"
                        all_beeswarms[i].data[0].marker.color = y_color

        # when the value of a slider is modified, the histograms and graphs are modified
        def on_value_change_skope1(*b1):
            slider_text_comb1.children[0].v_model = slider_skope1.v_model[0] / 100
            slider_text_comb1.children[2].v_model = slider_skope1.v_model[1] / 100
            new_list = [
                g
                for g in list(self.atk.dataset.X[self.atk.dataset.X.columns[0]].values)
                if g >= slider_skope1.v_model[0] / 100
                and g <= slider_skope1.v_model[1] / 100
            ]
            with histogram1.batch_update():
                histogram1.data[1].x = new_list
            if self.__valider_bool:
                modifier_tous_histograms(
                    slider_skope1.v_model[0] / 100, slider_skope1.v_model[1] / 100, 0
                )
            if bout_temps_reel_graph1.v_model:
                self.__all_rules[0][0] = float(deepcopy(slider_skope1.v_model[0] / 100))
                self.__all_rules[0][4] = float(deepcopy(slider_skope1.v_model[1] / 100))
                une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
                tout_modifier_graphique()

        def on_value_change_skope2(*b):
            slider_text_comb2.children[0].v_model = slider_skope2.v_model[0] / 100
            slider_text_comb2.children[2].v_model = slider_skope2.v_model[1] / 100
            new_list = [
                g
                for g in list(self.atk.dataset.X[self.atk.dataset.X.columns[1]].values)
                if g >= slider_skope2.v_model[0] / 100
                and g <= slider_skope2.v_model[1] / 100
            ]
            with histogram2.batch_update():
                histogram2.data[1].x = new_list
            if self.__valider_bool:
                modifier_tous_histograms(
                    slider_skope2.v_model[0] / 100, slider_skope2.v_model[1] / 100, 1
                )
            if bout_temps_reel_graph2.v_model:
                self.__all_rules[1][0] = float(deepcopy(slider_skope2.v_model[0] / 100))
                self.__all_rules[1][4] = float(deepcopy(slider_skope2.v_model[1] / 100))
                une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
                tout_modifier_graphique()

        def on_value_change_skope3(*b):
            slider_text_comb3.children[0].v_model = slider_skope3.v_model[0] / 100
            slider_text_comb3.children[2].v_model = slider_skope3.v_model[1] / 100
            new_list = [
                g
                for g in list(self.atk.dataset.X[self.atk.dataset.X.columns[2]].values)
                if g >= slider_skope3.v_model[0] / 100
                and g <= slider_skope3.v_model[1] / 100
            ]
            with histogram3.batch_update():
                histogram3.data[1].x = new_list
            if self.__valider_bool:
                modifier_tous_histograms(
                    slider_skope3.v_model[0] / 100, slider_skope3.v_model[1] / 100, 2
                )
            if bout_temps_reel_graph3.v_model:
                self.__all_rules[2][0] = float(deepcopy(slider_skope3.v_model[0] / 100))
                self.__all_rules[2][4] = float(deepcopy(slider_skope3.v_model[1] / 100))
                une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
                tout_modifier_graphique()

        # allows you to display certain values ​​correctly (here su skope du shap)
        def transform_string(skope_result):
            chaine_carac = str(skope_result).split()
            regles = []
            regles.append(chaine_carac[0][2:] + " ")
            regles.append(chaine_carac[1] + " ")
            if chaine_carac[2][-1] == ",":
                regles.append(str(round(float(chaine_carac[2][:-2]), 3)))
            else:
                regles.append(str(round(float(chaine_carac[2]), 3)))
            ii = 3
            while ii < len(chaine_carac):
                if chaine_carac[ii] == "and":
                    regles.append("\n")
                    regles.append(chaine_carac[ii + 1] + " ")
                    regles.append(chaine_carac[ii + 2] + " ")
                    if chaine_carac[ii + 3][-1] == ",":
                        regles.append(str(round(float(chaine_carac[ii + 3][:-2]), 3)))
                    else:
                        regles.append(str(round(float(chaine_carac[ii + 3]), 3)))
                else:
                    break
                ii += 4
            precision = []
            precision.append(chaine_carac[ii][1:-1])
            precision.append(chaine_carac[ii + 1][:-1])
            precision.append(chaine_carac[ii + 2][:-2])
            return ["".join(regles), precision]

        def transform_string_shap(skope_result):
            chaine_carac = str(skope_result).split()
            regles = []
            if chaine_carac[1] == "<" or chaine_carac[1] == "<=":
                regles.append("min <= ")
                regles.append(chaine_carac[0][2:] + " ")
                regles.append(chaine_carac[1] + " ")
                if chaine_carac[2][-1] == ",":
                    regles.append(str(round(float(chaine_carac[2][:-2]), 3)))
                else:
                    regles.append(str(round(float(chaine_carac[2]), 3)))
            else:
                if chaine_carac[2][-1] == ",":
                    regles.append(str(round(float(chaine_carac[2][:-2]), 3)))
                else:
                    regles.append(str(round(float(chaine_carac[2]), 3)))
                regles.append(" <= ")
                regles.append(chaine_carac[0][2:] + " ")
                regles.append(chaine_carac[1] + " ")
                regles.append("max")

            ii = 3
            while ii < len(chaine_carac):
                if chaine_carac[ii] == "and":
                    regles.append(" ")
                    if chaine_carac[ii + 2] == "<" or chaine_carac[ii + 2] == "<=":
                        regles.append("min <= ")
                        regles.append(chaine_carac[ii + 1] + " ")
                        regles.append(chaine_carac[ii + 2] + " ")
                        if chaine_carac[ii + 3][-1] == ",":
                            regles.append(
                                str(round(float(chaine_carac[ii + 3][:-2]), 3))
                            )
                        else:
                            regles.append(str(round(float(chaine_carac[ii + 3]), 3)))
                    else:
                        if chaine_carac[ii + 3][-1] == ",":
                            regles.append(
                                str(round(float(chaine_carac[ii + 3][:-2]), 3))
                            )
                        else:
                            regles.append(str(round(float(chaine_carac[ii + 3]), 3)))
                        regles.append(" <= ")
                        regles.append(chaine_carac[ii + 1] + " ")
                        regles.append(chaine_carac[ii + 2] + " ")
                        regles.append("max")
                else:
                    break
                ii += 4
            precision = []
            precision.append(chaine_carac[ii][1:-1])
            precision.append(chaine_carac[ii + 1][:-1])
            precision.append(chaine_carac[ii + 2][:-2])
            return ["".join(regles), precision]

        # function to grab skope values ​​in float, used for modification sliders!
        def re_transform_string(chaine):
            chaine_carac = str(chaine).split()
            self.atk.dataset.X.columns = []
            valeurs = []
            symbole = []
            for i in range(len(chaine_carac)):
                if "<" in chaine_carac[i] or ">" in chaine_carac[i]:
                    self.atk.dataset.X.columns.append(chaine_carac[i - 1])
                    symbole.append(chaine_carac[i])
                    if chaine_carac[i + 1][-1] == ",":
                        valeurs.append(float(chaine_carac[i + 1][:-2]))
                    else:
                        valeurs.append(float(chaine_carac[i + 1]))
            return [self.atk.dataset.X.columns, symbole, valeurs]

        def generate_card(chaine):
            chaine_carac = str(chaine).split()
            taille = int(len(chaine_carac) / 5)
            l = []
            for i in range(taille):
                l.append(
                    v.CardText(
                        children=[
                            v.Row(
                                class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                                children=[
                                    chaine_carac[5 * i],
                                    v.Icon(children=["mdi-less-than-or-equal"]),
                                    chaine_carac[5 * i + 2],
                                    v.Icon(children=["mdi-less-than-or-equal"]),
                                    chaine_carac[5 * i + 4],
                                ],
                            )
                        ]
                    )
                )
                if i != taille - 1:
                    l.append(v.Divider())
            return l

        def liste_to_string_skope(liste):
            chaine = ""
            for regle in liste:
                for i in range(len(regle)):
                    if type(regle[i]) == float:
                        chaine += str(np.round(float(regle[i]), 2))
                    else:
                        chaine += str(regle[i])
                    chaine += " "
            return chaine

        # commit skope changes
        def fonction_change_valider_1(*change):
            a = deepcopy(float(slider_skope1.v_model[0] / 100))
            b = deepcopy(float(slider_skope1.v_model[1] / 100))
            self.__all_rules[0][0] = a
            self.__all_rules[0][4] = b
            une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
            tout_modifier_graphique()
            fonction_scores_models(None)

        def fonction_change_valider_2(*change):
            self.__all_rules[1][0] = float(slider_skope2.v_model[0] / 100)
            self.__all_rules[1][4] = float(slider_skope2.v_model[1] / 100)
            une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
            tout_modifier_graphique()
            fonction_scores_models(None)

        def fonction_change_valider_3(*change):
            self.__all_rules[2][0] = float(slider_skope3.v_model[0] / 100)
            self.__all_rules[2][4] = float(slider_skope3.v_model[1] / 100)
            une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
            tout_modifier_graphique()
            fonction_scores_models(None)

        valider_change_1.on_event("click", fonction_change_valider_1)
        valider_change_2.on_event("click", fonction_change_valider_2)
        valider_change_3.on_event("click", fonction_change_valider_3)

        def regles_to_indices():
            liste_bool = [True] * len(X)
            for i in range(len(X)):
                for j in range(len(self.__all_rules)):
                    colonne = list(X.columns).index(self.__all_rules[j][2])
                    if (
                        self.__all_rules[j][0] > self.atk.dataset.X.iloc[i, colonne]
                        or self.atk.dataset.X.iloc[i, colonne] > self.__all_rules[j][4]
                    ):
                        liste_bool[i] = False
            temp = [i for i in range(len(X)) if liste_bool[i]]
            return temp

        def fonction_scores_models(temp):
            if temp == None:
                temp = regles_to_indices()
            result_models = fonction_models(X.iloc[temp, :], y.iloc[temp])
            score_tot = []
            for i in range(len(self.sub_models)):
                score_tot.append(compute.fonction_score(y.iloc[temp], result_models[i][-2]))
            score_init = compute.fonction_score(y.iloc[temp], self.atk.dataset.y_pred[temp])
            if score_init == 0:
                l_compar = ["/"] * len(self.sub_models)
            else:
                l_compar = [
                    round(100 * (score_init - score_tot[i]) / score_init, 1)
                    for i in range(len(self.sub_models))
                ]

            self.__score_models = []
            for i in range(len(self.sub_models)):
                self.__score_models.append(
                    [
                        score_tot[i],
                        score_init,
                        l_compar[i],
                    ]
                )

            def str_md(i):
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
                mods.children[i].children[0].children[1].children = str_md(i)

        X_train = self.atk.dataset.X.copy()

        # when you click on the skope-rules button
        def fonction_validation_skope(*sender):
            y_train = self.__y_train
            loading_models.class_ = "d-flex"
            self.__valider_bool = True
            if y_train == None:
                texte_skopeEV.children[1].children = [
                    widgets.HTML("Please select points")
                ]
                texte_skopeEE.children[1].children = [
                    widgets.HTML("Please select points")
                ]
            elif 0 not in y_train or 1 not in y_train:
                texte_skopeEV.children[1].children = [
                    widgets.HTML("You can't choose everything/nothing !")
                ]
                texte_skopeEE.children[1].children = [
                    widgets.HTML("You can't choose everything/nothing !")
                ]
            else:
                # skope calculation for X
                skope_rules_clf = SkopeRules(
                    feature_names=X_train.columns,
                    random_state=42,
                    n_estimators=5,
                    recall_min=0.2,
                    precision_min=0.2,
                    max_depth_duplication=0,
                    max_samples=1.0,
                    max_depth=3,
                )
                skope_rules_clf.fit(X_train, y_train)

                # skope calculation for SHAP
                skope_rules_clf_shap = SkopeRules(
                    feature_names=self.__SHAP_train.columns,
                    random_state=42,
                    n_estimators=5,
                    recall_min=0.2,
                    precision_min=0.2,
                    max_depth_duplication=0,
                    max_samples=1.0,
                    max_depth=3,
                )
                skope_rules_clf_shap.fit(self.__SHAP_train, y_train)
                # if no rule for one of the two, nothing is displayed
                if (
                    len(skope_rules_clf.rules_) == 0
                    or len(skope_rules_clf_shap.rules_) == 0
                ):
                    texte_skopeEV.children[1].children = [
                        widgets.HTML("No rule found")
                    ]
                    texte_skopeEE.children[1].children = [
                        widgets.HTML("No rule found")
                    ]
                    indices_respectent_skope = [0]
                # otherwise we display
                else:
                    chaine_carac = transform_string(skope_rules_clf.rules_[0])
                    texte_skopeEV.children[0].children[3].children = [
                        "p = "
                        + str(np.round(float(chaine_carac[1][0]) * 100, 5))
                        + "%"
                        + " r = "
                        + str(np.round(float(chaine_carac[1][1]) * 100, 5))
                        + "%"
                        + " ext. of the tree = "
                        + chaine_carac[1][2]
                    ]

                    # there we find the values ​​of the skope to use them for the sliders
                    self.atk.dataset.X.columns, symbole, valeurs = re_transform_string(
                        chaine_carac[0]
                    )
                    self.__other_columns = [
                        g for g in self.atk.dataset.X.columns if g not in self.atk.dataset.X.columns
                    ]
                    widget_list_add_skope.items = self.__other_columns
                    widget_list_add_skope.v_model = self.__other_columns[0]
                    liste_val_histo = [0] * len(self.atk.dataset.X.columns)
                    liste_index = [0] * len(self.atk.dataset.X.columns)
                    le_top = []
                    le_min = []
                    self.__all_rules = []

                    def f_rond(a):
                        return np.round(a, 2)

                    for i in range(len(self.atk.dataset.X.columns)):
                        une_regle = [0] * 5
                        une_regle[2] = self.atk.dataset.X.columns[i]
                        if symbole[i] == "<":
                            une_regle[0] = f_rond(
                                float(min(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                            )
                            une_regle[1] = "<"
                            une_regle[3] = "<"
                            une_regle[4] = f_rond(float(valeurs[i]))
                            X1 = [
                                g
                                for g in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)
                                if g < valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].index.values)
                                if self.atk.dataset.X[self.atk.dataset.X.columns[i]][h] < valeurs[i]
                            ]
                            le_top.append(valeurs[i])
                            le_min.append(min(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                        elif symbole[i] == ">":
                            une_regle[0] = f_rond(float(valeurs[i]))
                            une_regle[1] = "<"
                            une_regle[3] = "<"
                            une_regle[4] = f_rond(
                                float(max(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                            )
                            X1 = [
                                g
                                for g in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)
                                if g > valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].index.values)
                                if self.atk.dataset.X[self.atk.dataset.X.columns[i]][h] > valeurs[i]
                            ]
                            le_min.append(valeurs[i])
                            le_top.append(max(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                        elif symbole[i] == "<=":
                            une_regle[0] = f_rond(
                                float(min(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                            )
                            une_regle[1] = "<="
                            une_regle[3] = "<="
                            une_regle[4] = f_rond(float(valeurs[i]))
                            le_top.append(valeurs[i])
                            le_min.append(min(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                            X1 = [
                                g
                                for g in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)
                                if g <= valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].index.values)
                                if self.atk.dataset.X[self.atk.dataset.X.columns[i]][h] <= valeurs[i]
                            ]
                            liste_index[i] = X2
                        elif symbole[i] == ">=":
                            une_regle[0] = f_rond(float(valeurs[i]))
                            une_regle[1] = "<="
                            une_regle[3] = "<="
                            une_regle[4] = f_rond(
                                float(max(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                            )
                            le_min.append(valeurs[i])
                            le_top.append(max(list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)))
                            X1 = [
                                g
                                for g in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].values)
                                if g >= valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(self.atk.dataset.X[self.atk.dataset.X.columns[i]].index.values)
                                if self.atk.dataset.X[self.atk.dataset.X.columns[i]][h] >= valeurs[i]
                            ]
                        liste_index[i] = X2
                        liste_val_histo[i] = X1
                        self.__all_rules.append(une_regle)
                    self.__save_rules = deepcopy(self.__all_rules)

                    une_carte_EV.children = generate_card(
                        liste_to_string_skope(self.__all_rules)
                    )

                    [new_y, marker] = fonction_beeswarm_shap(self.atk.dataset.X.columns[0])
                    essaim1.data[0].y = new_y
                    essaim1.data[0].x = self.__explanatory_values[self.atk.dataset.X.columns[0] + "_shap"]
                    essaim1.data[0].marker = marker

                    all_histograms = [histogram1]
                    if len(self.atk.dataset.X.columns) > 1:
                        all_histograms = [histogram1, histogram2]
                        [new_y, marker] = fonction_beeswarm_shap(self.atk.dataset.X.columns[1])
                        essaim2.data[0].y = new_y
                        essaim2.data[0].x = self.__explanatory_values[self.atk.dataset.X.columns[1] + "_shap"]
                        essaim2.data[0].marker = marker

                    if len(self.atk.dataset.X.columns) > 2:
                        all_histograms = [histogram1, histogram2, histogram3]
                        [new_y, marker] = fonction_beeswarm_shap(self.atk.dataset.X.columns[2])
                        essaim3.data[0].y = new_y
                        essaim3.data[0].x = self.__explanatory_values[self.atk.dataset.X.columns[2] + "_shap"]
                        essaim3.data[0].marker = marker

                    if len(self.atk.dataset.X.columns) == 1:
                        indices_respectent_skope = liste_index[0]
                    elif len(self.atk.dataset.X.columns) == 2:
                        indices_respectent_skope = [
                            a for a in liste_index[0] if a in liste_index[1]
                        ]
                    else:
                        indices_respectent_skope = [
                            a
                            for a in liste_index[0]
                            if a in liste_index[1] and a in liste_index[2]
                        ]
                    y_shape_skope = []
                    y_color_skope = []
                    y_opa_skope = []
                    for i in range(len(self.atk.dataset.X)):
                        if i in indices_respectent_skope:
                            y_shape_skope.append("circle")
                            y_color_skope.append("blue")
                            y_opa_skope.append(0.5)
                        else:
                            y_shape_skope.append("cross")
                            y_color_skope.append("grey")
                            y_opa_skope.append(0.5)
                    couleur_radio.v_model = "Selec actuelle"
                    fonction_changement_couleur(None)

                    if len(self.atk.dataset.X.columns) == 2:
                        accordion_skope.children = [
                            dans_accordion1_n,
                            dans_accordion2_n,
                        ]
                        dans_accordion1_n.children[0].children[0].children = (
                            "X1 (" + self.atk.dataset.X.columns[0] + ")"
                        )
                        dans_accordion2_n.children[0].children[0].children = (
                            "X2 (" + self.atk.dataset.X.columns[1] + ")"
                        )
                    elif len(self.atk.dataset.X.columns) == 3:
                        accordion_skope.children = [
                            dans_accordion1_n,
                            dans_accordion2_n,
                            dans_accordion3_n,
                        ]
                        dans_accordion1_n.children[0].children[0].children = (
                            "X1 (" + self.atk.dataset.X.columns[0] + ")"
                        )
                        dans_accordion2_n.children[0].children[0].children = (
                            "X2 (" + self.atk.dataset.X.columns[1] + ")"
                        )
                        dans_accordion3_n.children[0].children[0].children = (
                            "X3 (" + self.atk.dataset.X.columns[2] + ")"
                        )
                    elif len(self.atk.dataset.X.columns) == 1:
                        accordion_skope.children = [dans_accordion1_n]
                        dans_accordion1_n.children[0].children[0].children = (
                            "X1 (" + self.atk.dataset.X.columns[0] + ")"
                        )

                    slider_skope1.min = -10e10
                    slider_skope1.max = 10e10
                    slider_skope2.min = -10e10
                    slider_skope2.max = 10e10
                    slider_skope3.min = -10e10
                    slider_skope3.max = 10e10

                    slider_skope1.max = (
                        round(max(list(self.atk.dataset.X[self.atk.dataset.X.columns[0]].values)), 1)
                    ) * 100
                    slider_skope1.min = (
                        round(min(list(self.atk.dataset.X[self.atk.dataset.X.columns[0]].values)), 1)
                    ) * 100
                    slider_skope1.v_model = [
                        round(le_min[0], 1) * 100,
                        round(le_top[0], 1) * 100,
                    ]

                    [
                        slider_text_comb1.children[0].v_model,
                        slider_text_comb1.children[2].v_model,
                    ] = [slider_skope1.v_model[0] / 100, slider_skope1.v_model[1] / 100]

                    if len(self.atk.dataset.X.columns) > 1:
                        slider_skope2.max = (
                            max(list(self.atk.dataset.X[self.atk.dataset.X.columns[1]].values))
                        ) * 100
                        slider_skope2.min = (
                            min(list(self.atk.dataset.X[self.atk.dataset.X.columns[1]].values))
                        ) * 100
                        slider_skope2.v_model = [
                            round(le_min[1], 1) * 100,
                            round(le_top[1], 1) * 100,
                        ]
                        [
                            slider_text_comb2.children[0].v_model,
                            slider_text_comb2.children[2].v_model,
                        ] = [
                            slider_skope2.v_model[0] / 100,
                            slider_skope2.v_model[1] / 100,
                        ]

                    if len(self.atk.dataset.X.columns) > 2:
                        slider_skope3.description = self.atk.dataset.X.columns[2]
                        slider_skope3.max = (
                            max(list(self.atk.dataset.X[self.atk.dataset.X.columns[2]].values))
                        ) * 100
                        slider_skope3.min = (
                            min(list(self.atk.dataset.X[self.atk.dataset.X.columns[2]].values))
                        ) * 100
                        slider_skope3.v_model = [
                            round(le_min[2], 1) * 100,
                            round(le_top[2], 1) * 100,
                        ]
                        [
                            slider_text_comb3.children[0].v_model,
                            slider_text_comb3.children[2].v_model,
                        ] = [
                            slider_skope3.v_model[0] / 100,
                            slider_skope3.v_model[1] / 100,
                        ]

                    with histogram1.batch_update():
                        histogram1.data[0].x = list(
                            self.atk.dataset.X[re_transform_string(chaine_carac[0])[0][0]]
                        )
                        if len(histogram1.data) > 1:
                            histogram1.data[1].x = liste_val_histo[0]

                    if len(self.atk.dataset.X.columns) > 1:
                        with histogram2.batch_update():
                            histogram2.data[0].x = list(
                                self.atk.dataset.X[re_transform_string(chaine_carac[0])[0][1]]
                            )
                            if len(histogram2.data) > 1:
                                histogram2.data[1].x = liste_val_histo[1]
                            else:
                                histogram2.add_trace(
                                    go.Histogram(
                                        x=liste_val_histo[1],
                                        bingroup=1,
                                        marker_color="LightSkyBlue",
                                        opacity=0.7,
                                    )
                                )

                    if len(self.atk.dataset.X.columns) > 2:
                        with histogram3.batch_update():
                            histogram3.data[0].x = list(
                                self.atk.dataset.X[re_transform_string(chaine_carac[0])[0][2]]
                            )
                            if len(histogram3.data) > 1:
                                histogram3.data[1].x = liste_val_histo[2]
                            else:
                                histogram3.add_trace(
                                    go.Histogram(
                                        x=liste_val_histo[2],
                                        bingroup=1,
                                        marker_color="LightSkyBlue",
                                        opacity=0.7,
                                    )
                                )

                    modifier_tous_histograms(
                        slider_skope1.v_model[0], slider_skope1.v_model[1], 0
                    )

                    chaine_carac = transform_string_shap(skope_rules_clf_shap.rules_[0])
                    print(skope_rules_clf_shap.rules_)
                    texte_skopeEE.children[0].children[3].children = [
                        # str(skope_rules_clf.rules_[0])
                        # + "\n"
                        "p = "
                        + str(np.round(float(chaine_carac[1][0]) * 100, 5))
                        + "%"
                        + " r = "
                        + str(np.round(np.round(float(chaine_carac[1][1]), 2) * 100, 5))
                        + "%"
                        + " ext. of the tree ="
                        + chaine_carac[1][2]
                    ]
                    une_carte_EE.children = generate_card(chaine_carac[0])
                    fonction_scores_models(indices_respectent_skope)
                    self.selection = indices_respectent_skope
            slider_skope1.on_event("input", on_value_change_skope1)
            slider_skope2.on_event("input", on_value_change_skope2)
            slider_skope3.on_event("input", on_value_change_skope3)

            loading_models.class_ = "d-none"

            fonction_changement_couleur(None)

        def reinit_skope(*b):
            self.__all_rules = self.__save_rules
            fonction_validation_skope(None)
            fonction_scores_models(None)

        boutton_reinit_skope.on_event("click", reinit_skope)

        # here to see the values ​​of the selected points (EV and EE)
        out_selec = widgets.Output()
        with out_selec:
            display(
                HTML(
                    "Select points on the figure to see their values ​​here"
                )
            )
        out_selec_SHAP = widgets.Output()
        with out_selec_SHAP:
            display(
                HTML(
                    "Select points on the figure to see their values ​​here"
                )
            )
        out_selec_2 = v.Alert(
            class_="ma-1 pa-3",
            max_height="400px",
            style_="overflow: auto",
            elevation="3",
            children=[
                widgets.HBox(children=[out_selec, out_selec_SHAP]),
            ],
        )

        out_accordion = v.ExpansionPanels(
            class_="ma-2",
            children=[
                v.ExpansionPanel(
                    children=[
                        v.ExpansionPanelHeader(children=["Data selected"]),
                        v.ExpansionPanelContent(children=[out_selec_2]),
                    ]
                )
            ],
        )

        trouve_clusters = v.Btn(
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

        texte_slider_cluster = v.Html(
            tag="h3",
            class_="ma-3 mb-0",
            children=["Number of clusters " + str(slider_clusters.v_model)],
        )

        def fonct_texte_clusters(*b):
            texte_slider_cluster.children = [
                "Number of clusters " + str(slider_clusters.v_model)
            ]

        slider_clusters.on_event("input", fonct_texte_clusters)

        check_nb_clusters = v.Checkbox(
            v_model=True, label="Optimal number of clusters :", class_="ma-3"
        )

        def bool_nb_opti(*b):
            slider_clusters.disabled = check_nb_clusters.v_model

        check_nb_clusters.on_event("change", bool_nb_opti)

        partie_clusters = v.Layout(
            class_="d-flex flex-row",
            children=[
                trouve_clusters,
                check_nb_clusters,
                slider_clusters,
                texte_slider_cluster,
            ],
        )

        new_df = pd.DataFrame([], columns=["Region #", "Number of points"])
        colonnes = [{"text": c, "sortable": True, "value": c} for c in new_df.columns]
        resultats_clusters_table = v.DataTable(
            class_="w-100",
            style_="width : 100%",
            v_model=[],
            show_select=False,
            headers=colonnes,
            items=new_df.to_dict("records"),
            item_value="Region #",
            item_key="Region #",
        )
        resultats_clusters = v.Row(
            children=[
                v.Layout(
                    class_="flex-grow-0 flex-shrink-0",
                    children=[v.Btn(class_="d-none", elevation=0, disabled=True)],
                ),
                v.Layout(
                    class_="flex-grow-1 flex-shrink-0",
                    children=[resultats_clusters_table],
                ),
            ],
        )

        # allows you to make clusters on one or the other of the spaces
        def fonction_clusters(*b):
            loading_clusters.class_ = "d-flex"
            if check_nb_clusters.v_model:
                result = fonction_auto_clustering(X, self.__explanatory_values, 3, True)
            else:
                nb_clusters = slider_clusters.v_model
                result = fonction_auto_clustering(X, self.__explanatory_values, nb_clusters, False)
            self.__result_dyadic_clustering = result
            labels = result[1]
            self.__Y_auto = labels
            with fig1.batch_update():
                fig1.data[0].marker.color = labels
                fig1.update_traces(marker=dict(showscale=False))
            with fig2.batch_update():
                fig2.data[0].marker.color = labels
            with fig1_3D.batch_update():
                fig1_3D.data[0].marker.color = labels
                fig1_3D.update_traces(marker=dict(showscale=False))
            with fig2_3D.batch_update():
                fig2_3D.data[0].marker.color = labels
            labels_regions = result[0]
            new_df = []
            for i in range(len(labels_regions)):
                new_df.append(
                    [
                        i + 1,
                        len(labels_regions[i]),
                        str(round(len(labels_regions[i]) / len(X) * 100, 5)) + "%",
                    ]
                )
            new_df = pd.DataFrame(
                new_df,
                columns=["Region #", "Number of points", "Percentage of the dataset"],
            )
            donnees = new_df.to_dict("records")
            colonnes = [
                {"text": c, "sortable": False, "value": c} for c in new_df.columns
            ]
            resultats_clusters_table = v.DataTable(
                class_="w-100",
                style_="width : 100%",
                show_select=False,
                single_select=True,
                v_model=[],
                headers=colonnes,
                items=donnees,
                item_value="Region #",
                item_key="Region #",
            )
            all_chips = []
            all_radio = []
            N_etapes = len(labels_regions)
            Multip = 100
            debut = 0
            fin = (N_etapes * Multip - 1) * (1 + 1 / (N_etapes - 1))
            pas = (N_etapes * Multip - 1) / (N_etapes - 1)
            scale_couleurs = np.arange(debut, fin, pas)
            a = 0
            for i in scale_couleurs:
                color = sns.color_palette("viridis", N_etapes * Multip).as_hex()[
                    round(i)
                ]
                all_chips.append(v.Chip(class_="rounded-circle", color=color))
                all_radio.append(v.Radio(class_="mt-4", value=str(a)))
                a += 1
            all_radio[-1].class_ = "mt-4 mb-0 pb-0"
            partie_radio = v.RadioGroup(
                v_model=None,
                class_="mt-10 ml-7",
                style_="width : 10%",
                children=all_radio,
            )
            partie_chips = v.Col(
                class_="w-10 ma-5 mt-10 mb-12 pb-4 d-flex flex-column justify-space-between",
                style_="width : 10%",
                children=all_chips,
            )
            resultats_clusters = v.Row(
                children=[
                    v.Layout(
                        class_="flex-grow-0 flex-shrink-0", children=[partie_radio]
                    ),
                    v.Layout(
                        class_="flex-grow-0 flex-shrink-0", children=[partie_chips]
                    ),
                    v.Layout(
                        class_="flex-grow-1 flex-shrink-0",
                        children=[resultats_clusters_table],
                    ),
                ],
            )
            partie_selection.children = partie_selection.children[:-1] + [
                resultats_clusters
            ]

            couleur_radio.v_model = "Clustering auto"

            partie_selection.children[-1].children[0].children[0].on_event(
                "change", fonction_choix_cluster
            )
            loading_clusters.class_ = "d-none"
            return N_etapes

        def fonction_choix_cluster(widget, event, data):
            result = self.__result_dyadic_clustering
            labels = result[1]
            indice = partie_selection.children[-1].children[0].children[0].v_model
            liste = [i for i, d in enumerate(labels) if d == float(indice)]
            selection_fn(None, None, None, liste)
            couleur_radio.v_model = "Clustering auto"
            fonction_changement_couleur(opacity=False)

        trouve_clusters.on_event("click", fonction_clusters)

        # function which is called as soon as the points are selected (step 1)
        def selection_fn(trace, points, selector, *args):
            if len(args) > 0:
                liste = args[0]
                les_points = liste
            else:
                les_points = points.point_inds
            self.selection = les_points
            if len(les_points) == 0:
                card_selec.children[0].children[1].children = "0 point !"
                texte_selec.value = texte_base_debut
                return
            card_selec.children[0].children[1].children = (
                str(len(les_points))
                + " points selected ("
                + str(round(len(les_points) / len(self.atk.dataset.X) * 100, 2))
                + "% of the overall)"
            )
            texte_selec.value = (
                texte_base
                + str(len(les_points))
                + " points selected ("
                + str(round(len(les_points) / len(self.atk.dataset.X) * 100, 2))
                + "% of the overall)"
            )
            opa = []
            self.__y_train = []
            for i in range(len(fig2.data[0].x)):
                if i in les_points:
                    opa.append(1)
                    self.__y_train.append(1)
                else:
                    opa.append(0.1)
                    self.__y_train.append(0)
            with fig2.batch_update():
                fig2.data[0].marker.opacity = opa
            with fig1.batch_update():
                fig1.data[0].marker.opacity = opa

            X_train = self.atk.dataset.X.copy()
            self.__SHAP_train = self.__explanatory_values.copy()

            X_mean = (
                pd.DataFrame(
                    X_train.iloc[self.selection, :].mean(axis=0).values.reshape(1, -1),
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
                    self.__SHAP_train.iloc[self.selection, :]
                    .mean(axis=0)
                    .values.reshape(1, -1),
                    columns=self.__SHAP_train.columns,
                )
                .round(2)
                .rename(index={0: "Mean of the selection"})
            )
            SHAP_mean_tot = (
                pd.DataFrame(
                    self.__SHAP_train.mean(axis=0).values.reshape(1, -1),
                    columns=self.__SHAP_train.columns,
                )
                .round(2)
                .rename(index={0: "Mean of the whole dataset"})
            )
            SHAP_mean = pd.concat([SHAP_mean, SHAP_mean_tot], axis=0)

            with out_selec:
                clear_output()
                display(HTML("<h4> Values Space </h4>"))
                display(HTML("<h5>Average point of the selection :<h5>"))
                display(HTML(X_mean.to_html()))
                display(HTML("<h5>Points selected :<h5>"))
                display(HTML(X_train.iloc[self.selection, :].to_html(index=False)))
            with out_selec_SHAP:
                clear_output()
                display(HTML("<h4> Explanatory Space </h4>"))
                display(HTML("<h5>Average point of the selection :<h5>"))
                display(HTML(SHAP_mean.to_html()))
                display(HTML("<h5>Points selected :<h5>"))
                display(HTML(self.__SHAP_train.iloc[self.selection, :].to_html(index=False)))

        # function that is called when validating a tile to add it to the set of regions
        def fonction_validation_une_tuile(*args):
            if len(args) == 0:
                pass
            else:
                if self.__model_choice == None:
                    nom_model = None
                    score_model = [1] * len(self.__score_models[0])
                    indice_model = -1
                else:
                    nom_model = self.sub_models[self.__model_choice].__class__.__name__
                    score_model = self.__score_models[self.__model_choice]
                    indice_model = self.__model_choice
                a = [0] * 10
                if self.__all_rules == None or self.__all_rules == [
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                ]:
                    return
                nouvelle_tuile = self.atk.dataset.X[
                    (self.atk.dataset.X[self.__all_rules[0][2]] >= self.__all_rules[0][0])
                    & (self.atk.dataset.X[self.__all_rules[0][2]] <= self.__all_rules[0][4])
                ].index
                for i in range(1, len(self.__all_rules)):
                    X_temp = self.atk.dataset.X[
                        (self.atk.dataset.X[self.__all_rules[i][2]] >= self.__all_rules[i][0])
                        & (self.atk.dataset.X[self.__all_rules[i][2]] <= self.__all_rules[i][4])
                    ].index
                    nouvelle_tuile = [g for g in nouvelle_tuile if g in X_temp]
                self.__list_of_sub_models.append([nom_model, score_model, indice_model])
                # here we will force so that all the points of the new tile belong only to it: we will modify the existing tiles
                self.__list_of_regions = _conflict_handler(self.__list_of_regions, nouvelle_tuile)
                self.__list_of_regions.append(nouvelle_tuile)
            for i in range(len(self.__color_regions)):
                if i in self.selection:
                    self.__color_regions[i] = len(self.__list_of_regions)

            toute_somme = 0
            temp = []
            score_tot = 0
            score_tot_glob = 0
            autre_toute_somme = 0
            for i in range(len(self.__list_of_regions)):
                if self.__list_of_sub_models[i][0] == None:
                    temp.append(
                        [
                            i + 1,
                            len(self.__list_of_regions[i]),
                            np.round(len(self.__list_of_regions[i]) / len(self.atk.dataset.X) * 100, 2),
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
                            len(self.__list_of_regions[i]),
                            np.round(len(self.__list_of_regions[i]) / len(self.atk.dataset.X) * 100, 2),
                            self.__list_of_sub_models[i][0],
                            self.__list_of_sub_models[i][1][0],
                            self.__list_of_sub_models[i][1][1],
                            str(self.__list_of_sub_models[i][1][2]) + "%",
                        ]
                    )
                    score_tot += self.__list_of_sub_models[i][1][0] * len(self.__list_of_regions[i])
                    score_tot_glob += self.__list_of_sub_models[i][1][1] * len(
                        self.__list_of_regions[i]
                    )
                    autre_toute_somme += len(self.__list_of_regions[i])
                toute_somme += len(self.__list_of_regions[i])
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
                donnees = new_df[:-1].to_dict("records")
                total = new_df[-1:].iloc[:, 1:].to_dict("records")
                colonnes = [
                    {"text": c, "sortable": True, "value": c} for c in new_df.columns
                ]
                colonnes_total = [
                    {"text": c, "sortable": True, "value": c}
                    for c in new_df.columns[1:]
                ]
                table_donnes = v.DataTable(
                    v_model=[],
                    show_select=True,
                    headers=colonnes,
                    items=donnees,
                    item_value="Region #",
                    item_key="Region #",
                )
                table_total = v.DataTable(
                    v_model=[],
                    headers=colonnes_total,
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

                def fonction_suppression_tuiles(*b):
                    if table_donnes.v_model == []:
                        return
                    taille = len(table_donnes.v_model)
                    a = 0
                    for i in range(taille):
                        indice = table_donnes.v_model[i]["Region #"] - 1
                        self.__list_of_regions.pop(indice - a)
                        self.__list_of_sub_models.pop(indice - a)
                        fonction_validation_une_tuile()
                        self.__all_tiles_rules.pop(indice - a)
                        a += 1
                    couleur_radio.v_model = "Régions"
                    fonction_changement_couleur()

                supprimer_toutes_les_tuiles.on_event(
                    "click", fonction_suppression_tuiles
                )

                display(ensemble_tables)

            a = [0] * 10
            pas_avoir = [a, a, a, a, a, a, a, a, a, a]
            if self.__all_rules != pas_avoir:
                self.__all_tiles_rules.append(deepcopy(self.__all_rules))

        valider_une_region.on_event("click", fonction_validation_une_tuile)
        button_valider_skope.on_event("click", fonction_validation_skope)

        fig1.data[0].on_selection(selection_fn)
        fig2.data[0].on_selection(selection_fn)

        def fonction_fig_size(*args):
            with fig1.batch_update():
                fig1.layout.width = int(fig_size.v_model)
            with fig2.batch_update():
                fig2.layout.width = int(fig_size.v_model)
            with fig1_3D.batch_update():
                fig1_3D.layout.width = int(fig_size.v_model)
            with fig2_3D.batch_update():
                fig2_3D.layout.width = int(fig_size.v_model)
            for i in range(len(all_histograms)):
                with all_histograms[i].batch_update():
                    all_histograms[i].layout.width = 0.9 * int(fig_size.v_model)
                with all_beeswarms[i].batch_update():
                    all_beeswarms[i].layout.width = 0.9 * int(fig_size.v_model)

        fig_size.on_event("input", fonction_fig_size)

        boutton_add_skope = v.Btn(
            class_="ma-4 pa-1 mb-0",
            children=[v.Icon(children=["mdi-plus"]), "Add a rule"],
        )

        widget_list_add_skope = v.Select(
            class_="mr-3 mb-0",
            items=["/"],
            v_model="/",
        )

        add_group = widgets.HBox([boutton_add_skope, widget_list_add_skope])

        def fonction_add_skope(*b):
            nouvelle_regle = [0] * 5
            colonne = widget_list_add_skope.v_model
            if self.__other_columns == None:
                return
            self.__other_columns = [a for a in self.__other_columns if a != colonne]
            nouvelle_regle[2] = colonne
            nouvelle_regle[0] = round(min(list(self.atk.dataset.X[colonne].values)), 1)
            nouvelle_regle[1] = "<="
            nouvelle_regle[3] = "<="
            nouvelle_regle[4] = round(max(list(self.atk.dataset.X[colonne].values)), 1)
            self.__all_rules.append(nouvelle_regle)
            une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))

            new_valider_change = v.Btn(
                class_="ma-3",
                children=[
                    v.Icon(class_="mr-2", children=["mdi-check"]),
                    "Validate the change",
                ],
            )

            new_slider_skope = v.RangeSlider(
                class_="ma-3",
                v_model=[nouvelle_regle[0] * 100, nouvelle_regle[4] * 100],
                min=nouvelle_regle[0] * 100,
                max=nouvelle_regle[4] * 100,
                step=1,
                label=nouvelle_regle[2],
            )

            new_histogram = go.FigureWidget(
                data=[
                    go.Histogram(
                        x=self.atk.dataset.X[colonne].values,
                        bingroup=1,
                        nbinsx=nombre_bins,
                        marker_color="grey",
                    )
                ]
            )
            new_histogram.update_layout(
                barmode="overlay",
                bargap=0.1,
                width=0.9 * int(fig_size.v_model),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=150,
            )

            new_histogram.add_trace(
                go.Histogram(
                    x=self.atk.dataset.X[colonne].values,
                    bingroup=1,
                    nbinsx=nombre_bins,
                    marker_color="LightSkyBlue",
                    opacity=0.6,
                )
            )
            new_histogram.add_trace(
                go.Histogram(
                    x=self.atk.dataset.X[colonne].values,
                    bingroup=1,
                    nbinsx=nombre_bins,
                    marker_color="blue",
                )
            )

            all_histograms.append(new_histogram)

            def new_fonction_change_valider(*change):
                ii = -1
                for i in range(len(self.__all_rules)):
                    if self.__all_rules[i][2] == colonne_2:
                        ii = int(i)
                a = deepcopy(float(new_slider_skope.v_model[0] / 100))
                b = deepcopy(float(new_slider_skope.v_model[1] / 100))
                self.__all_rules[ii][0] = a
                self.__all_rules[ii][4] = b
                une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
                tout_modifier_graphique()
                fonction_scores_models(None)

            new_valider_change.on_event("click", new_fonction_change_valider)

            new_bout_temps_reel_graph = v.Checkbox(
                v_model=False, label="Real-time updates on the graphs", class_="ma-3"
            )

            new_slider_text_comb = v.Layout(
                children=[
                    v.TextField(
                        style_="max-width:100px",
                        v_model=new_slider_skope.v_model[0] / 100,
                        hide_details=True,
                        type="number",
                        density="compact",
                    ),
                    new_slider_skope,
                    v.TextField(
                        style_="max-width:100px",
                        v_model=new_slider_skope.v_model[1] / 100,
                        hide_details=True,
                        type="number",
                        density="compact",
                    ),
                ],
            )

            def new_update_validate(*args):
                if new_bout_temps_reel_graph.v_model:
                    new_valider_change.disabled = True
                else:
                    new_valider_change.disabled = False

            new_bout_temps_reel_graph.on_event("change", new_update_validate)

            new_deux_fin = widgets.HBox([new_valider_change, new_bout_temps_reel_graph])

            new_ens_slider_histo = widgets.VBox(
                [new_slider_text_comb, new_histogram, new_deux_fin]
            )

            colonne_shap = colonne + "_shap"
            y_histo_shap = [0] * len(self.__explanatory_values)
            new_essaim = go.FigureWidget(
                data=[go.Scatter(x=self.__explanatory_values[colonne_shap], y=y_histo_shap, mode="markers")]
            )
            new_essaim.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=200,
                width=0.9 * int(fig_size.v_model),
            )
            new_essaim.update_yaxes(visible=False, showticklabels=False)
            [new_y, marker] = fonction_beeswarm_shap(colonne)
            new_essaim.data[0].y = new_y
            new_essaim.data[0].x = self.__explanatory_values[colonne_shap]
            new_essaim.data[0].marker = marker

            new_choix_couleur_essaim = v.Row(
                class_="pt-3 mt-0 ml-4",
                children=[
                    "Valeur de Xi",
                    v.Switch(
                        class_="ml-3 mr-2 mt-0 pt-0",
                        v_model=False,
                        label="",
                    ),
                    "Current selection",
                ],
            )

            def new_changement_couleur_essaim_shap(*args):
                if new_choix_couleur_essaim.children[1].value == False:
                    marker = fonction_beeswarm_shap(self.__all_rules[len(self.__all_rules) - 1][2])[1]
                    new_essaim.data[0].marker = marker
                    new_essaim.update_traces(marker=dict(showscale=True))
                else:
                    modifier_tous_histograms(
                        new_slider_skope.v_model[0] / 100,
                        new_slider_skope.v_model[1] / 100,
                        0,
                    )
                    new_essaim.update_traces(marker=dict(showscale=False))

            new_choix_couleur_essaim.children[1].on_event(
                "change", new_changement_couleur_essaim_shap
            )

            new_essaim_tot = widgets.VBox([new_choix_couleur_essaim, new_essaim])
            new_essaim_tot.layout.margin = "0px 0px 0px 20px"

            all_beeswarms_total.append(new_essaim_tot)

            if not check_beeswarm.v_model:
                new_essaim_tot.layout.display = "none"

            all_beeswarms.append(new_essaim)

            all_color_choosers_beeswarms.append(new_choix_couleur_essaim)

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
                colonne_2 = new_slider_skope.label
                ii = 0
                for i in range(len(self.__all_rules)):
                    if self.__all_rules[i][2] == colonne_2:
                        ii = i
                        break
                elements_final_accordion.pop(ii)
                all_beeswarms_total.pop(ii)
                all_histograms.pop(ii)
                self.__all_rules.pop(ii)
                all_beeswarms.pop(ii)
                all_color_choosers_beeswarms.pop(ii)
                self.__other_columns = [colonne_2] + self.__other_columns
                une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))
                widget_list_add_skope.items = self.__other_columns
                widget_list_add_skope.v_model = self.__other_columns[0]
                accordion_skope.children = [
                    a for a in accordion_skope.children if a != new_dans_accordion_n
                ]
                for i in range(ii, len(accordion_skope.children)):
                    col = "X" + str(i + 1) + " (" + self.__all_rules[i][2] + ")"
                    accordion_skope.children[i].titles = [col]
                tout_modifier_graphique()

            new_b_delete_skope.on_event("click", new_delete_skope)

            new_dans_accordion = widgets.HBox(
                [new_ens_slider_histo, new_essaim_tot, new_b_delete_skope],
                layout=Layout(align_items="center"),
            )

            new_dans_accordion_n = v.ExpansionPanels(
                class_="ma-2 mt-0 mb-1",
                children=[
                    v.ExpansionPanel(
                        children=[
                            v.ExpansionPanelHeader(children=["Xn"]),
                            v.ExpansionPanelContent(children=[new_dans_accordion]),
                        ]
                    )
                ],
            )

            accordion_skope.children = [*accordion_skope.children, new_dans_accordion_n]
            nom_colcol = "X" + str(len(accordion_skope.children)) + " (" + colonne + ")"
            accordion_skope.children[-1].children[0].children[0].children = nom_colcol

            with new_histogram.batch_update():
                new_list = [
                    g
                    for g in list(self.atk.dataset.X[colonne].values)
                    if g >= new_slider_skope.v_model[0] / 100
                    and g <= new_slider_skope.v_model[1] / 100
                ]
                new_histogram.data[1].x = new_list

                colonne_2 = new_slider_skope.label
                new_list_regle = self.atk.dataset.X.index[
                    self.atk.dataset.X[colonne_2].between(
                        new_slider_skope.v_model[0] / 100,
                        new_slider_skope.v_model[1] / 100,
                    )
                ].tolist()
                new_list_tout = new_list_regle.copy()
                for i in range(1, len(self.__all_rules)):
                    new_list_temp = self.atk.dataset.X.index[
                        self.atk.dataset.X[self.__all_rules[i][2]].between(
                            self.__all_rules[i][0], self.__all_rules[i][4]
                        )
                    ].tolist()
                    new_list_tout = [g for g in new_list_tout if g in new_list_temp]
                new_list_tout_new = self.atk.dataset.X[colonne_2][new_list_tout]
                new_histogram.data[2].x = new_list_tout_new

            def new_on_value_change_skope(*b1):
                new_slider_text_comb.children[0].v_model = (
                    new_slider_skope.v_model[0] / 100
                )
                new_slider_text_comb.children[2].v_model = (
                    new_slider_skope.v_model[1] / 100
                )
                colonne_2 = new_slider_skope.label
                ii = 0
                for i in range(len(self.__all_rules)):
                    if self.__all_rules[i][2] == colonne_2:
                        ii = i
                        break
                new_list = [
                    g
                    for g in list(self.atk.dataset.X[colonne_2].values)
                    if g >= new_slider_skope.v_model[0] / 100
                    and g <= new_slider_skope.v_model[1] / 100
                ]
                with new_histogram.batch_update():
                    new_histogram.data[1].x = new_list
                if self.__valider_bool:
                    modifier_tous_histograms(
                        new_slider_skope.v_model[0] / 100,
                        new_slider_skope.v_model[1] / 100,
                        ii,
                    )
                if new_bout_temps_reel_graph.v_model:
                    self.__all_rules[ii - 1][0] = float(
                        deepcopy(new_slider_skope.v_model[0] / 100)
                    )
                    self.__all_rules[ii - 1][4] = float(
                        deepcopy(new_slider_skope.v_model[1] / 100)
                    )
                    une_carte_EV.children = generate_card(
                        liste_to_string_skope(self.__all_rules)
                    )
                    tout_modifier_graphique()

            new_slider_skope.on_event("input", new_on_value_change_skope)

            elements_final_accordion.append(new_dans_accordion)

        fonction_validation_une_tuile()

        boutton_add_skope.on_event("click", fonction_add_skope)

        param_EV = v.Menu(
            v_slots=[
                {
                    "name": "activator",
                    "variable": "props",
                    "children": v.Btn(
                        v_on="props.on",
                        icon=True,
                        size="x-large",
                        children=[v.Icon(children=["mdi-cogs"], size="large")],
                        class_="ma-2 pa-3",
                        elevation="3",
                    ),
                }
            ],
            children=[
                v.Card(
                    class_="pa-4",
                    rounded=True,
                    children=[params_proj_EV],
                    min_width="500",
                )
            ],
            v_model=False,
            close_on_content_click=False,
            offset_y=True,
        )

        param_EV.v_slots[0]["children"].children = [
            add_tooltip(
                param_EV.v_slots[0]["children"].children[0],
                "Settings of the projection in the Values Space",
            )
        ]

        param_EE = v.Menu(
            v_slots=[
                {
                    "name": "activator",
                    "variable": "props",
                    "children": v.Btn(
                        v_on="props.on",
                        icon=True,
                        size="x-large",
                        children=[v.Icon(children=["mdi-cogs"], size="large")],
                        class_="ma-2 pa-3",
                        elevation="3",
                    ),
                }
            ],
            children=[
                v.Card(
                    class_="pa-4",
                    rounded=True,
                    children=[params_proj_EE],
                    min_width="500",
                )
            ],
            v_model=False,
            close_on_content_click=False,
            offset_y=True,
        )
        param_EE.v_slots[0]["children"].children = [
            add_tooltip(
                param_EE.v_slots[0]["children"].children[0],
                "Settings of the projection in the Explanatory Space",
            )
        ]

        projEV_et_load = widgets.HBox(
            [
                EV_proj,
                v.Layout(children=[param_EV]),
                out_loading1,
            ]
        )
        projEE_et_load = widgets.HBox(
            [
                EE_proj,
                v.Layout(children=[param_EE]),
                out_loading2,
            ]
        )

        bouton_reinit_opa = v.Btn(
            icon=True,
            children=[v.Icon(children=["mdi-opacity"])],
            class_="ma-2 ml-6 pa-3",
            elevation="3",
        )

        bouton_reinit_opa.children = [
            add_tooltip(
                bouton_reinit_opa.children[0],
                "Reset the opacity of the points",
            )
        ]

        def fonction_reinit_opa(*args):
            with fig1.batch_update():
                fig1.data[0].marker.opacity = 1
            with fig2.batch_update():
                fig2.data[0].marker.opacity = 1

        bouton_reinit_opa.on_event("click", fonction_reinit_opa)

        items = [{'text': "SHAP", 'disabled': True},
                {'text': "LIME", 'disabled': True}]
        if type(self.explanation) != str:
            items = [{'text': "Imported", 'disabled': False}] + items
            item_default = "Imported"
        else :
            item_default = self.explanation

        for item in items:
            if item['text'] == item_default:
                item['disabled'] = False

        choose_explanation = v.Select(
            label="Explainability method",
            items=items,
            v_model=item_default,
            class_="ma-2 ml-6",
            style_="width: 150px",
            disabled = False,
        )

        def fonction_choose_explanation(widget, event, data):
            exp_val = eval('self.atk.dataset.explain[\"'+data+'\"]')
            if self.dim_red['EE'][self.__explanation][self.__projection] == None:
                dim_red = compute.DimensionalityReduction(EE_proj.v_model, True)
                self.dim_red['EE'][self.__explanation][self.__projection] = [dim_red.compute(exp_val, 2), dim_red.compute(exp_val, 3)]

            with fig1.batch_update():
                fig1.data[0].x = self.dim_red['EV'][self.__projection][0][0]
                fig1.data[0].y = self.dim_red['EV'][self.__projection][0][1]
            with fig1_3D.batch_update():
                fig1_3D.data[0].x = self.dim_red['EV'][self.__projection][1][0]
                fig1_3D.data[0].y = self.dim_red['EV'][self.__projection][1][1]
                fig1_3D.data[0].z = self.dim_red['EV'][self.__projection][0][2]
            with fig2.batch_update():
                fig2.data[0].x = self.dim_red['EE'][self.__explanation][self.__projection][0][0]
                fig2.data[0].y = self.dim_red['EE'][self.__explanation][self.__projection][0][1]
            with fig2_3D.batch_update():
                fig2_3D.data[0].x = self.dim_red['EE'][self.__explanation][self.__projection][1][0]
                fig2_3D.data[0].y = self.dim_red['EE'][self.__explanation][self.__projection][1][1]
                fig2_3D.data[0].z = self.dim_red['EE'][self.__explanation][self.__projection][0][2]
        
        choose_explanation.on_event("change", fonction_choose_explanation)

        def prog_other(titre):
            progress_other = v.ProgressLinear(
                style_="width: 80%",
                v_model=0,
                color="primary",
                height="15",
                striped=True,
            )
            widget = v.Col(
                class_="d-flex flex-column align-center",
                children=[
                        v.Html(
                            tag="h3",
                            class_="mb-3",
                            children=["Compute " + titre + " values"],
                    ),
                    progress_other,
                    v.TextField(
                        class_="w-100",
                        style_="width: 100%",
                        v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
                        readonly=True,
                    ),
                    v.Btn(
                        children=[v.Icon(class_="mr-2", children=["mdi-calculator-variant"]), "Compute values"],
                        class_="ma-2 ml-6 pa-3",
                        elevation="3",
                        v_model=titre,
                        color="primary",
                    ),
                ],
            )
            return widget
        
        new_prog_SHAP = prog_other("SHAP")
        new_prog_LIME = prog_other("LIME")

        if self.calculus == True:
            if self.explanation == "SHAP":
                new_prog_SHAP.children[1].v_model = 100
                new_prog_SHAP.children[2].v_model = "Computations already done !"
                new_prog_SHAP.children[-1].disabled = True
            elif self.explanation == "LIME":
                new_prog_LIME.children[1].v_model = 100
                new_prog_LIME.children[2].v_model = "Computations already done !"
                new_prog_LIME.children[-1].disabled = True

        def function_validation_explanation(widget, event, data):
            if widget.v_model == "SHAP":
                self.__compute_SHAP = LongTask.compute_SHAP(self.atk.dataset.X, self.atk.dataset.X_all, self.atk.dataset.model)
                widgets.jslink((new_prog_SHAP.children[1], "v_model"), (self.__compute_SHAP.progress_widget, "v_model"))
                widgets.jslink((new_prog_SHAP.children[2], "v_model"), (self.__compute_SHAP.text_widget, "v_model"))
                widgets.jslink((new_prog_SHAP.children[-1], "color"), (self.__compute_SHAP.done_widget, "v_model"))
                self.__compute_SHAP.compute_in_thread()
                new_prog_SHAP.children[-1].disabled = True
            if widget.v_model == "LIME":
                self.__compute_LIME = LongTask.compute_LIME(self.atk.dataset.X, self.atk.dataset.X_all, self.atk.dataset.model)
                widgets.jslink((new_prog_LIME.children[1], "v_model"), (self.__compute_LIME.progress_widget, "v_model"))
                widgets.jslink((new_prog_LIME.children[2], "v_model"), (self.__compute_LIME.text_widget, "v_model"))
                widgets.jslink((new_prog_LIME.children[-1], "color"), (self.__compute_LIME.done_widget, "v_model"))
                self.__compute_LIME.compute_in_thread()
                new_prog_LIME.children[-1].disabled = True
                
        def ok_SHAP(*args):
            self.atk.dataset.explain["SHAP"] = self.__compute_SHAP.value
            items = choose_explanation.items.copy()
            for item in items:
                if item['text'] == "SHAP":
                    item['disabled'] = False
            choose_explanation.items = items.copy() + [{'text': "Update", 'disabled': False}]
            choose_explanation.items = choose_explanation.items[:-1]

        def ok_LIME(*args):
            self.atk.dataset.explain["SHAP"] = self.__compute_LIME.value
            items = choose_explanation.items.copy()
            for item in items:
                if item['text'] == "LIME":
                    item['disabled'] = False
            choose_explanation.items = items.copy() + [{'text': "Update", 'disabled': False}]
            choose_explanation.items = choose_explanation.items[:-1]


        new_prog_SHAP.children[-1].observe(ok_SHAP, "color")
        new_prog_LIME.children[-1].observe(ok_LIME, "color")

        new_prog_SHAP.children[-1].on_event("click", function_validation_explanation)
        new_prog_LIME.children[-1].on_event("click", function_validation_explanation)

        time_computing = v.Card(
            class_="m-0 p-0",
            elevation="0",
            children=[
                v.Tabs(
                    class_="w-100",
                    v_model="tabs",
                    children=[
                        v.Tab(value="one", children=["SHAP"]),
                        v.Tab(value="two", children=["LIME"]),
                    ],
                ),
                v.CardText(
                    class_="w-100",
                    children=[
                        v.Window(
                            class_="w-100",
                            v_model="tabs",
                            children=[
                                v.WindowItem(value=0, children=[new_prog_SHAP]),
                                v.WindowItem(value=1, children=[new_prog_LIME]),
                            ],
                        )
                    ],
                ),
            ],
        )

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
                        children=[v.Icon(children=["mdi-timer-sand"], size="large")],
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
            

        boutons = widgets.HBox(
            [
                dimension_projection_text,
                v.Layout(
                    class_="pa-2 ma-2",
                    elevation="3",
                    children=[
                        add_tooltip(
                            v.Icon(
                                children=["mdi-format-color-fill"],
                                class_="mr-4",
                            ),
                            "Color of the points",
                        ),
                        couleur_radio,
                        bouton_reinit_opa,
                        choose_explanation,
                        choose_computing,
                    ],
                ),
                v.Layout(class_="mt-5", children=[projEV_et_load, projEE_et_load]),
            ],
            layout=Layout(
                width="100%",
                display="flex",
                flex_flow="row",
                justify_content="space-around",
            ),
        )
        figures = widgets.VBox([fig_2D_ou_3D], layout=Layout(width="100%"))

        check_beeswarm = v.Checkbox(
            v_model=True,
            label="Show Shapley's beeswarm plots",
            class_="ma-1 mr-3",
        )

        def fonction_check_beeswarm(*b):
            if not check_beeswarm.v_model:
                for i in range(len(all_beeswarms_total)):
                    all_beeswarms_total[i].layout.display = "none"
            else:
                for i in range(len(all_beeswarms_total)):
                    all_beeswarms_total[i].layout.display = "block"

        check_beeswarm.on_event("change", fonction_check_beeswarm)

        buttons_skope = v.Layout(
            class_="d-flex flex-row",
            children=[
                button_valider_skope,
                boutton_reinit_skope,
                v.Spacer(),
                check_beeswarm,
            ],
        )

        deux_b = widgets.VBox([buttons_skope, texte_skope])

        bouton_magique = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(children=["mdi-creation"], class_="mr-3"),
                "Magic button",
            ],
        )

        partie_magique = v.Layout(
            class_="d-flex flex-row justify-center align-center",
            children=[
                v.Spacer(),
                bouton_magique,
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

        def fonction_checkbox_magique(*args):
            if partie_magique.children[2].v_model:
                partie_magique.children[3].disabled = False
            else:
                partie_magique.children[3].disabled = True

        partie_magique.children[2].on_event("change", fonction_checkbox_magique)

        def find_best_score():
            a = 1000
            for i in range(len(self.__score_models)):
                score = self.__score_models[i][0]
                if score < a:
                    a = score
                    indice = i
            return indice

        def fonction_bouton_magique(*args):
            demo = partie_magique.children[2].v_model
            if demo == False:
                etapes.children[0].v_model = 3
            N_etapes = fonction_clusters(None)
            if demo:
                tempo = int(partie_magique.children[3].v_model) / 10
                if tempo < 0:
                    tempo = 0
            else:
                tempo = 0
            time.sleep(tempo)
            for i in range(N_etapes):
                partie_selection.children[-1].children[0].children[0].v_model = str(i)
                fonction_choix_cluster(None, None, None)
                time.sleep(tempo)
                if demo:
                    etapes.children[0].v_model = 1
                time.sleep(tempo)
                fonction_validation_skope(None)
                time.sleep(tempo)
                if demo:
                    etapes.children[0].v_model = 2
                time.sleep(tempo)
                indice = find_best_score()
                mods.children[indice].children[0].color = "blue lighten-4"
                changement(None, None, None, False)
                time.sleep(tempo)
                mods.children[indice].children[0].color = "white"
                if demo:
                    etapes.children[0].v_model = 3
                time.sleep(tempo)
                fonction_validation_une_tuile(None)
                time.sleep(tempo)
                if i != N_etapes - 1:
                    if demo:
                        etapes.children[0].v_model = 0
                    time.sleep(tempo)
            couleur_radio.v_model = "Régions"
            fonction_changement_couleur(None)

        bouton_magique.on_event("click", fonction_bouton_magique)

        loading_clusters = v.ProgressLinear(
            indeterminate=True, class_="ma-3", style_="width : 100%"
        )

        loading_clusters.class_ = "d-none"

        partie_selection = v.Col(
            children=[
                card_selec,
                out_accordion,
                partie_clusters,
                loading_clusters,
                resultats_clusters,
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

        partie_skope = widgets.VBox([deux_b, accordion_skope, add_group])
        partie_modele = widgets.VBox([loading_models, mods])
        partie_toutes_regions = widgets.VBox([selection, table_regions])

        etapes = v.Card(
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
                                v.WindowItem(value=0, children=[partie_selection]),
                                v.WindowItem(value=1, children=[partie_skope]),
                                v.WindowItem(value=2, children=[partie_modele]),
                                v.WindowItem(value=3, children=[partie_toutes_regions]),
                            ],
                        )
                    ],
                ),
            ],
        )

        widgets.jslink(
            (etapes.children[0], "v_model"), (etapes.children[1].children[0], "v_model")
        )

        partie_data = widgets.VBox(
            [
                barre_menu,
                dialogue_save,
                boutons,
                figures,
                etapes,
                partie_magique,
            ],
            layout=Layout(width="100%"),
        )

        display(partie_data)
        #return partie_data

    def results(self, num_reg: int = None, item: str = None):
        """
        This function returns the results.

        Parameters
        ----------
        num_reg : int, optional
            The number of the region you want to see the results. The default is None.
        item : str, optional
            The item you want to see the results. Can be the following :
                - "X" : the X values of the region
                - "y" : the y values of the region
                - "indices" : the indices of the region in the original dataset
                - "SHAP" : the SHAP values of the region
                - "model name" : the name of the sub-model used for the region
                - "model" : the sub-model used for the region
                - "model score" : the score of the sub-model used for the region
                - "rules" : the rules defining the region

        Returns
        -------
        dictio : dict
            The results.
        """

        L_f = []
        if len(self.__list_of_regions) == 0:
            return "No region has been created !"
        for i in range(len(self.__list_of_regions)):
            dictio = dict()
            dictio["X"] = self.atk.dataset.X.iloc[self.__list_of_regions[i], :].reset_index(
                drop=True
            )
            dictio["y"] = y.iloc[self.__list_of_regions[i]].reset_index(drop=True)
            dictio["indices"] = self.__list_of_regions[i]
            dictio["SHAP"] = self.atk.dataset.explain["SHAP"].iloc[self.__list_of_regions[i], :].reset_index(
                drop=True
            )
            if self.__list_of_sub_models[i][-1] == -1:
                dictio["model name"] = None
                dictio["model score"] = None
                dictio["model"] = None
            else:
                dictio["model name"] = self.__list_of_sub_models[i][0]
                dictio["model score"] = self.__list_of_sub_models[i][1]
                dictio["model"] = self.sub_models[self.__list_of_sub_models[i][2]]
            dictio["rules"] = self.__all_tiles_rules[i]
            L_f.append(dictio)
        if num_reg == None or item == None:
            return L_f
        else:
            return L_f[num_reg][item]
