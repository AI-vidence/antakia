#Importations

# Imports Python
import os
import sys
import time
from copy import deepcopy
import json
import webbrowser
import threading

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
import umap
import pacmap
import shap
import lime
import lime.lime_tabular
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

from antakia._compute import red_PCA
from antakia._compute import red_TSNE
from antakia._compute import red_UMAP
from antakia._compute import red_PACMAP


class Gui():
    """
    Gui object.

    Attributes
    -------
    xplainer : Xplainer object
        The Xplainer object containing the data to explain.
    explanation_values : pandas dataframe
        The dataframe containing the explanations of the model. Can be given by the user (in explanation) or computed by the Gui.
    selection : list
        The list of the indices of the data currently selected by the user (lasso, skope-rules, or else).
    """
    def __init__(
        self,
        xplainer,
        explanation: str | pd.DataFrame = "SHAP",
        X_all: pd.DataFrame = None,
        default_projection: str = "PaCMAP",
        sub_models: list = None,
        save_regions: list = None,
    ):
        """Function that creates the interface.

        Parameters
        ----------
        explanation : str or pandas dataframe
            The type of explanation to display.
            If not computed : string ("SHAP" or "LIME", default : "SHAP")
            If already computed : pandas dataframe containing the explanations
            TODO : les calculs de SHAP ou LIME sont lancés depuis AntakIA en local ou envoyés à un serveur (avec GPU) distant. Je suggère que les fonctions de calcul long implémentent l'interface LongTask avec les métodes (start, update etc.)
        X_all : pandas dataframe
            The dataframe containing the entire data. It is used to compute the explanations if they are not already computed.
        default_projection : str
            The default projection to display. It can be "PaCMAP", "PCA", "t-SNE" or "UMAP".
            TODO : ça mériterait une interface "Projection" où chaque implémentation fournit son nom via un getProjType par ex. Ces types pourraient être à choisir parmi une liste de constantes (PCA, TSNE, UMAP ...) définies dans l'interface
        sub_models : list
            The list of sub-models to choose from for each region created by the user.
            # TODO : qu'est-ce que ça fait là ? S'il s'agit du constructeur du GUI. EN outre, c'est plutôt à AntakIA à maintenir cette liste surrogates
        save_regions : list
            The list of regions to display.
            # TODO : ce n'est pas au GUI de maintenir cette liste, mais à AntakIA
        """
        self.xplainer = xplainer
        self.explanation = explanation
        self.explanation_values = None
        self.X_all = X_all
        self.default_projection = default_projection
        self.sub_models = sub_models
        self.save_regions = save_regions

        # Publique :
        self.selection = []

        # Privé :
        self.__list_of_regions = []
        self.__list_of_sub_models = []
        self.__color_regions = []
        self.__columns_names = None
        self.__save_rules = None
        self.__other_columns = None
        self.__valider_bool = False
        self.__SHAP_train = None # class Dataset ? Intêret ? À discuter !
        a = [0] * 10
        self.__all_rules = [a, a, a, a, a, a, a, a, a, a]
        self.__X_not_scaled = None
        self.__Y_not_scaled = None 
        self.__SHAP_not_scaled = None
        self.__model_choice = None
        self.__Y_auto = None
        self.__all_tiles_rules = []
        self.__result_dyadic_clustering = None
        self.__all_models = None
        self.__score_models = []

    def display(self):
        """Function that displays the interface.

        Returns
        -------
        An interface
        """

        xplainer = self.xplainer
        explanation = self.explanation
        explanation_values = self.explanation_values
        X_all = self.X_all
        default_projection = self.default_projection
        sub_models = self.sub_models
        save_regions = self.save_regions
        
        X = xplainer.X
        Y = xplainer.Y
        model = xplainer.model
        # TODO Mettre toutes ces fonctions dans un module compute.py (par ex). Ce ne sont que des implémentatiosn de la même interface DimensionReduc. Et leur ferai implémenter une 2ème interface, "LongTask"

    

        # TODO : Ces submodels ont vocation à être nombreux. On pourrait créer un module surrogates.py, avec une classe abstraite Surrogate (ou bien une classe abstraite Model de Scikitlearn ?) et plusieurs implémentation. Il doit être facile à un développeur d'en importer d'autres
        # list of sub_models used
        # we will used these models only if the user do not give a list of sub_models
        self.__all_models = [
            linear_model.LinearRegression(),
            RandomForestRegressor(random_state=9),
            ensemble.GradientBoostingRegressor(random_state=9),
        ]

        if type(explanation) == str:
            calculus = True
        elif type(explanation) == type(pd.DataFrame()):
            calculus = False
            explanation_values = explanation
        else:
            raise ValueError(
                "The explanation parameter must be a string or a pandas dataframe.")

        # TODO et si on mettait dans la classe Potato, en plus des règles, une instance d'un modèle de substitution. Du coup le score serait calculé dans Potato
        def fonction_score(y, y_chap):
            # function that calculates the score of a machine-learning model
            y = np.array(y)
            y_chap = np.array(y_chap)
            return round(np.sqrt(sum((y - y_chap) ** 2) / len(y)), 3)

        # TODO GUI doit avoir un constructeur qui crée tous les widgets et leur définit un tooltip.
        def add_tooltip(widget, text):
            # function that allows you to add a tooltip to a widget
            wid = v.Tooltip(
                bottom=True,
                v_slots=[
                    {
                        "name": "activator",
                        "variable": "tooltip",
                        "children": widget,
                    }
                ],
                children=[text],
            )
            widget.v_on = "tooltip.on"
            return wid

        # we initialize the variables once: case where we launch antakia.start() twice in the same notebook!

        if sub_models != None:
            self.__all_models = sub_models
        liste_red = ["PCA", "t-SNE", "UMAP", "PaCMAP"]
        X = X.reset_index(drop=True)
        self.__list_of_sub_models = []

        self.__Y_not_scaled = Y.copy()

        self.__columns_names = X.columns.values[:3]

        def check_all(
            X, Y, explanation, explanation_values, model, X_all, default_projection
        ):
            # function that allows you to check that you have all the necessary information
            if sub_models != None and len(sub_models) > 9:
                return "Il faut renseigner moins de 10 modèles ! (changements à venir)"

            return True

        if (
            check_all(X, Y, explanation, explanation_values, model, X_all, default_projection)
            != True
        ):
            return check_all(
                X, Y, explanation, explanation_values, model, X_all, default_projection
            )

        if X_all is None:
            X_all = X.copy()

        if sub_models == None:
            sub_models = [
                linear_model.LinearRegression(),
                RandomForestRegressor(),
                ensemble.GradientBoostingRegressor(),
            ]

        def fonction_models(X, Y):
            # function that returns a list with the name/score/perf of the different models imported for a given X and Y
            models_liste = []
            for i in range(len(sub_models)):
                l = []
                sub_models[i].fit(X, Y)
                l.append(sub_models[i].__class__.__name__)
                l.append(str(round(sub_models[i].score(X, Y), 3)))
                l.append("MSE")
                l.append(sub_models[i].predict(X))
                l.append(sub_models[i])
                models_liste.append(l)
            return models_liste

        Y_pred = model.predict(X)
        columns_de_X = X.columns
        X_base = X.copy()
        # self.__X_not_scaled is the starting X, before standardization
        self.__X_not_scaled = X.copy()
        X = pd.DataFrame(StandardScaler().fit_transform(X))
        X.columns = [columns_de_X[i].replace(" ", "_") for i in range(len(X.columns))]
        X_base.columns = X.columns

        # wait screen definition
        # TODO et pourquopi pas une sous classe Splash Screen ? Que l'on traiterait comme un widget
        from importlib.resources import files

        data_path = files("antakia.assets").joinpath("logo_antakia.png")

        logo_antakia = widgets.Image(
            value=open(data_path, "rb").read(), layout=Layout(width="230px")
        )

        # TODO ces barres de progresssion doivent refléter le travail d'une LongTask qui s'éxécute dans un Thread dédié
        # waiting screen progress bars definition
        progress_shap = v.ProgressLinear(
            style_="width: 80%",
            class_="py-0 mx-5",
            v_model=0,
            color="primary",
            height="15",
            striped=True,
        )

        # EV dimension reduction progress bar
        progress_red = v.ProgressLinear(
            style_="width: 80%",
            class_="py-0 mx-5",
            v_model=0,
            color="primary",
            height="15",
            striped=True,
        )

        # consolidation of progress bars and progress texts in a single HBox
        prog_shap = v.Row(
            style_="width:85%;",
            children=[
                v.Col(
                    children=[
                        v.Html(
                            tag="h3",
                            class_="text-right",
                            children=["Calcul des valeurs explicatives"],
                        )
                    ]
                ),
                v.Col(children=[progress_shap]),
                v.Col(
                    children=[
                        v.Html(
                            tag="p",
                            class_="text-left font-weight-medium",
                            children=["0.00% [0/?] - 0m0s (temps estimé : /min /s)"],
                        )
                    ]
                ),
            ],
        )
        prog_red = v.Row(
            style_="width:85%;",
            children=[
                v.Col(
                    children=[
                        v.Html(
                            class_="text-right",
                            tag="h3",
                            children=["Calcul de réduction de dimension"],
                        )
                    ]
                ),
                v.Col(children=[progress_red]),
                v.Col(
                    children=[
                        v.Html(
                            tag="p",
                            class_="text-left font-weight-medium",
                            children=["..."],
                        )
                    ]
                ),
            ],
        )

        # definition of the splash screen which includes all the elements,
        splash = v.Layout(
            class_="d-flex flex-column align-center justify-center",
            children=[logo_antakia, prog_shap, prog_red],
        )

        # we send the splash screen
        display(splash)

        # if we import the explanatory values, the progress bar of this one is at 100
        if not calculus:
            progress_shap.v_model = 100
            prog_shap.children[2].children[
                0
            ].children = "Valeurs explicatives importées"

        # TODO Vivement que tout ce code avant et ci-dessous soit dans une sous-classe ad hoc! D'ailleurs, ce serait encore mieux de mettre ça dans un module SplashScreen.py
        def generation_texte(i, tot, time_init, progress):
            # allows to generate the progress text of the progress bar
            time_now = round((time.time() - time_init) / progress * 100, 1)
            minute = int(time_now / 60)
            seconde = time_now - minute * 60
            minute_passee = int((time.time() - time_init) / 60)
            seconde_passee = int((time.time() - time_init) - minute_passee * 60)
            return (
                str(round(progress_shap.v_model, 1))
                + "%"
                + " ["
                + str(i + 1)
                + "/"
                + str(tot)
                + "]"
                + " - "
                + str(minute_passee)
                + "m"
                + str(seconde_passee)
                + "s (temps estimé : "
                + str(minute)
                + "min "
                + str(round(seconde))
                + "s)"
            )

        # TODO A mettre dans AntakIA et à exécuter dans un Thread LongTask
        def get_SHAP(X, model):
            # calculates SHAP explanatory values
            time_init = time.time()
            explainer = shap.Explainer(model.predict, X_all)
            shap_values = pd.DataFrame().reindex_like(X)
            j = list(X.columns)
            for i in range(len(j)):
                j[i] = j[i] + "_shap"
            for i in range(len(X)):
                shap_value = explainer(X[i : i + 1], max_evals=1400)
                shap_values.iloc[i] = shap_value.values
                progress_shap.v_model += 100 / len(X)
                prog_shap.children[2].children[0].children = generation_texte(
                    i, len(X), time_init, progress_shap.v_model
                )
            shap_values.columns = j
            return shap_values

        # TODO id
        def get_LIME(X, model):
            # allows to calculate LIME explanatory values ​​(NOT WORKING YET)
            time_init = time.time()
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_all,
                feature_names=X.columns,
                class_names=["price"],
                verbose=True,
                mode="regression",
            )
            N = len(X)
            LIME = pd.DataFrame(np.zeros((N, 6)))
            l = []
            for j in range(N):
                l = []
                exp = explainer.explain_instance(
                    X.values[j], model.predict, num_features=6
                )
                for i in range(len(exp.as_list())):
                    l.append(exp.as_list()[i][1])
                progress_shap.v_model += 100 / N
                prog_shap.children[2].children[0].children = generation_texte(
                    j, N, time_init, progress_shap.v_model
                )
            return LIME

        print(calculus)

        if calculus == True:
            if explanation == "SHAP":
                SHAP = get_SHAP(self.__X_not_scaled, model)
            elif explanation == "LIME":
                SHAP = get_LIME(self.__X_not_scaled, model)
            else:
                SHAP = explanation_values
        else:
            SHAP = explanation_values

        self.explanation_values = SHAP.copy()

        self.__SHAP_not_scaled = SHAP.copy()

        choix_init_proj = 3

        # definition of the default projection
        # base, we take the PaCMAP projection
        # TODO : penser à traduire en EN
        # TODO : on ne pourrait pas (comme SHAP) choisir PACMAC par défaut ?
        if default_projection == "UMAP":
            prog_red.children[2].children[0].children = "Espace des valeurs... "
            choix_init_proj = 2
            Espace_valeurs = ["None", "None", red_UMAP(X, 2, True), "None"]
            Espace_valeurs_3D = ["None", "None", red_UMAP(X, 3, True), "None"]
            progress_red.v_model = +50
            prog_red.children[2].children[
                0
            ].children = "Espace des valeurs... Espace des explications..."
            Espace_explications = ["None", "None", red_UMAP(SHAP, 2, True), "None"]
            Espace_explications_3D = ["None", "None", red_UMAP(SHAP, 3, True), "None"]
            progress_red.v_model = +50

        elif default_projection == "t-SNE":
            choix_init_proj = 1
            prog_red.children[2].children[0].children = "Espace des valeurs... "
            Espace_valeurs = ["None", red_TSNE(X, 2, True), "None", "None"]
            Espace_valeurs_3D = ["None", red_TSNE(X, 3, True), "None", "None"]
            progress_red.v_model = +50
            prog_red.children[2].children[
                0
            ].children = "Espace des valeurs... Espace des explications..."
            Espace_explications = ["None", red_TSNE(SHAP, 2, True), "None", "None"]
            Espace_explications_3D = ["None", red_TSNE(SHAP, 3, True), "None", "None"]
            progress_red.v_model = +50

        elif default_projection == "PCA":
            choix_init_proj = 0
            prog_red.children[2].children[0].children = "Espace des valeurs... "
            Espace_valeurs = [red_PCA(X, 2, True), "None", "None", "None"]
            Espace_valeurs_3D = [red_PCA(X, 3, True), "None", "None", "None"]
            progress_red.v_model = +50
            prog_red.children[2].children[
                0
            ].children = "Espace des valeurs... Espace des explications..."
            Espace_explications = [red_PCA(SHAP, 2, True), "None", "None", "None"]
            Espace_explications_3D = [red_PCA(SHAP, 3, True), "None", "None", "None"]
            progress_red.v_model = +50

        else:
            prog_red.children[2].children[0].children = "Espace des valeurs... "
            choix_init_proj = 3
            Espace_valeurs = ["None", "None", "None", red_PACMAP(X, 2, True)]
            Espace_valeurs_3D = ["None", "None", "None", red_PACMAP(X, 3, True)]
            progress_red.v_model = +50
            prog_red.children[2].children[
                0
            ].children = "Espace des valeurs... Espace des explications..."
            Espace_explications = ["None", "None", "None", red_PACMAP(SHAP, 2, True)]
            Espace_explications_3D = ["None", "None", "None", red_PACMAP(SHAP, 3, True)]
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
            label="Projection dans l'EV :",
            items=["PCA", "t-SNE", "UMAP", "PaCMAP"],
        )

        EV_proj.v_model = EV_proj.items[choix_init_proj]

        # dropdown allowing to choose the projection in the space of the explanations
        EE_proj = v.Select(
            label="Projection dans l'EE :",
            items=["PCA", "t-SNE", "UMAP", "PaCMAP"],
        )

        EE_proj.v_model = EE_proj.items[choix_init_proj]

        # here the sliders of the parameters for the EV!
        slider_param_PaCMAP_voisins_EV = v.Layout(
            class_="mt-3",
            children=[
                v.Slider(
                    v_model=10, min=5, max=30, step=1, label="Nombre de voisins :"
                ),
                v.Html(class_="ml-3", tag="h3", children=["10"]),
            ],
        )

        slider_param_PaCMAP_mn_ratio_EV = v.Layout(
            class_="mt-3",
            children=[
                v.Slider(v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio :"),
                v.Html(class_="ml-3", tag="h3", children=["0.5"]),
            ],
        )

        slider_param_PaCMAP_fp_ratio_EV = v.Layout(
            class_="mt-3",
            children=[
                v.Slider(v_model=2, min=0.1, max=5, step=0.1, label="FP ratio :"),
                v.Html(class_="ml-3", tag="h3", children=["2"]),
            ],
        )

        def fonction_update_sliderEV(widget, event, data):
            # function that updates the values ​​when there is a change of sliders in the parameters of PaCMAP for the EV
            if widget.label == "Nombre de voisins :":
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
                "Valider",
            ]
        )

        reinit_params_proj_EV = v.Btn(
            class_="ml-4",
            children=[
                v.Icon(left=True, children=["mdi-skip-backward"]),
                "Réinitialiser",
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
                Espace_valeurs[3] = red_PACMAP(
                    X, 2, False, n_neighbors, MN_ratio, FP_ratio
                )
                Espace_valeurs_3D[3] = red_PACMAP(
                    X, 3, False, n_neighbors, MN_ratio, FP_ratio
                )
                out_loading1.layout.visibility = "hidden"
            with fig1.batch_update():
                fig1.data[0].x = Espace_valeurs[liste_red.index(EV_proj.v_model)][0]
                fig1.data[0].y = Espace_valeurs[liste_red.index(EV_proj.v_model)][1]
            with fig1_3D.batch_update():
                fig1_3D.data[0].x = Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    0
                ]
                fig1_3D.data[0].y = Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    1
                ]
                fig1_3D.data[0].z = Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    2
                ]

        valider_params_proj_EV.on_event("click", changement_params_EV)

        def reinit_param_EV(*b):
            # reset projection settings
            if liste_red.index(EV_proj.v_model) == 3:
                out_loading1.layout.visibility = "visible"
                Espace_valeurs[3] = red_PACMAP(X, 2, True)
                Espace_valeurs_3D[3] = red_PACMAP(X, 3, True)
                out_loading1.layout.visibility = "hidden"

            with fig1.batch_update():
                fig1.data[0].x = Espace_valeurs[liste_red.index(EV_proj.v_model)][0]
                fig1.data[0].y = Espace_valeurs[liste_red.index(EV_proj.v_model)][1]
            with fig1_3D.batch_update():
                fig1_3D.data[0].x = Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    0
                ]
                fig1_3D.data[0].y = Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    1
                ]
                fig1_3D.data[0].z = Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][
                    2
                ]

        reinit_params_proj_EV.on_event("click", reinit_param_EV)

        # here the sliders of the parameters for the EE!

        slider_param_PaCMAP_voisins_EE = v.Layout(
            class_="mt-3",
            children=[
                v.Slider(
                    v_model=10, min=5, max=30, step=1, label="Nombre de voisins :"
                ),
                v.Html(class_="ml-3", tag="h3", children=["10"]),
            ],
        )

        slider_param_PaCMAP_mn_ratio_EE = v.Layout(
            class_="mt-3",
            children=[
                v.Slider(v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio :"),
                v.Html(class_="ml-3", tag="h3", children=["0.5"]),
            ],
        )

        slider_param_PaCMAP_fp_ratio_EE = v.Layout(
            class_="mt-3",
            children=[
                v.Slider(v_model=2, min=0.1, max=5, step=0.1, label="FP ratio :"),
                v.Html(class_="ml-3", tag="h3", children=["2"]),
            ],
        )

        def fonction_update_sliderEE(widget, event, data):
            if widget.label == "Nombre de voisins :":
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
            n_neighbors = slider_param_PaCMAP_voisins_EE.children[0].v_model
            MN_ratio = slider_param_PaCMAP_mn_ratio_EE.children[0].v_model
            FP_ratio = slider_param_PaCMAP_fp_ratio_EE.children[0].v_model
            if liste_red.index(EV_proj.v_model) == 3:
                out_loading2.layout.visibility = "visible"
                Espace_explications[3] = red_PACMAP(
                    SHAP, 2, False, n_neighbors, MN_ratio, FP_ratio
                )
                Espace_explications_3D[3] = red_PACMAP(
                    SHAP, 3, False, n_neighbors, MN_ratio, FP_ratio
                )
                out_loading2.layout.visibility = "hidden"
            with fig2.batch_update():
                fig2.data[0].x = Espace_explications[liste_red.index(EE_proj.v_model)][
                    0
                ]
                fig2.data[0].y = Espace_explications[liste_red.index(EE_proj.v_model)][
                    1
                ]
            with fig2_3D.batch_update():
                fig2_3D.data[0].x = Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][0]
                fig2_3D.data[0].y = Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][1]
                fig2_3D.data[0].z = Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][2]

        valider_params_proj_EE.on_event("click", changement_params_EE)

        def reinit_param_EE(*b):
            if EE_proj.v_model == "PaCMAP":
                out_loading2.layout.visibility = "visible"
                Espace_explications[3] = red_PACMAP(SHAP, 2, True)
                Espace_explications_3D[3] = red_PACMAP(SHAP, 3, True)
                out_loading2.layout.visibility = "hidden"
            with fig2.batch_update():
                fig2.data[0].x = Espace_explications[liste_red.index(EE_proj.v_model)][
                    0
                ]
                fig2.data[0].y = Espace_explications[liste_red.index(EE_proj.v_model)][
                    1
                ]
            with fig2_3D.batch_update():
                fig2_3D.data[0].x = Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][0]
                fig2_3D.data[0].y = Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][1]
                fig2_3D.data[0].z = Espace_explications_3D[
                    liste_red.index(EE_proj.v_model)
                ][2]

        reinit_params_proj_EE.on_event("click", reinit_param_EE)

        # allows you to choose the color of the points
        # Y, Y hat, residuals, current selection, regions, unselected points, automatic clustering
        couleur_radio = v.BtnToggle(
            color="blue",
            mandatory=True,
            v_model="Y",
            children=[
                v.Btn(
                    icon=True,
                    children=[v.Icon(children=["mdi-alpha-y-circle-outline"])],
                    value="Y",
                    v_model=True,
                ),
                v.Btn(
                    icon=True,
                    children=[v.Icon(children=["mdi-alpha-y-circle"])],
                    value="Y^",
                    v_model=True,
                ),
                v.Btn(
                    icon=True,
                    children=[v.Icon(children=["mdi-minus-box-multiple"])],
                    value="Résidus",
                ),
                v.Btn(
                    icon=True,
                    children=[v.Icon(children="mdi-lasso")],
                    value="Selec actuelle",
                ),
                v.Btn(
                    icon=True,
                    children=[v.Icon(children=["mdi-ungroup"])],
                    value="Régions",
                ),
                v.Btn(
                    icon=True,
                    children=[v.Icon(children=["mdi-select-off"])],
                    value="Non selec",
                ),
                v.Btn(
                    icon=True,
                    children=[v.Icon(children=["mdi-star"])],
                    value="Clustering auto",
                ),
            ],
        )

        # added all tooltips!
        couleur_radio.children[0].children = [
            add_tooltip(couleur_radio.children[0].children[0], "Valeurs réelles")
        ]
        couleur_radio.children[1].children = [
            add_tooltip(couleur_radio.children[1].children[0], "Valeurs prédites")
        ]
        couleur_radio.children[2].children = [
            add_tooltip(couleur_radio.children[2].children[0], "Résidus")
        ]
        couleur_radio.children[3].children = [
            add_tooltip(
                couleur_radio.children[3].children[0],
                "Points de la séléction actuelle",
            )
        ]
        couleur_radio.children[4].children = [
            add_tooltip(couleur_radio.children[4].children[0], "Régions formées")
        ]
        couleur_radio.children[5].children = [
            add_tooltip(
                couleur_radio.children[5].children[0],
                "Points non sélectionnés",
            )
        ]
        couleur_radio.children[6].children = [
            add_tooltip(
                couleur_radio.children[6].children[0],
                "Clusters dyadique automatique",
            )
        ]

        def fonction_changement_couleur(*args, opacity: bool = True):
            # allows you to change the color of the points when you click on the buttons
            couleur = None
            scale = True
            a_modifier = True
            if couleur_radio.v_model == "Y":
                couleur = Y
            elif couleur_radio.v_model == "Y^":
                couleur = Y_pred
            elif couleur_radio.v_model == "Selec actuelle":
                scale = False
                couleur = ["grey"] * len(X_base)
                for i in range(len(self.selection)):
                    couleur[self.selection[i]] = "blue"
            elif couleur_radio.v_model == "Résidus":
                couleur = Y - Y_pred
                couleur = [abs(i) for i in couleur]
            elif couleur_radio.v_model == "Régions":
                scale = False
                couleur = [0] * len(X_base)
                for i in range(len(X_base)):
                    for j in range(len(self.__list_of_regions)):
                        if i in self.__list_of_regions[j]:
                            couleur[i] = j + 1
            elif couleur_radio.v_model == "Non selec":
                scale = False
                couleur = ["red"] * len(X_base)
                if len(self.__list_of_regions) > 0:
                    for i in range(len(X_base)):
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
        EVX = np.array(Espace_valeurs[choix_init_proj][0])
        EVY = np.array(Espace_valeurs[choix_init_proj][1])

        # marker 1 is the marker of figure 1
        marker1 = dict(
            color=Y,
            colorscale="Viridis",
            colorbar=dict(
                title="Y",
                thickness=20,
            ),
        )

        # marker 2 is the marker of figure 2 (without colorbar therefore)
        marker2 = dict(color=Y, colorscale="Viridis")

        fig_size = v.Slider(
            style_="width:20%",
            v_model=700,
            min=200,
            max=1200,
            label="Taille des graphiques (en px)",
        )

        fig_size_text = widgets.IntText(
            value="700", disabled=True, layout=Layout(width="40%")
        )

        widgets.jslink((fig_size, "v_model"), (fig_size_text, "value"))

        fig_size_et_texte = v.Row(children=[fig_size, fig_size_text])

        # the different menu buttons

        bouton_save = v.Btn(
            icon=True, children=[v.Icon(children=["mdi-content-save"])], elevation=0
        )
        bouton_save.children = [
            add_tooltip(bouton_save.children[0], "Gestion des sauvegardes")
        ]
        bouton_settings = v.Btn(
            icon=True, children=[v.Icon(children=["mdi-tune"])], elevation=0
        )
        bouton_settings.children = [
            add_tooltip(bouton_settings.children[0], "Paramètres de l'interface")
        ]
        bouton_website = v.Btn(
            icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
        )
        bouton_website.children = [
            add_tooltip(bouton_website.children[0], "Site web d'AI-Vidence")
        ]

        bouton_website.on_event(
            "click", lambda *args: webbrowser.open_new_tab("https://ai-vidence.com/")
        )

        data_path = files("antakia.assets").joinpath("logo_ai-vidence.png")
        with open(data_path, "rb") as f:
            logo = f.read()
        logo_aividence1 = widgets.Image(
            value=logo, height=str(864 / 20) + "px", width=str(3839 / 20) + "px"
        )
        logo_aividence1.layout.object_fit = "contain"
        logo_aividence = v.Layout(
            children=[logo_aividence1],
            class_="mt-1",
        )

        # menu bar (logo, links etc...)
        barre_menu = v.AppBar(
            elevation="4",
            class_="ma-4",
            rounded=True,
            children=[
                logo_aividence,
                v.Spacer(),
                bouton_save,
                bouton_settings,
                bouton_website,
            ],
        )

        # function to save the calculated explanatory values ​​(avoid redoing the calculations each time...)
        def save_shap(*args):
            def blockPrint():
                sys.stdout = open(os.devnull, "w")

            def enablePrint():
                sys.stdout = sys.__stdout__

            blockPrint()
            SHAP.to_csv("data_save/explanation_values.csv")
            enablePrint()

        bouton_save_shap = v.Btn(
            children=[
                v.Icon(children=["mdi-content-save"]),
                "Sauvegarder les valeurs explicatives calculées",
            ],
            elevation=4,
            class_="ma-4",
        )

        bouton_save_shap.on_event("click", save_shap)

        # function to open the dialog of the parameters of the size of the figures in particular (before finding better)
        dialogue_size = v.Dialog(
            children=[
                v.Card(
                    children=[
                        v.CardTitle(
                            children=[
                                v.Icon(class_="mr-5", children=["mdi-cogs"]),
                                "Paramètres",
                            ]
                        ),
                        v.CardText(children=[fig_size_et_texte]),
                        v.CardText(
                            children=[
                                v.Layout(
                                    class_="m d-flex justify-center",
                                    children=[
                                        # solara.FileDownload.widget(
                                        # data=dl_shap, filename="shap_values.csv"
                                        # )
                                        bouton_save_shap
                                    ],
                                )
                            ]
                        ),
                    ],
                    width="100%",
                )
            ],
            width="70%",
        )
        dialogue_size.v_model = False

        def ouvre_dialogue(*args):
            dialogue_size.v_model = True

        bouton_settings.on_event("click", ouvre_dialogue)

        if save_regions == None:
            save_regions = []

        len_init_regions = len(save_regions)

        # for the part on backups
        def init_save(new: bool = False):
            texte_regions = "Il n'y a pas de sauvegarde importée"
            for i in range(len(save_regions)):
                if len(save_regions[i]["liste"]) != len(X_all):
                    raise Exception("Votre sauvegarde n'est pas de la bonne taille !")
                    save_regions.pop(i)
            if len(save_regions) > 0:
                texte_regions = str(len(save_regions)) + " sauvegarde importée(s)"
            table_save = []
            for i in range(len(save_regions)):
                sous_mod_bool = "Non"
                new_or_not = "Importée"
                if i > len_init_regions:
                    new_or_not = "Créée"
                if (
                    len(save_regions[i]["sub_models"])
                    == max(save_regions[i]["liste"]) + 1
                ):
                    sous_mod_bool = "Oui"
                table_save.append(
                    [
                        i + 1,
                        save_regions[i]["nom"],
                        new_or_not,
                        max(save_regions[i]["liste"]) + 1,
                        sous_mod_bool,
                    ]
                )
            table_save = pd.DataFrame(
                table_save,
                columns=[
                    "Sauvegarde #",
                    "Nom",
                    "Origine",
                    "Nombre de régions",
                    "Sous-modèles ?",
                ],
            )

            colonnes = [
                {"text": c, "sortable": True, "value": c} for c in table_save.columns
            ]

            table_save = v.DataTable(
                v_model=[],
                show_select=True,
                single_select=True,
                headers=colonnes,
                items=table_save.to_dict("records"),
                item_value="Sauvegarde #",
                item_key="Sauvegarde #",
            )
            return [table_save, texte_regions]

        # the table that contains the backups
        table_save = init_save()[0]

        # view a selected backup
        visu_save = v.Btn(
            class_="ma-4",
            children=[v.Icon(children=["mdi-eye"])],
        )
        visu_save.children = [
            add_tooltip(visu_save.children[0], "Visualiser la sauvegarde sélectionnée")
        ]
        delete_save = v.Btn(
            class_="ma-4",
            children=[
                v.Icon(children=["mdi-trash-can"]),
            ],
        )
        delete_save.children = [
            add_tooltip(delete_save.children[0], "Supprimer la sauvegarde sélectionnée")
        ]
        new_save = v.Btn(
            class_="ma-4",
            children=[
                v.Icon(children=["mdi-plus"]),
            ],
        )
        new_save.children = [
            add_tooltip(new_save.children[0], "Créer une nouvelle sauvegarde")
        ]

        nom_sauvegarde = v.TextField(
            label="Nom de la sauvegarde",
            v_model="Default name",
            class_="ma-4",
        )

        # save a backup
        def delete_save_fonction(*args):
            if table_save.v_model == []:
                return
            table_save = carte_save.children[1]
            indice = table_save.v_model[0]["Sauvegarde #"] - 1
            save_regions.pop(indice)
            table_save = init_save(True)
            carte_save.children = [table_save[1], table_save[0]] + carte_save.children[
                2:
            ]

        delete_save.on_event("click", delete_save_fonction)

        # to view a backup
        def fonction_visu_save(*args):
            table_save = carte_save.children[1]
            if len(table_save.v_model) == 0:
                return
            indice = table_save.v_model[0]["Sauvegarde #"] - 1
            n = []
            for i in range(int(max(save_regions[indice]["liste"])) + 1):
                temp = []
                for j in range(len(save_regions[indice]["liste"])):
                    if save_regions[indice]["liste"][j] == i:
                        temp.append(j)
                if len(temp) > 0:
                    n.append(temp)
            self.__list_of_regions = n
            couleur = deepcopy(save_regions[indice]["liste"])
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
            if len(save_regions[indice]["sub_models"]) != len(self.__list_of_regions):
                self.__list_of_sub_models = [[None, None, None]] * len(self.__list_of_regions)
            else:
                self.__list_of_sub_models = []
                for i in range(len(self.__list_of_regions)):
                    nom = save_regions[indice]["sub_models"][i].__class__.__name__
                    indices_respectent = self.__list_of_regions[i]
                    score_init = fonction_score(
                        Y.iloc[indices_respectent], Y_pred[indices_respectent]
                    )
                    save_regions[indice]["sub_models"][i].fit(
                        X.iloc[indices_respectent], Y.iloc[indices_respectent]
                    )
                    score_reg = fonction_score(
                        Y.iloc[indices_respectent],
                        save_regions[indice]["sub_models"][i].predict(
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
                    l_m.append(sub_models[self.__list_of_sub_models[i][-1]])
            save = create_save(self.__color_regions, nom_sauvegarde.v_model, l_m)
            save_regions.append(save)
            table_save = init_save(True)
            carte_save.children = [table_save[1], table_save[0]] + carte_save.children[
                2:
            ]

        new_save.on_event("click", fonction_new_save)

        # save backups locally
        bouton_save_all = v.Btn(
            class_="ma-0",
            children=[
                v.Icon(children=["mdi-content-save"], class_="mr-2"),
                "Sauvegarder",
            ],
        )

        out_save = v.Alert(
            class_="ma-4 white--text",
            children=[
                "Sauvegarde effectuée avec succès",
            ],
            v_model=False,
        )

        # save local backups!
        def fonction_save_all(*args):
            emplacement = partie_local_save.children[1].children[1].v_model
            fichier = partie_local_save.children[1].children[2].v_model
            if len(emplacement) == 0 or len(fichier) == 0:
                out_save.color = "error"
                out_save.children = ["Veuillez remplir tous les champs"]
                out_save.v_model = True
                time.sleep(3)
                out_save.v_model = False
                return
            destination = emplacement + "/" + fichier + ".json"
            destination = destination.replace("//", "/")
            destination = destination.replace(" ", "_")

            for i in range(len(save_regions)):
                save_regions[i]["liste"] = list(save_regions[i]["liste"])
                save_regions[i]["sub_models"] = save_regions[i][
                    "sub_models"
                ].__class__.__name__
            with open(destination, "w") as fp:
                json.dump(save_regions, fp)
            out_save.color = "success"
            out_save.children = [
                "Enregsitrement effectué avec à cet emplacement : ",
                destination,
            ]
            out_save.v_model = True
            time.sleep(3)
            out_save.v_model = False
            return

        bouton_save_all.on_event("click", fonction_save_all)

        # part to save locally: choice of name, location, etc...
        partie_local_save = v.Col(
            class_="text-center d-flex flex-column align-center justify-center",
            children=[
                v.Html(tag="h3", children=["Sauvegarder en local_path"]),
                v.Row(
                    children=[
                        v.Spacer(),
                        v.TextField(
                            prepend_icon="mdi-folder-search",
                            class_="w-40 ma-3 mr-0",
                            elevation=3,
                            variant="outlined",
                            style_="width: 30%;",
                            v_model="data_save",
                            label="Emplacement",
                        ),
                        v.TextField(
                            prepend_icon="mdi-slash-forward",
                            class_="w-50 ma-3 mr-0 ml-0",
                            elevation=3,
                            variant="outlined",
                            style_="width: 50%;",
                            v_model="ma_sauvegarde",
                            label="Nom du fichier",
                        ),
                        v.Html(class_="mt-7 ml-0 pl-0", tag="p", children=[".json"]),
                        v.Spacer(),
                    ]
                ),
                bouton_save_all,
                out_save,
            ],
        )

        # card to manage backups, which opens
        carte_save = v.Card(
            elevation=0,
            children=[
                v.Html(tag="h3", children=[init_save()[1]]),
                table_save,
                v.Row(
                    children=[
                        v.Spacer(),
                        visu_save,
                        delete_save,
                        v.Spacer(),
                        new_save,
                        nom_sauvegarde,
                        v.Spacer(),
                    ]
                ),
                v.Divider(class_="mt mb-5"),
                partie_local_save,
            ],
        )

        dialogue_save = v.Dialog(
            children=[
                v.Card(
                    children=[
                        v.CardTitle(
                            children=[
                                v.Icon(class_="mr-5", children=["mdi-content-save"]),
                                "Gestion des sauvegardes",
                            ]
                        ),
                        v.CardText(children=[carte_save]),
                    ],
                    width="100%",
                )
            ],
            width="50%",
        )
        dialogue_save.v_model = False

        def ouvre_save(*args):
            dialogue_save.v_model = True

        bouton_save.on_event("click", ouvre_save)

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
        EEX = np.array(Espace_explications[choix_init_proj][0])
        EEY = np.array(Espace_explications[choix_init_proj][1])

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
            dimension_projection_text, "Dimension des projections"
        )

        def fonction_dimension_projection(*args):
            if dimension_projection.v_model:
                fig_2D_ou_3D.children = [fig1_3D_et_texte, fig2_3D_et_texte]
            else:
                fig_2D_ou_3D.children = [fig1_et_texte, fig2_et_texte]

        dimension_projection.on_event("change", fonction_dimension_projection)

        # coordinates of points in the 3D value space
        EVX_3D = np.array(Espace_valeurs_3D[choix_init_proj][0])
        EVY_3D = np.array(Espace_valeurs_3D[choix_init_proj][1])
        EVZ_3D = np.array(Espace_valeurs_3D[choix_init_proj][2])

        # coordinates of the points of the space of the explanations in 3D
        EEX_3D = np.array(Espace_explications_3D[choix_init_proj][0])
        EEY_3D = np.array(Espace_explications_3D[choix_init_proj][1])
        EEZ_3D = np.array(Espace_explications_3D[choix_init_proj][2])

        # marker 3D is the marker of figure 1 in 3D
        marker_3D = dict(
            color=Y,
            colorscale="Viridis",
            colorbar=dict(
                thickness=20,
            ),
            size=3,
        )

        # marker 3D_2 is the marker of figure 2 in 3D (without the colorbar therefore!)
        marker_3D_2 = dict(color=Y, colorscale="Viridis", size=3)

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
        texteEV = widgets.HTML("<h3>Espace des Valeurs<h3>")
        texteEE = widgets.HTML("<h3>Espace des Explications<h3>")

        # we display the figures and the text above!
        fig1_et_texte = widgets.VBox(
            [texteEV, fig1],
            layout=Layout(
                display="flex", align_items="center", margin="0px 0px 0px 0px"
            ),
        )
        fig2_et_texte = widgets.VBox(
            [texteEE, fig2],
            layout=Layout(
                display="flex", align_items="center", margin="0px 0px 0px 0px"
            ),
        )
        fig1_3D_et_texte = widgets.VBox(
            [texteEV, fig1_3D],
            layout=Layout(
                display="flex", align_items="center", margin="0px 0px 0px 0px"
            ),
        )
        fig2_3D_et_texte = widgets.VBox(
            [texteEE, fig2_3D],
            layout=Layout(
                display="flex", align_items="center", margin="0px 0px 0px 0px"
            ),
        )

        # HBox which allows you to choose between 2D and 3D figures by changing its children parameter!
        fig_2D_ou_3D = widgets.HBox([fig1_et_texte, fig2_et_texte])

        # allows to update graphs 1 & 2 according to the chosen projection
        def update_scatter(*args):
            val_act_EV = deepcopy(liste_red.index(EV_proj.v_model))
            val_act_EE = deepcopy(liste_red.index(EE_proj.v_model))
            param_EV.v_slots[0]["children"].disabled = True
            param_EE.v_slots[0]["children"].disabled = True
            if str(Espace_valeurs[liste_red.index(EV_proj.v_model)]) == "None":
                out_loading1.layout.visibility = "visible"
                if liste_red.index(EV_proj.v_model) == 0:
                    Espace_valeurs[liste_red.index(EV_proj.v_model)] = red_PCA(
                        X, 2, True
                    )
                    Espace_valeurs_3D[liste_red.index(EV_proj.v_model)] = red_PCA(
                        X, 3, True
                    )
                elif liste_red.index(EV_proj.v_model) == 1:
                    Espace_valeurs[liste_red.index(EV_proj.v_model)] = red_TSNE(
                        X, 2, True
                    )
                    Espace_valeurs_3D[liste_red.index(EV_proj.v_model)] = red_TSNE(
                        X, 3, True
                    )
                elif liste_red.index(EV_proj.v_model) == 2:
                    Espace_valeurs[liste_red.index(EV_proj.v_model)] = red_UMAP(
                        X, 2, True
                    )
                    Espace_valeurs_3D[liste_red.index(EV_proj.v_model)] = red_UMAP(
                        X, 3, True
                    )
                elif liste_red.index(EV_proj.v_model) == 3:
                    Espace_valeurs[liste_red.index(EV_proj.v_model)] = red_PACMAP(
                        X, 2, True
                    )
                    Espace_valeurs_3D[liste_red.index(EV_proj.v_model)] = red_PACMAP(
                        X, 3, True
                    )
                out_loading1.layout.visibility = "hidden"
            if str(Espace_explications[liste_red.index(EE_proj.v_model)]) == "None":
                out_loading2.layout.visibility = "visible"
                if liste_red.index(EE_proj.v_model) == 0:
                    Espace_explications[liste_red.index(EE_proj.v_model)] = red_PCA(
                        SHAP, 2, True
                    )
                    Espace_explications_3D[liste_red.index(EE_proj.v_model)] = red_PCA(
                        SHAP, 3, True
                    )
                elif liste_red.index(EE_proj.v_model) == 1:
                    Espace_explications[liste_red.index(EE_proj.v_model)] = red_TSNE(
                        SHAP, 2, True
                    )
                    Espace_explications_3D[liste_red.index(EE_proj.v_model)] = red_TSNE(
                        SHAP, 3, True
                    )
                elif liste_red.index(EE_proj.v_model) == 2:
                    Espace_explications[liste_red.index(EE_proj.v_model)] = red_UMAP(
                        SHAP, 2, True
                    )
                    Espace_explications_3D[liste_red.index(EE_proj.v_model)] = red_UMAP(
                        SHAP, 3, True
                    )
                elif liste_red.index(EE_proj.v_model) == 3:
                    Espace_explications[liste_red.index(EE_proj.v_model)] = red_PACMAP(
                        SHAP, 2, True
                    )
                    Espace_explications_3D[
                        liste_red.index(EE_proj.v_model)
                    ] = red_PACMAP(SHAP, 3, True)
                out_loading2.layout.visibility = "hidden"
            if liste_red.index(EE_proj.v_model) == 3:
                param_EE.v_slots[0]["children"].disabled = False
            if liste_red.index(EV_proj.v_model) == 3:
                param_EV.v_slots[0]["children"].disabled = False
            with fig1.batch_update():
                fig1.data[0].x = np.array(
                    Espace_valeurs[liste_red.index(EV_proj.v_model)][0]
                )
                fig1.data[0].y = np.array(
                    Espace_valeurs[liste_red.index(EV_proj.v_model)][1]
                )
            with fig2.batch_update():
                fig2.data[0].x = np.array(
                    Espace_explications[liste_red.index(EE_proj.v_model)][0]
                )
                fig2.data[0].y = np.array(
                    Espace_explications[liste_red.index(EE_proj.v_model)][1]
                )
            with fig1_3D.batch_update():
                fig1_3D.data[0].x = np.array(
                    Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][0]
                )
                fig1_3D.data[0].y = np.array(
                    Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][1]
                )
                fig1_3D.data[0].z = np.array(
                    Espace_valeurs_3D[liste_red.index(EV_proj.v_model)][2]
                )
            with fig2_3D.batch_update():
                fig2_3D.data[0].x = np.array(
                    Espace_explications_3D[liste_red.index(EE_proj.v_model)][0]
                )
                fig2_3D.data[0].y = np.array(
                    Espace_explications_3D[liste_red.index(EE_proj.v_model)][1]
                )
                fig2_3D.data[0].z = np.array(
                    Espace_explications_3D[liste_red.index(EE_proj.v_model)][2]
                )

            EV_proj.v_model = liste_red[val_act_EV]
            EE_proj.v_model = liste_red[val_act_EE]

        # we observe the changes in the values ​​of the dropdowns to change the method of reduction
        EV_proj.on_event("change", update_scatter)
        EE_proj.on_event("change", update_scatter)

        self.__color_regions = [0] * len(X_base)

        # definition of the table that will show the different results of the regions, with a stat of info about them
        table_regions = widgets.Output()

        # definition of the text that will give information on the selection
        texte_base = "Information sur la sélection : \n"
        texte_base_debut = (
            "Information sur la sélection : \n0 point sélectionné (0% de l'ensemble)"
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
                                "0 point sélectionné : utilisez le lasso sur les graphiques ci-dessus ou utilisez l'outil de sélection automatique"
                            ],
                        ),
                    ]
                ),
            ],
        )

        # button that applies skope-rules to the selection

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
                "Revenir aux règles initiales",
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
                                "En attente du lancement du skope-rules...",
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
                        v.CardTitle(children=["Résultat du skope sur l'EV :"]),
                        v.Spacer(),
                        v.Html(
                            class_="mr-5 mt-5 font-italic",
                            tag="p",
                            children=["précision = /"],
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
                                "En attente du lancement du skope-rules...",
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
                        v.CardTitle(children=["Résultat du skope sur l'EE :"]),
                        v.Spacer(),
                        v.Html(
                            class_="mr-5 mt-5 font-italic",
                            tag="p",
                            children=["précision = /"],
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

        # texts that will contain the information on the sub_models
        liste_mods = []
        for i in range(len(sub_models)):
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
                                        children=[sub_models[i].__class__.__name__]
                                    ),
                                ],
                            ),
                            v.CardText(
                                class_="mt-0 pt-0",
                                children=["Performances du modèle"],
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
                "Valider la sélection",
            ],
        )
        # button to delete all regions
        supprimer_toutes_les_tuiles = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-3", children=["mdi-trash-can-outline"]),
                "Supprimer les régions sélectionnées",
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
            v_model=False, label="Temps réel sur le graph.", class_="ma-3"
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
            v_model=False, label="Temps réel sur le graph.", class_="ma-3"
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
            v_model=False, label="Temps réel sur le graph.", class_="ma-3"
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
                "Valider la modification",
            ],
        )
        valider_change_2 = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-2", children=["mdi-check"]),
                "Valider la modification",
            ],
        )
        valider_change_3 = v.Btn(
            class_="ma-3",
            children=[
                v.Icon(class_="mr-2", children=["mdi-check"]),
                "Valider la modification",
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
            y_histo_shap = [0] * len(SHAP)
            nombre_div = 60
            garde_indice = []
            garde_valeur_y = []
            for i in range(nombre_div):
                garde_indice.append([])
                garde_valeur_y.append([])
            liste_scale = np.linspace(
                min(SHAP[nom_colonne_shap]), max(SHAP[nom_colonne_shap]), nombre_div + 1
            )
            for i in range(len(SHAP)):
                for j in range(nombre_div):
                    if (
                        SHAP[nom_colonne_shap][i] >= liste_scale[j]
                        and SHAP[nom_colonne_shap][i] <= liste_scale[j + 1]
                    ):
                        garde_indice[j].append(i)
                        garde_valeur_y[j].append(Y[i])
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
                color=X_base[nom_colonne],
                colorscale="Bluered_r",
                colorbar=dict(thickness=20, title=nom_colonne),
            )
            return [y_histo_shap, marker_shap]

        # definitions of the different color choices for the swarm

        choix_couleur_essaim1 = v.Row(
            class_="pt-3 mt-0 ml-4",
            children=[
                "Valeur de Xi",
                v.Switch(
                    class_="ml-3 mr-2 mt-0 pt-0",
                    v_model=False,
                    label="",
                ),
                "Sélection actuelle",
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

        y_histo_shap = [0] * len(SHAP)
        nom_col_shap = str(X.columns[0]) + "_shap"
        essaim1 = go.FigureWidget(
            data=[go.Scatter(x=SHAP[nom_col_shap], y=y_histo_shap, mode="markers")]
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
                "Valeur de Xi",
                v.Switch(
                    class_="ml-3 mr-2 mt-0 pt-0",
                    v_model=False,
                    label="",
                ),
                "Sélection actuelle",
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
            data=[go.Scatter(x=SHAP[nom_col_shap], y=y_histo_shap, mode="markers")]
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
            data=[go.Scatter(x=SHAP[nom_col_shap], y=y_histo_shap, mode="markers")]
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
                "Valeur de Xi",
                v.Switch(
                    class_="ml-3 mr-2 mt-0 pt-0",
                    v_model=False,
                    label="",
                ),
                "Sélection actuelle",
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
            nouvelle_tuile = X_base[
                (X_base[self.__all_rules[0][2]] >= self.__all_rules[0][0])
                & (X_base[self.__all_rules[0][2]] <= self.__all_rules[0][4])
            ].index
            for i in range(1, len(self.__all_rules)):
                X_temp = X_base[
                    (X_base[self.__all_rules[i][2]] >= self.__all_rules[i][0])
                    & (X_base[self.__all_rules[i][2]] <= self.__all_rules[i][4])
                ].index
                nouvelle_tuile = [g for g in nouvelle_tuile if g in X_temp]
            y_shape_skope = []
            y_color_skope = []
            y_opa_skope = []
            self.selection = nouvelle_tuile
            for i in range(len(X_base)):
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
            new_list_tout = X_base.index[
                X_base[self.__all_rules[indice][2]].between(value_min, value_max)
            ].tolist()
            for i in range(len(self.__all_rules)):
                min = self.__all_rules[i][0]
                max = self.__all_rules[i][4]
                if i != indice:
                    new_list_temp = X_base.index[
                        X_base[self.__all_rules[i][2]].between(min, max)
                    ].tolist()
                    new_list_tout = [g for g in new_list_tout if g in new_list_temp]
            for i in range(len(self.__all_rules)):
                with all_histograms[i].batch_update():
                    all_histograms[i].data[2].x = X_base[self.__all_rules[i][2]][new_list_tout]
                if all_color_choosers_beeswarms[i].children[1].v_model:
                    with all_beeswarms[i].batch_update():
                        y_color = [0] * len(SHAP)
                        if i == indice:
                            indices = X_base.index[
                                X_base[self.__all_rules[i][2]].between(value_min, value_max)
                            ].tolist()
                        else:
                            indices = X_base.index[
                                X_base[self.__all_rules[i][2]].between(
                                    self.__all_rules[i][0], self.__all_rules[i][4]
                                )
                            ].tolist()
                        for j in range(len(SHAP)):
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
                for g in list(X_base[self.__columns_names[0]].values)
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
                for g in list(X_base[self.__columns_names[1]].values)
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
                for g in list(X_base[self.__columns_names[2]].values)
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
            self.__columns_names = []
            valeurs = []
            symbole = []
            for i in range(len(chaine_carac)):
                if "<" in chaine_carac[i] or ">" in chaine_carac[i]:
                    self.__columns_names.append(chaine_carac[i - 1])
                    symbole.append(chaine_carac[i])
                    if chaine_carac[i + 1][-1] == ",":
                        valeurs.append(float(chaine_carac[i + 1][:-2]))
                    else:
                        valeurs.append(float(chaine_carac[i + 1]))
            return [self.__columns_names, symbole, valeurs]

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
                        self.__all_rules[j][0] > X_base.iloc[i, colonne]
                        or X_base.iloc[i, colonne] > self.__all_rules[j][4]
                    ):
                        liste_bool[i] = False
            temp = [i for i in range(len(X)) if liste_bool[i]]
            return temp

        def fonction_scores_models(temp):
            if temp == None:
                temp = regles_to_indices()
            result_models = fonction_models(X.iloc[temp, :], Y.iloc[temp])
            score_tot = []
            for i in range(len(sub_models)):
                score_tot.append(fonction_score(Y.iloc[temp], result_models[i][-2]))
            score_init = fonction_score(Y.iloc[temp], Y_pred[temp])
            if score_init == 0:
                l_compar = ["/"] * len(sub_models)
            else:
                l_compar = [
                    round(100 * (score_init - score_tot[i]) / score_init, 1)
                    for i in range(len(sub_models))
                ]

            self.__score_models = []
            for i in range(len(sub_models)):
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
                        + " (contre "
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
                            + " (contre "
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
                            + " (contre "
                            + str(score_init)
                            + ", "
                            + str(
                                round(100 * (score_init - score_tot[i]) / score_init, 1)
                            )
                            + "%)"
                        )

            for i in range(len(sub_models)):
                mods.children[i].children[0].children[1].children = str_md(i)

        X_train = X_base.copy()

        # when you click on the skope-rules button
        def fonction_validation_skope(*sender):
            y_train = self.__y_train
            loading_models.class_ = "d-flex"
            self.__valider_bool = True
            if y_train == None:
                texte_skopeEV.children[1].children = [
                    widgets.HTML("Veuillez sélectionner des points")
                ]
                texte_skopeEE.children[1].children = [
                    widgets.HTML("Veuillez sélectionner des points")
                ]
            elif 0 not in y_train or 1 not in y_train:
                texte_skopeEV.children[1].children = [
                    widgets.HTML("Vous ne pouvez pas tout/rien choisir !")
                ]
                texte_skopeEE.children[1].children = [
                    widgets.HTML("Vous ne pouvez pas tout/rien choisir !")
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
                        widgets.HTML("Pas de règle trouvée")
                    ]
                    texte_skopeEE.children[1].children = [
                        widgets.HTML("Pas de règle trouvée")
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
                        + " ext. de l'arbre = "
                        + chaine_carac[1][2]
                    ]

                    # there we find the values ​​of the skope to use them for the sliders
                    self.__columns_names, symbole, valeurs = re_transform_string(
                        chaine_carac[0]
                    )
                    self.__other_columns = [
                        g for g in X_base.columns if g not in self.__columns_names
                    ]
                    widget_list_add_skope.items = self.__other_columns
                    widget_list_add_skope.v_model = self.__other_columns[0]
                    liste_val_histo = [0] * len(self.__columns_names)
                    liste_index = [0] * len(self.__columns_names)
                    le_top = []
                    le_min = []
                    self.__all_rules = []

                    def f_rond(a):
                        return np.round(a, 2)

                    for i in range(len(self.__columns_names)):
                        une_regle = [0] * 5
                        une_regle[2] = self.__columns_names[i]
                        if symbole[i] == "<":
                            une_regle[0] = f_rond(
                                float(min(list(X_base[self.__columns_names[i]].values)))
                            )
                            une_regle[1] = "<"
                            une_regle[3] = "<"
                            une_regle[4] = f_rond(float(valeurs[i]))
                            X1 = [
                                g
                                for g in list(X_base[self.__columns_names[i]].values)
                                if g < valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(X_base[self.__columns_names[i]].index.values)
                                if X_base[self.__columns_names[i]][h] < valeurs[i]
                            ]
                            le_top.append(valeurs[i])
                            le_min.append(min(list(X_base[self.__columns_names[i]].values)))
                        elif symbole[i] == ">":
                            une_regle[0] = f_rond(float(valeurs[i]))
                            une_regle[1] = "<"
                            une_regle[3] = "<"
                            une_regle[4] = f_rond(
                                float(max(list(X_base[self.__columns_names[i]].values)))
                            )
                            X1 = [
                                g
                                for g in list(X_base[self.__columns_names[i]].values)
                                if g > valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(X_base[self.__columns_names[i]].index.values)
                                if X_base[self.__columns_names[i]][h] > valeurs[i]
                            ]
                            le_min.append(valeurs[i])
                            le_top.append(max(list(X_base[self.__columns_names[i]].values)))
                        elif symbole[i] == "<=":
                            une_regle[0] = f_rond(
                                float(min(list(X_base[self.__columns_names[i]].values)))
                            )
                            une_regle[1] = "<="
                            une_regle[3] = "<="
                            une_regle[4] = f_rond(float(valeurs[i]))
                            le_top.append(valeurs[i])
                            le_min.append(min(list(X_base[self.__columns_names[i]].values)))
                            X1 = [
                                g
                                for g in list(X_base[self.__columns_names[i]].values)
                                if g <= valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(X_base[self.__columns_names[i]].index.values)
                                if X_base[self.__columns_names[i]][h] <= valeurs[i]
                            ]
                            liste_index[i] = X2
                        elif symbole[i] == ">=":
                            une_regle[0] = f_rond(float(valeurs[i]))
                            une_regle[1] = "<="
                            une_regle[3] = "<="
                            une_regle[4] = f_rond(
                                float(max(list(X_base[self.__columns_names[i]].values)))
                            )
                            le_min.append(valeurs[i])
                            le_top.append(max(list(X_base[self.__columns_names[i]].values)))
                            X1 = [
                                g
                                for g in list(X_base[self.__columns_names[i]].values)
                                if g >= valeurs[i]
                            ]
                            X2 = [
                                h
                                for h in list(X_base[self.__columns_names[i]].index.values)
                                if X_base[self.__columns_names[i]][h] >= valeurs[i]
                            ]
                        liste_index[i] = X2
                        liste_val_histo[i] = X1
                        self.__all_rules.append(une_regle)
                    self.__save_rules = deepcopy(self.__all_rules)

                    une_carte_EV.children = generate_card(
                        liste_to_string_skope(self.__all_rules)
                    )

                    [new_y, marker] = fonction_beeswarm_shap(self.__columns_names[0])
                    essaim1.data[0].y = new_y
                    essaim1.data[0].x = SHAP[self.__columns_names[0] + "_shap"]
                    essaim1.data[0].marker = marker

                    all_histograms = [histogram1]
                    if len(self.__columns_names) > 1:
                        all_histograms = [histogram1, histogram2]
                        [new_y, marker] = fonction_beeswarm_shap(self.__columns_names[1])
                        essaim2.data[0].y = new_y
                        essaim2.data[0].x = SHAP[self.__columns_names[1] + "_shap"]
                        essaim2.data[0].marker = marker

                    if len(self.__columns_names) > 2:
                        all_histograms = [histogram1, histogram2, histogram3]
                        [new_y, marker] = fonction_beeswarm_shap(self.__columns_names[2])
                        essaim3.data[0].y = new_y
                        essaim3.data[0].x = SHAP[self.__columns_names[2] + "_shap"]
                        essaim3.data[0].marker = marker

                    if len(self.__columns_names) == 1:
                        indices_respectent_skope = liste_index[0]
                    elif len(self.__columns_names) == 2:
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
                    for i in range(len(X_base)):
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

                    if len(self.__columns_names) == 2:
                        accordion_skope.children = [
                            dans_accordion1_n,
                            dans_accordion2_n,
                        ]
                        dans_accordion1_n.children[0].children[0].children = (
                            "X1 (" + self.__columns_names[0] + ")"
                        )
                        dans_accordion2_n.children[0].children[0].children = (
                            "X2 (" + self.__columns_names[1] + ")"
                        )
                    elif len(self.__columns_names) == 3:
                        accordion_skope.children = [
                            dans_accordion1_n,
                            dans_accordion2_n,
                            dans_accordion3_n,
                        ]
                        dans_accordion1_n.children[0].children[0].children = (
                            "X1 (" + self.__columns_names[0] + ")"
                        )
                        dans_accordion2_n.children[0].children[0].children = (
                            "X2 (" + self.__columns_names[1] + ")"
                        )
                        dans_accordion3_n.children[0].children[0].children = (
                            "X3 (" + self.__columns_names[2] + ")"
                        )
                    elif len(self.__columns_names) == 1:
                        accordion_skope.children = [dans_accordion1_n]
                        dans_accordion1_n.children[0].children[0].children = (
                            "X1 (" + self.__columns_names[0] + ")"
                        )

                    slider_skope1.min = -10e10
                    slider_skope1.max = 10e10
                    slider_skope2.min = -10e10
                    slider_skope2.max = 10e10
                    slider_skope3.min = -10e10
                    slider_skope3.max = 10e10

                    slider_skope1.max = (
                        round(max(list(X_base[self.__columns_names[0]].values)), 1)
                    ) * 100
                    slider_skope1.min = (
                        round(min(list(X_base[self.__columns_names[0]].values)), 1)
                    ) * 100
                    slider_skope1.v_model = [
                        round(le_min[0], 1) * 100,
                        round(le_top[0], 1) * 100,
                    ]

                    [
                        slider_text_comb1.children[0].v_model,
                        slider_text_comb1.children[2].v_model,
                    ] = [slider_skope1.v_model[0] / 100, slider_skope1.v_model[1] / 100]

                    if len(self.__columns_names) > 1:
                        slider_skope2.max = (
                            max(list(X_base[self.__columns_names[1]].values))
                        ) * 100
                        slider_skope2.min = (
                            min(list(X_base[self.__columns_names[1]].values))
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

                    if len(self.__columns_names) > 2:
                        slider_skope3.description = self.__columns_names[2]
                        slider_skope3.max = (
                            max(list(X_base[self.__columns_names[2]].values))
                        ) * 100
                        slider_skope3.min = (
                            min(list(X_base[self.__columns_names[2]].values))
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
                            X_base[re_transform_string(chaine_carac[0])[0][0]]
                        )
                        if len(histogram1.data) > 1:
                            histogram1.data[1].x = liste_val_histo[0]

                    if len(self.__columns_names) > 1:
                        with histogram2.batch_update():
                            histogram2.data[0].x = list(
                                X_base[re_transform_string(chaine_carac[0])[0][1]]
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

                    if len(self.__columns_names) > 2:
                        with histogram3.batch_update():
                            histogram3.data[0].x = list(
                                X_base[re_transform_string(chaine_carac[0])[0][2]]
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
                    texte_skopeEE.children[0].children[3].children = [
                        # str(skope_rules_clf.rules_[0])
                        # + "\n"
                        "p = "
                        + str(np.round(float(chaine_carac[1][0]) * 100, 5))
                        + "%"
                        + " r = "
                        + str(np.round(np.round(float(chaine_carac[1][1]), 2) * 100, 5))
                        + "%"
                        + " ext. de l'arbre = "
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
                    "Sélectionnez des points sur la figure pour voir leurs valeurs ici"
                )
            )
        out_selec_SHAP = widgets.Output()
        with out_selec_SHAP:
            display(
                HTML(
                    "Sélectionnez des points sur la figure pour voir leurs valeurs de Shapley ici"
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
                        v.ExpansionPanelHeader(children=["Données sélectionnées"]),
                        v.ExpansionPanelContent(children=[out_selec_2]),
                    ]
                )
            ],
        )

        trouve_clusters = v.Btn(
            class_="ma-1 mt-2 mb-0",
            elevation="2",
            children=[v.Icon(children=["mdi-magnify"]), "Trouver des clusters"],
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
            children=["Nombre de clusters : " + str(slider_clusters.v_model)],
        )

        def fonct_texte_clusters(*b):
            texte_slider_cluster.children = [
                "Nombre de clusters : " + str(slider_clusters.v_model)
            ]

        slider_clusters.on_event("input", fonct_texte_clusters)

        check_nb_clusters = v.Checkbox(
            v_model=True, label="Nombre optimal de cluster", class_="ma-3"
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

        new_df = pd.DataFrame([], columns=["Régions #", "Nombre de points"])
        colonnes = [{"text": c, "sortable": True, "value": c} for c in new_df.columns]
        resultats_clusters_table = v.DataTable(
            class_="w-100",
            style_="width : 100%",
            v_model=[],
            show_select=False,
            headers=colonnes,
            items=new_df.to_dict("records"),
            item_value="Régions #",
            item_key="Régions #",
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
                result = fonction_auto_clustering(X, SHAP, 3, True)
            else:
                nb_clusters = slider_clusters.v_model
                result = fonction_auto_clustering(X, SHAP, nb_clusters, False)
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
                columns=["Régions #", "Nombre de points", "Pourcentage du total"],
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
                item_value="Régions #",
                item_key="Régions #",
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
                + " points sélectionnés ("
                + str(round(len(les_points) / len(X_base) * 100, 2))
                + "% de l'ensemble)"
            )
            texte_selec.value = (
                texte_base
                + str(len(les_points))
                + " points sélectionnés ("
                + str(round(len(les_points) / len(X_base) * 100, 2))
                + "% de l'ensemble)"
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

            X_train = X_base.copy()
            self.__SHAP_train = SHAP.copy()

            X_mean = (
                pd.DataFrame(
                    X_train.iloc[self.selection, :].mean(axis=0).values.reshape(1, -1),
                    columns=X_train.columns,
                )
                .round(2)
                .rename(index={0: "Moyenne sélection"})
            )
            X_mean_tot = (
                pd.DataFrame(
                    X_train.mean(axis=0).values.reshape(1, -1), columns=X_train.columns
                )
                .round(2)
                .rename(index={0: "Moyenne ensemble"})
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
                .rename(index={0: "Moyenne sélection"})
            )
            SHAP_mean_tot = (
                pd.DataFrame(
                    self.__SHAP_train.mean(axis=0).values.reshape(1, -1),
                    columns=self.__SHAP_train.columns,
                )
                .round(2)
                .rename(index={0: "Moyenne ensemble"})
            )
            SHAP_mean = pd.concat([SHAP_mean, SHAP_mean_tot], axis=0)

            with out_selec:
                clear_output()
                display(HTML("<h4> Espace des Valeurs </h4>"))
                display(HTML("<h5>Point moyen de la sélection :<h5>"))
                display(HTML(X_mean.to_html()))
                display(HTML("<h5>Ensemble des points de la sélection :<h5>"))
                display(HTML(X_train.iloc[self.selection, :].to_html(index=False)))
            with out_selec_SHAP:
                clear_output()
                display(HTML("<h4> Espace des Explications </h4>"))
                display(HTML("<h5>Point moyen de la sélection :<h5>"))
                display(HTML(SHAP_mean.to_html()))
                display(HTML("<h5>Ensemble des points de la sélection :<h5>"))
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
                    nom_model = sub_models[self.__model_choice].__class__.__name__
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
                nouvelle_tuile = X_base[
                    (X_base[self.__all_rules[0][2]] >= self.__all_rules[0][0])
                    & (X_base[self.__all_rules[0][2]] <= self.__all_rules[0][4])
                ].index
                for i in range(1, len(self.__all_rules)):
                    X_temp = X_base[
                        (X_base[self.__all_rules[i][2]] >= self.__all_rules[i][0])
                        & (X_base[self.__all_rules[i][2]] <= self.__all_rules[i][4])
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
                            np.round(len(self.__list_of_regions[i]) / len(X_base) * 100, 2),
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
                            np.round(len(self.__list_of_regions[i]) / len(X_base) * 100, 2),
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
                    np.round(toute_somme / len(X_base) * 100, 2),
                    "/",
                    score_tot,
                    score_tot_glob,
                    percent,
                ]
            )
            new_df = pd.DataFrame(
                temp,
                columns=[
                    "Régions #",
                    "Nombre de points",
                    "% de l'ensemble",
                    "Modèle",
                    "Score du modèle régional (MSE)",
                    "Score du modèle global (MSE)",
                    "Gain en MSE",
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
                    item_value="Régions #",
                    item_key="Régions #",
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
                        v.Html(tag="h2", children=["Bilan des régions :"]),
                        table_total,
                    ],
                )

                def fonction_suppression_tuiles(*b):
                    if table_donnes.v_model == []:
                        return
                    taille = len(table_donnes.v_model)
                    a = 0
                    for i in range(taille):
                        indice = table_donnes.v_model[i]["Régions #"] - 1
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

        choix_coul_metier = widgets.Dropdown(
            options=["Régions", "Y"],
            value="Y",
            description="Couleur des points :",
            disabled=False,
            style={"description_width": "initial"},
        )

        boutton_add_skope = v.Btn(
            class_="ma-4 pa-1 mb-0",
            children=[v.Icon(children=["mdi-plus"]), "Ajouter un paramètre sélectif"],
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
            nouvelle_regle[0] = round(min(list(X_base[colonne].values)), 1)
            nouvelle_regle[1] = "<="
            nouvelle_regle[3] = "<="
            nouvelle_regle[4] = round(max(list(X_base[colonne].values)), 1)
            self.__all_rules.append(nouvelle_regle)
            une_carte_EV.children = generate_card(liste_to_string_skope(self.__all_rules))

            new_valider_change = v.Btn(
                class_="ma-3",
                children=[
                    v.Icon(class_="mr-2", children=["mdi-check"]),
                    "Valider la modification",
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
                        x=X_base[colonne].values,
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
                    x=X_base[colonne].values,
                    bingroup=1,
                    nbinsx=nombre_bins,
                    marker_color="LightSkyBlue",
                    opacity=0.6,
                )
            )
            new_histogram.add_trace(
                go.Histogram(
                    x=X_base[colonne].values,
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
                v_model=False, label="Temps réel sur le graph.", class_="ma-3"
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
            y_histo_shap = [0] * len(SHAP)
            new_essaim = go.FigureWidget(
                data=[go.Scatter(x=SHAP[colonne_shap], y=y_histo_shap, mode="markers")]
            )
            new_essaim.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=200,
                width=0.9 * int(fig_size.v_model),
            )
            new_essaim.update_yaxes(visible=False, showticklabels=False)
            [new_y, marker] = fonction_beeswarm_shap(colonne)
            new_essaim.data[0].y = new_y
            new_essaim.data[0].x = SHAP[colonne_shap]
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
                    "Sélection actuelle",
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
                    for g in list(X_base[colonne].values)
                    if g >= new_slider_skope.v_model[0] / 100
                    and g <= new_slider_skope.v_model[1] / 100
                ]
                new_histogram.data[1].x = new_list

                colonne_2 = new_slider_skope.label
                new_list_regle = X_base.index[
                    X_base[colonne_2].between(
                        new_slider_skope.v_model[0] / 100,
                        new_slider_skope.v_model[1] / 100,
                    )
                ].tolist()
                new_list_tout = new_list_regle.copy()
                for i in range(1, len(self.__all_rules)):
                    new_list_temp = X_base.index[
                        X_base[self.__all_rules[i][2]].between(
                            self.__all_rules[i][0], self.__all_rules[i][4]
                        )
                    ].tolist()
                    new_list_tout = [g for g in new_list_tout if g in new_list_temp]
                new_list_tout_new = X_base[colonne_2][new_list_tout]
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
                    for g in list(X_base[colonne_2].values)
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
                "Paramètres de la projection dans l'EV",
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
                "Paramètres de la projection dans l'EE",
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
                "Réinitialiser l'opacité des points",
            )
        ]

        def fonction_reinit_opa(*args):
            with fig1.batch_update():
                fig1.data[0].marker.opacity = 1
            with fig2.batch_update():
                fig2.data[0].marker.opacity = 1

        bouton_reinit_opa.on_event("click", fonction_reinit_opa)

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
                            "Régler la couleur des points",
                        ),
                        couleur_radio,
                        bouton_reinit_opa,
                    ],
                ),
                v.Layout(children=[projEV_et_load, projEE_et_load]),
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
            label="Afficher les valeurs de Shapley sous forme d'essaim",
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
                "Bouton magique",
            ],
        )

        partie_magique = v.Layout(
            class_="d-flex flex-row justify-center align-center",
            children=[
                v.Spacer(),
                bouton_magique,
                v.Checkbox(v_model=True, label="Mode démonstration", class_="ma-4"),
                v.TextField(
                    class_="shrink",
                    type="number",
                    label="Temps entre les étapes (ds)",
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
                        v.Tab(value="one", children=["1. Sélection en cours"]),
                        v.Tab(value="two", children=["2. Ajustement de la sélection"]),
                        v.Tab(value="three", children=["3. Choix du sous-modèle"]),
                        v.Tab(value="four", children=["4. Bilan des régions"]),
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
                dialogue_size,
                boutons,
                figures,
                etapes,
                partie_magique,
            ],
            layout=Layout(width="100%"),
        )

        display(partie_data)
        #return partie_data

    def results(self, num_reg: int = None, chose: str = None):
        L_f = []
        if len(self.__list_of_regions) == 0:
            return "Aucune région de validée !"
        for i in range(len(self.__list_of_regions)):
            dictio = dict()
            dictio["X"] = self.__X_not_scaled.iloc[self.__list_of_regions[i], :].reset_index(
                drop=True
            )
            dictio["y"] = self.__Y_not_scaled.iloc[self.__list_of_regions[i]].reset_index(drop=True)
            dictio["indices"] = self.__list_of_regions[i]
            dictio["SHAP"] = self.__SHAP_not_scaled.iloc[self.__list_of_regions[i], :].reset_index(
                drop=True
            )
            if self.__list_of_sub_models[i][-1] == -1:
                dictio["model name"] = None
                dictio["model score"] = None
                dictio["model"] = None
            else:
                dictio["model name"] = self.__list_of_sub_models[i][0]
                dictio["model score"] = self.__list_of_sub_models[i][1]
                dictio["model"] = self.__all_models[self.__list_of_sub_models[i][2]]
            dictio["rules"] = self.__all_tiles_rules[i]
            L_f.append(dictio)
        if num_reg == None or chose == None:
            return L_f
        else:
            return L_f[num_reg][chose]
