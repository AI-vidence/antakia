import pandas as pd
import numpy as np

# TODO : these references to IPython should be removed in favor of a new scheme (see Wiki)
import ipyvuetify as v
import ipywidgets as widgets
from IPython.display import display 

from sklearn.preprocessing import StandardScaler

from antakia import gui_elements

import time

from copy import deepcopy

class Dataset():
    """Dataset object.
    This object contains the all data, the model to explain, the explanations and the predictions.

    Attributes
    -------
    X : pandas dataframe
        The dataframe containing the dataset (might not be the entire dataset, see `frac` method)
    X_all : pandas dataframe
        The dataframe containing the entire dataset, in order for the explanations to be computed.
    X_scaled : pandas dataframe
        The dataframe containing the scaled dataset.
    y : pandas series
        The series containing the target values.
    model : model object
        The "black-box" model to explain.
    y_pred : pandas series
        The series containing the predictions of the model. If None, the predictions are computed using the model and the data.
    comments : list of str
        The comments associated to each feature.
    sensible : list of bool
        The list of boolean indicating if the feature is sensible or not. If True, a warning will be displayed when the feature is used in the explanations. More to come in the future.
    lat : str
        The name of the latitude column.
    long : str
        The name of the longitude column.
    """

    def __init__(self, X:pd.DataFrame = None, csv:str = None, y:pd.Series = None, model = None):
        """
        Constructor of the class Dataset.
        
        Parameters
        ---------
        X : pandas dataframe
            The dataframe containing the dataset.
        csv : str
            The path to the csv file containing the dataset.
        y : pandas series
            The series containing the target values.
        model : model object
            The "black-box" model to explain. The model must have a predict method.
        """

        X.columns = [X.columns[i].replace(" ", "_") for i in range(len(X.columns))]
        X = X.reset_index(drop=True)

        if X is None and csv is None :
            raise ValueError("You must provide a dataframe or a csv file")
        if X is not None and csv is not None :
            raise ValueError("You must provide either a dataframe or a csv file, not both")
        if X is not None :
            self.X = X
        else :
            self.X = pd.read_csv(csv)
        self.X_all = X
        self.model = model
        self.y = y
        self.X_scaled = pd.DataFrame(StandardScaler().fit_transform(X))
        self.X_scaled.columns = X.columns

        self.y_pred = pd.Series(self.model.predict(self.X))

        self.verbose = None
        self.widget = None

        self.comments = [""]*len(self.X.columns)
        self.sensible = [False]*len(self.X.columns)

        self.fraction = 1
        self.frac_indexes = self.X.index

        self.long, self.lat = None, None

        for name in ['longitude', 'Longitude', 'Long', 'long']:
            if name in self.X.columns:
                self.long = name
        for name in ['latitude', 'Latitude', 'Lat', 'lat']:
            if name in self.X.columns:
                self.lat = name

    def __str__(self):
        texte = ' '.join(("Dataset:\n",
                    "------------------\n",
                    "      Number of observations:", str(self.X.shape[0]), "\n",
                    "      Number of variables:", str(self.X.shape[1]), "\n",
                    "Explanations:\n",
                    "------------------\n",
                    "      Imported:", str(self.explain["Imported"] != None), "\n",
                    "      SHAP:", str(self.explain["SHAP"] != None), "\n",
                    "      LIME:", str(self.explain["LIME"] != None)))
        return texte
    
    def __len__(self):
        return self.X.shape[0]
    
    def frac(self, p:float):
        """
        Reduces the dataset to a fraction of its size.

        Parameters
        ---------
        p : float
            The fraction of the dataset to keep.

        Examples
        --------
        >>> import antakia
        >>> import pandas as pd
        >>> X = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], columns=["a", "b"])
        >>> my_dataset = antakia.Dataset(X, model)
        >>> my_dataset.frac(0.5)
        >>> my_dataset.X
              a  b
        0     1  2
        1     5  6
        """
        self.X = self.X_all.sample(frac=p, random_state=9)
        self.frac_indexes = deepcopy(self.X.index)
        self.X_scaled = self.X_scaled.iloc[self.frac_indexes].reset_index(drop=True)
        self.y_pred = self.y_pred.iloc[self.frac_indexes].reset_index(drop=True)
        if self.y is not None:
            self.y = self.y.iloc[self.frac_indexes].reset_index(drop=True)
        self.fraction = p
        self.X.reset_index(drop=True, inplace=True)

    def setLongLat(self, long:str, lat:str):
        """
        Sets the longitude and latitude columns of the dataset.

        Parameters
        ---------
        long : str
            The name of the longitude column.
        lat : str
            The name of the latitude column.
        """
        self.long = long
        self.lat = lat

    def getLongLat(self):
        """
        Returns the longitude and latitude columns of the dataset.

        Returns
        -------
        long : str
            The name of the longitude column.
        lat : str
            The name of the latitude column.
        """
        return self.long, self.lat

    def improve(self):
        """
        Improves the dataset. Displays a widget to modify the dataset. For each feature, you can change its name, its type, its comment and if it is sensible or not.
        You also have the access to the general informations of the dataset.
        """
        general_infos = v.Row(class_="ma-2", children=[
            v.Icon(children=["mdi-database"], size="30px"),
            v.Html(tag="h3", class_="mb-3 mt-3 ml-4", children=[
                str(self.X.shape[0]) + " observations, " + str(self.X.shape[1]) + " features"
                ])])
        liste_slides = []
        for i in range(self.X.shape[1]):
            infos = [min(self.X.iloc[:,i]), max(self.X.iloc[:,i]), np.mean(self.X.iloc[:,i]), np.std(self.X.iloc[:,i])]
            infos = [round(infos[j], 3) for j in range(len(infos))]
            liste_slides.append(gui_elements.create_slide_dataset(self.X.columns[i], i+1, self.X.dtypes[i], len(self.X.columns), self.comments[i], self.sensible[i], infos))

        slidegroup = v.SlideGroup(
            v_model=None,
            class_="ma-3 pa-3",
            elevation=4,
            center_active=True,
            show_arrows=True,
            children=liste_slides,
        )

        def changement_sensible(widget, event, data):
            i = int(widget.class_)-1
            if widget.v_model :
                liste_slides[i].children[0].color = "red lighten-5"
                self.sensible[i] = True
            else:
                liste_slides[i].children[0].color = "white"
                self.sensible[i] = False

        def changement_names(widget, event, data):
            i = widget.value-1
            self.X = self.X.rename(columns={self.X.columns[i]: widget.v_model})

        def changement_type(widget, event, data):
            i = widget.value-1
            widget2 = liste_slides[i].children[0].children[-1].children[1].children[0]
            try :
                self.X = self.X.astype({self.X.columns[i]: widget2.v_model})
            except:
                print("The type of the column " + self.X.columns[i] + " cannot be changed to " + widget2.v_model)
                widget.color = "error"
                time.sleep(2)
                widget.color = ""
            else:
                widget.color = "success"
                time.sleep(2)
                widget.color = ""

        def changement_comment(widget, event, data):
            i = widget.value-1
            self.comments[i] = widget.v_model

        for i in range(len(liste_slides)):
            liste_slides[i].children[0].children[-1].children[2].on_event("change", changement_sensible)
            liste_slides[i].children[0].children[-1].children[3].on_event("change", changement_comment)
            liste_slides[i].children[0].children[0].children[0].on_event("change", changement_names)
            liste_slides[i].children[0].children[-1].children[1].children[-1].on_event("click", changement_type)

        widget = v.Col(children=[
            general_infos,
            slidegroup,
        ])
        display(widget)


