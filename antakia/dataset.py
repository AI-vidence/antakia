import pandas as pd
import numpy as np

# import antakia.longtask as LongTask

# TODO : these references to IPython should be removed in favor of a new scheme (see Wiki)
import ipyvuetify as v
import ipywidgets as widgets
from IPython.display import display 

from sklearn.preprocessing import StandardScaler

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

        Returns
        -------
        Dataset object
            A Dataset object.
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

        self.y_pred = self.model.predict(self.X)

        self.verbose = None
        self.widget = None

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
    
    def __create_progress(self, titre:str):
        widget = v.Col(
            class_="d-flex flex-column align-center",
            children=[
                    v.Html(
                        tag="h3",
                        class_="mb-3",
                        children=["Compute " + titre + " values"],
                ),
                v.ProgressLinear(
                    style_="width: 80%",
                    v_model=0,
                    color="primary",
                    height="15",
                    striped=True,
                ),
                v.TextField(
                    class_="w-100",
                    style_="width: 100%",
                    v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
                    readonly=True,
                ),
            ],
        )
        return widget
    
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
        >>> my_dataset = antakia.Dataset(X)
        >>> my_dataset.frac(0.5)
        >>> my_dataset.X
              a  b
        0     1  2
        1     5  6
        """

        self.X = self.X_all.sample(frac=p, random_state=9)
        self.y_pred = self.y_pred.sample(frac=p, random_state=9)
        if self.y is not None:
            self.y = self.y.sample(frac=p, random_state=9)

    def improve(self):
        """
        Improves the dataset.
        """
        colonnes = [
                {"text": c, "sortable": True, "value": c} for c in self.X.columns
            ]
        self.widget = v.DataTable(
            v_model=[],
            headers=colonnes,
            items=self.X.to_dict("records"),
        )
        display(self.widget)


