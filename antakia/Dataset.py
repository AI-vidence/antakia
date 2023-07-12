import pandas as pd
import numpy as np
import antakia.LongTask as LongTask
import ipyvuetify as v
import ipywidgets as widgets
from IPython.display import display

class Dataset():
    """
    Dataset object.
    This object contains the data to explain.
    """

    def __init__(self, X:pd.DataFrame = None, model = None, csv:str = None, explanation: pd.DataFrame = None, y:pd.Series = None, y_pred:pd.Series = None):
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

        if y_pred is None:
            self.y_pred = self.model.predict(self.X)
        else:
            self.y_pred = y_pred

        self.explainability = dict()
        self.explainability["Imported"] = explanation
        self.explainability["SHAP"] = None
        self.explainability["LIME"] = None

        self.verbose = None
        self.widget = None

    def __str__(self):
        """
        Returns
        -------
        str
            A string containing the information about the dataset
        """
        texte = ' '.join(("Dataset:\n",
                    "------------------\n",
                    "      Number of observations:", str(self.X.shape[0]), "\n",
                    "      Number of variables:", str(self.X.shape[1]), "\n",
                    "Explanations:\n",
                    "------------------\n",
                    "      Imported:", str(self.explainability["Imported"] != None), "\n",
                    "      SHAP:", str(self.explainability["SHAP"] != None), "\n",
                    "      LIME:", str(self.explainability["LIME"] != None)))
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
    
    def frac(self, p:float = 0.2):
        self.X = self.X_all.sample(frac=p, random_state=9)
        self.y_pred = self.y_pred.sample(frac=p, random_state=9)
        if self.y is not None:
            self.y = self.y.sample(frac=p, random_state=9)

    def compute_SHAP(self, verbose:bool = True):
        """
        Computes the SHAP values of the dataset.
        """
        shap = LongTask.compute_SHAP(self.X, self.X_all, self.model)
        if verbose:
            self.verbose = self.__create_progress("SHAP")
            widgets.jslink((self.widget.children[1], "v_model"), (shap.progress_widget, "v_model"))
            widgets.jslink((self.widget.children[2], "v_model"), (shap.text_widget, "v_model"))
            display(self.widget)
        self.explainability["SHAP"] = shap.compute()

    def compute_LIME(self, verbose:bool = True):
        """
        Computes the LIME values of the dataset.
        """
        lime = LongTask.compute_LIME(self.X, self.X_all, self.model)
        if verbose:
            self.verbose = self.__create_progress("SHAP")
            widgets.jslink((self.widget.children[1], "v_model"), (lime.progress_widget, "v_model"))
            widgets.jslink((self.widget.children[2], "v_model"), (lime.text_widget, "v_model"))
            display(self.widget)
        self.explainability["LIME"] = lime.compute()

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


