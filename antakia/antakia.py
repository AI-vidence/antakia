import pandas as pd

from dotenv import load_dotenv

from antakia.data import ExplanationMethod, Variable, is_valid_model
from antakia.gui.gui import GUI


class AntakIA():
    """
    AntakIA class.

    Antakia instances provide data and methods to explain a ML model.

    Instance attributes
    -------------------
    X_list : a list of one or more pd.DataFrame 
    X_method_list : a list starting with ExplanationMethod.NONE, followed by one or more ExplanationMethod
    y : a pd.Series
    Y_pred : a pd.Series
    variables : a list of Variables, describing X_list[0]
    model : Model
        the model to explain

    regions : List of Selection objects


    """

    def __init__(self, X:pd.DataFrame, y:pd.Series, model, variables=None, X_exp:pd.DataFrame=None):
        """
        AntakiIA constructor.

        Parameters:
        X : a pd.DataFrame
        """

        config = load_dotenv() 

        if not is_valid_model(model):
            raise ValueError(model, " should implement predict and score methods")
        
        self.X = X
        self.y = y
        self.model = model
        self.Y_pred = model.predict(X)
        self.variables = variables

        # It's common to have column names ending with _shap, so we remove them
        X_exp.columns = X_exp.columns.str.replace('_shap', '')
        self.X_exp = X_exp
        

        if self.variables is not None:
            if isinstance(self.variables, list):
                self.variables = Variable.import_variable_list(variables)
                if len(self.variables) != len(X[0].columns):
                    raise ValueError("Provided variable list must be the same length of the dataframe")
            elif isinstance(self.variables, pd.DataFrame):
                self.variables = Variable.import_variable_df(variables)
            else:
                raise ValueError("Provided variable list must be a list or a pandas DataFrame")
        else:
            self.variables = Variable.guess_variables(X[0])

        self.regions = []

    def start_gui(self)-> GUI:
        return GUI(self.X, self.y, self.model, self.variables, self.X_exp).show_splash_screen()