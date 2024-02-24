
import pandas as pd
from urllib.request import urlretrieve

examples = ["california_housing", "climate_change_survey", "wages"]

branch = "dev"

def fetch_dataset(ds: str) -> dict[pd.DataFrame]:
    """
    Fetchs the dataset from our examples/data folder. Called by our examples notebooks.
    """
    base_url = "https://raw.githubusercontent.com/AI-vidence/antakia/"+branch+"/examples/data/"

    if ds not in examples:
        raise ValueError(f"Dataset {ds} not found")
    elif ds == "climate_change_survey":
        X_train = pd.read_csv(base_url+"climate_change_survey/x_train.csv")
        X_test = pd.read_csv(base_url+"climate_change_survey/x_test.csv")
        y_train = pd.read_csv(base_url+"climate_change_survey/y_train.csv")
        y_test = pd.read_csv(base_url+"climate_change_survey/y_test.csv")
        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    else:
        return {'raw':pd.read_csv(base_url + ds + ".csv")}


def notebook_url(nb: str) -> str:
    """
    returns the ipynb file URL specified by the notebook name.
    """
    base_url = "https://raw.githubusercontent.com/AI-vidence/antakia/"+branch+"/examples/"

    if nb not in examples :
        raise ValueError(f"Notebook {nb} not found")
    else : 
        return base_url+nb+".ipynb"
