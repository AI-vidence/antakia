
from io import StringIO
import pandas as pd
import urllib.request
import requests
import json

import logging
from antakia.utils.logging import conf_logger
logger = logging.getLogger(__name__)
conf_logger(logger)

examples = ["california_housing", "climate_change_survey", "wages"]

branch = "dev" #TODO change to "main" when merging

def get_github_url(ex: str, ds: str = None) -> str:
    """
    Returns the Github url of the dataset from our examples/data folder

    Parameters
    ----------
    ex : str
        The name of the example. Must be one of the following: "california_housing", "climate_change_survey", "wages"
    ds : str (optional)
        The name of the dataset if needed. For example: "X_train", "X_test", "y_train", "y_test"
    
    Returns 
    -------
    str
        The Github url of the dataset requested of the specified example.
    """
    
    base_url = "https://raw.githubusercontent.com/AI-vidence/antakia/"+branch+"/examples/data/"

    if ex not in examples:
        raise ValueError(f"Example {ex} not found")
    elif ex == "climate_change_survey":
        if ds not in ["X_train", "X_test", "y_train", "y_test"]:
            raise ValueError(f"Dataset must be one of the following : 'X_train', 'X_test', 'y_train', 'y_test'.")
        url = base_url + ex + "/" + ds.lower() + ".csv"
        logger.debug(f"Dataset {ex}/{ds} download URL : {url}")
        return url
    else:
        url = base_url + ex + ".csv"
        logger.debug(f"Dataset {ex}/{ds} download URL : {url}")
        return url

def get_lfs_pointer(url: str) -> tuple[str, str, int]:
    """
    Reads LFS file metadata from its GitHub file, given its Github URL 

    Parameters
    ----------
    url : str
        The Github URL of the dataset
    
    Returns
    -------
    tuple[str, str, int]
        The LFS server URL, the SHA256 hash and the size of the file
    
    """
    path, _ = urllib.request.urlretrieve(url)
    f = open(path, 'r') #TODO would be better not to use a file -> we should use requests.get(url).text
    meta = f.read().strip().split('\n')
    f.close()
    server_url = meta[0].split(' ')[1]
    sha256 = meta[1].split(':')[1]
    size = int(meta[2].split(' ')[1])
    logger.debug(f"LFS metadata : {server_url}, {sha256}, {size}")
    return (server_url, sha256, size)

def get_download_url(oid:str, size:int) -> str:
    """
    Returns the download URL of a LFS Github file given its oid and size

    Parameters
    ----------
    oid : str
        The OID of the Github LFS file
    size : int
        The size of the file, in bytes
    
    Returns
    -------
    str
        The ream download URL of the file
    """

    url = "https://github.com/AI-vidence/antakia.git/info/lfs/objects/batch"
    payload = {
        'operation': 'download', 
        'transfer': ['basic'], 
        'objects': [
            {
                'oid': oid, 
                'size': size
            }
        ]
    }
    headers = {'content-type': 'application/json', 'Accept': 'application/vnd.git-lfs+json'}
    
    # Note the json.dumps(payload) that serializes the dict to a string, otherwise requests doesn't understand nested dicts !
    r = requests.post(url, data= json.dumps(payload), headers=headers)

    dwl_url = r.json()["objects"][0]['actions']['download']['href']
    logger.debug(f"Download URL : {dwl_url}")
    return dwl_url


def fetch_dataset(ex: str, ds: str = None) -> pd.DataFrame:
    """
    Fetchs the dataset from our examples/data folder. Called by our examples notebooks.

    Parameters
    ----------
    ex : str
        The name of the example. Must be one of the following: "california_housing", "climate_change_survey", "wages"
    ds : str (optional)
        The name of the dataset if needed. For example: "X_train", "X_test", "y_train", "y_test"
    
    Returns 
    -------
    pd.DataFrame
        A DataFrame corresponding to the dataset requested of the specified example.
    """
    _, oid, size = get_lfs_pointer(get_github_url(ex, ds))
    url = get_download_url(oid, size)
    r = requests.get(url)
    logger.debug(f"Fetched dataset file: {r.text}")
    return pd.read_csv(StringIO(r.text))