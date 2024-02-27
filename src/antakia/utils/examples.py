from io import StringIO
import pandas as pd
import urllib.request
import requests
import json

import logging
from antakia.utils.logging import conf_logger

logger = logging.getLogger(__name__)
conf_logger(logger)

AVAILABLE_EXAMPLES = ["california_housing", "climate_change_survey", "wages", "titanic"]

BRANCH = "dev"  # TODO change to "main" when merging


def get_github_url(dataset_name: str, frame_name: str = None) -> str:
    """
    Returns the Github url of the dataset from our examples/data folder

    Parameters
    ----------
    dataset_name : str
        The name of the example. Must be one of the following: "california_housing", "climate_change_survey", "wages"
    frame_name : str (optional)
        The name of the dataset if needed. For example: "X_train", "X_test", "y_train", "y_test"
    
    Returns 
    -------
    str
        The Github url of the dataset requested of the specified example.
    """

    base_url = "https://raw.githubusercontent.com/AI-vidence/antakia/" + BRANCH + "/examples/data/"

    if dataset_name not in AVAILABLE_EXAMPLES:
        raise ValueError(f"Example {dataset_name} not found, dataset should be one of {AVAILABLE_EXAMPLES}")
    elif dataset_name == "climate_change_survey":
        if frame_name not in ["X_train", "X_test", "y_train", "y_test"]:
            raise ValueError(f"Dataset must be one of the following : 'X_train', 'X_test', 'y_train', 'y_test'.")
        url = base_url + dataset_name + "/" + frame_name.lower() + ".csv"
        logger.debug(f"Dataset {dataset_name}/{frame_name} download URL : {url}")
        return url
    elif dataset_name == "titanic":
        if frame_name not in ["train", "test"]:
            raise ValueError(f"Dataset must be one of the following : 'train', 'test'.")
        url = base_url + dataset_name + "-" + frame_name + ".csv"
        logger.debug(f"Dataset {dataset_name}/{frame_name} download URL : {url}")
    else:
        url = base_url + dataset_name + ".csv"
        logger.debug(f"Dataset {dataset_name}/{frame_name} download URL : {url}")
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
    f = open(path, 'r')  # TODO would be better not to use a file -> we should use requests.get(url).text
    meta = f.read().strip().split('\n')
    f.close()
    server_url = meta[0].split(' ')[1]
    sha256 = meta[1].split(':')[1]
    file_size = int(meta[2].split(' ')[1])
    logger.debug(f"LFS metadata : {server_url}, {sha256}, {file_size}")
    return server_url, sha256, file_size


def get_download_url(oid: str, file_size: int) -> str:
    """
    Returns the download URL of a LFS Github file given its oid and size

    Parameters
    ----------
    oid : str
        The OID of the Github LFS file
    file_size : int
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
                'size': file_size
            }
        ]
    }
    headers = {'content-type': 'application/json', 'Accept': 'application/vnd.git-lfs+json'}

    # Note the json.dumps(payload) that serializes the dict to a string, otherwise requests doesn't understand nested dicts !
    r = requests.post(url, data=json.dumps(payload), headers=headers)

    dwl_url = r.json()["objects"][0]['actions']['download']['href']
    logger.debug(f"Download URL : {dwl_url}")
    return dwl_url


def fetch_dataset(dataset_name: str, frame_name: str = None) -> pd.DataFrame:
    """
    Fetchs the dataset from our examples/data folder. Called by our examples notebooks.

    Parameters
    ----------
    dataset_name : str
        The name of the example. Must be one of the following: "california_housing", "climate_change_survey", "wages", "titanic"
    frame_name : str (optional)
        The name of the dataset if needed. For example: "X_train", "X_test", "y_train", "y_test"
    
    Returns 
    -------
    pd.DataFrame
        A DataFrame corresponding to the dataset requested of the specified example.
    """
    _, oid, file_size = get_lfs_pointer(get_github_url(dataset_name, frame_name))
    url = get_download_url(oid, file_size)
    r = requests.get(url)
    logger.debug(f"Fetched dataset file: {r.text}")
    return pd.read_csv(StringIO(r.text))
