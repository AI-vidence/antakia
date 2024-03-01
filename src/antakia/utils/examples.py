from io import StringIO
import pandas as pd
import urllib.request
import requests
import json
from importlib.resources import files
import os

import logging
from antakia.utils.logging_utils import conf_logger

logger = logging.getLogger(__name__)
conf_logger(logger)

AVAILABLE_EXAMPLES = {
    "california_housing": [],
    "climate_change_survey": ["X_train", "X_test", "y_train", "y_test"],
    "wages": [],
    "titanic": ["train", "test"]
}

BRANCH = "dev"  # TODO change to "main" when merging


def _get_github_url(example: str, dataset_name: str = None) -> str:
    """
    Returns the Github url of the dataset from our examples/data folder

    Parameters
    ----------
    example : str
        The name of the example. Must be one of the following: "california_housing", "climate_change_survey", "wages"
    dataset_name : str (optional)
        The name of the dataset if needed. For example: "X_train", "X_test", "y_train", "y_test"
    
    Returns 
    -------
    str
        The Github url of the dataset requested of the specified example.
    """

    base_url = f"https://raw.githubusercontent.com/AI-vidence/antakia/{BRANCH}/examples/data/"

    if example not in AVAILABLE_EXAMPLES:
        raise ValueError(
            f"Example {example} not found, dataset should be one of {list(AVAILABLE_EXAMPLES.keys())}")
    if AVAILABLE_EXAMPLES[example] and dataset_name not in AVAILABLE_EXAMPLES[example]:
        raise ValueError(f"Dataset must be one of the following : {', '.join(AVAILABLE_EXAMPLES[example])}")
    if AVAILABLE_EXAMPLES[example]:
        url = base_url + example + "/" + dataset_name.lower() + ".csv"
        logger.debug(f"Dataset {example}/{dataset_name} download URL : {url}")
    else:
        url = base_url + example + ".csv"
        logger.debug(f"Dataset {example} download URL : {url}")
    return url


def _get_lfs_pointer(url: str) -> tuple[str, str, int]:
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


def _get_download_url(oid: str, file_size: int) -> str:
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


def _get_remote_dataset(example: str, dataset_name: str = None) -> pd.DataFrame:
    """
    Fetchs the dataset from github repository examples/data folder. Called by our examples notebooks.

    Parameters
    ----------
    example : str
        The name of the example. Must be one of the following: "california_housing", "climate_change_survey", "wages", "titanic"
    dataset_name : str (optional)
        The name of the dataset if needed. For example: "X_train", "X_test", "y_train", "y_test"

    Returns
    -------
    pd.DataFrame
        A DataFrame corresponding to the dataset requested of the specified example.
    """
    _, oid, file_size = _get_lfs_pointer(_get_github_url(example, dataset_name))
    url = _get_download_url(oid, file_size)
    r = requests.get(url)
    logger.debug(f"Fetched dataset file: {r.text}")
    return pd.read_csv(StringIO(r.text))


def fetch_dataset(example: str, dataset_name: str = None) -> pd.DataFrame | dict[str:pd.DataFrame]:
    """
    Fetchs the dataset from our examples/data folder. Called by our examples notebooks.

    Parameters
    ----------
    example : str
        The name of the example. Must be one of the following: "california_housing", "climate_change_survey", "wages", "titanic"
    dataset_name : str (optional)
        The name of the dataset if needed. For example: "X_train", "X_test", "y_train", "y_test"
    
    Returns 
    -------
    pd.DataFrame
        A DataFrame corresponding to the dataset requested of the specified example.
    """
    if dataset_name:
        file_name = f'{example}-{dataset_name}.csv'
    elif AVAILABLE_EXAMPLES[example]:
        datasets = {}
        for name in AVAILABLE_EXAMPLES[example]:
            datasets[name] = fetch_dataset(example, name)
        return datasets
    else:
        file_name = f'{example}.csv'
    example_folder = files("antakia").joinpath("assets/examples/")
    location = str(example_folder.joinpath(f"{file_name}"))
    if os.path.exists(location):
        dataset = pd.read_csv(location)
        if 'Unnamed: 0' == dataset.columns[0]:
            dataset = dataset.set_index(dataset.columns[0])
            dataset.index.name=None
        return dataset
    else:
        if not os.path.exists(example_folder):
            os.mkdir(example_folder)
        dataset = _get_remote_dataset(example, dataset_name)
        dataset.to_csv(location, index=False)
        if 'Unnamed: 0' == dataset.columns[0]:
            dataset = dataset.set_index(dataset.columns[0])
            dataset.index.name = None
        return dataset
