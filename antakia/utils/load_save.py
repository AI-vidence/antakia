import json


def load_save(local_path: str = None):
    """
    Function that allows to load a save file.
    The save file is a json file that contains a list of dictionaries, usually generated in the interface (see antakia.interface).

    Parameter
    ---------
    local_path : str
        The path to the save file. If None, the function will return a message saying that no save file was loaded.

    Returns
    -------
    data : list
        A list of dictionaries, each dictionary being a save file. This list can directly be passed to the function antakia.interface so as to load the save file.
    """
    if local_path is None:
        return "Aucun fichier de sauvegarde n'a été chargé"
    with open(local_path) as json_file:
        data = json.load(json_file)
    for dictio in data:
        dictio["sub_models"] = eval(dictio["sub_models"] + "()")
    return data
