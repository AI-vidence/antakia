def create_save(liste=None, nom: str = "Default name", sub_models: list = None):
    """
    Function that allows to create a save file from a list of pre-defined regions.
    The save file is a json file that contains a list of dictionaries, usually generated in the interface (see antakia.interface).

    Parameter
    ---------
    liste : list
        The list of pre-defined regions to save.
    nom : str
        The name of the save file.
    sub_models : list
        The list of sub_models used to generate the pre-defined regions.

    Returns
    -------
    retour : dict
        A dictionary containing the name of the save file, the list of pre-defined regions and the list of sub_models used to generate the pre-defined regions.
    """
    if sub_models is None:
        sub_models = []
    return {"nom": nom, "liste": liste, "sub_models": sub_models}
