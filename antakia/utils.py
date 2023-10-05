"""
Utils module for the antakia package.
"""

import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster

from copy import deepcopy

import mvlearn

from skrules import SkopeRules
import sklearn.cluster

import json
import textwrap
import logging
from logging import Logger, Handler

import ipywidgets as widgets
from ipywidgets.widgets.widget import Widget
import ipyvuetify as v


def simpleType(o) -> str:
    if isinstance(o, pd.DataFrame) : return "Dataframe " + str(o.shape)
    elif isinstance(o, pd.Series) :  return "Series " + str(o.shape)
    else : return type(o)

class OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget"""

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {"width": "100%", "height": "160px", "border": "1px solid black", "overflow_y" : "auto"}
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """Overload of logging.Handler method"""
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def show_logs(self):
        """Show the logs"""
        display(self.out)

    def clear_logs(self):
        """Clear the current logs"""
        self.out.clear_output()

def confLogger(logger : Logger) -> Handler:
    handler = OutputWidgetHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s:%(module)s|%(lineno)s:: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return handler

def wrapRepr(widget : Widget, size : int = 200) -> str:
    text = widget.__repr__()
    if widget.layout is None :
        text += " Layout is None !"
    else :
        text += " Visibility : "+ widget.layout.visibility
    s_wrap_list = textwrap.wrap(text, size)
    return  '\n'.join(s_wrap_list)

def overlapHandler(ens_potatoes, liste):
    # function that allows you to manage conflicts in the list of regions.
    # indeed, as soon as a region is added to the list of regions, the points it contains are removed from the other regions
    gliste = [x.indexes for x in ens_potatoes]
    for i in range(len(gliste)):
        a = 0
        for j in range(len(gliste[i])):
            if gliste[i][j - a] in liste:
                gliste[i].pop(j - a)
                a += 1
    for i in range(len(ens_potatoes)):
        ens_potatoes[i].setIndexes(gliste[i])
    return ens_potatoes

def _inverrtList(aList : list, size : int) -> list:
    newList = [[] for _ in range(size)]
    for i in range(len(aList)):
        newList[aList[i]].append(i)
    return newList


def _restList(l):
    l = list(l)
    for i in range(max(l) + 1):
        if l.count(i) == 0:
            l = list(np.array(l) - 1)
    return l


def _find_best_k(X, indices, recall_min, precision_min):
    recall_min = 0.7
    precision_min = 0.7
    new_X = X.iloc[indices]
    ind_f = 2
    for i in range(2, 9):
        #kmeans = sklearn.cluster.KMeans(n_clusters=i, random_state=9, n_init="auto")
        agglo = sklearn.cluster.AgglomerativeClustering(n_clusters=i)
        agglo.fit(new_X)
        labels = agglo.labels_
        a = True
        for j in range(max(labels) + 1):
            y = []
            for k in range(len(X)):
                if k in indices and labels[indices.index(k)] == j:
                    y.append(1)
                else:
                    y.append(0)
            skope_rules_clf = SkopeRules(
                feature_names=new_X.columns,
                random_state=42,
                n_estimators=5,
                recall_min=recall_min,
                precision_min=precision_min,
                max_depth_duplication=0,
                max_samples=1.0,
                max_depth=3,
            )
            skope_rules_clf.fit(X, y)
            if len(skope_rules_clf.rules_) == 0:
                ind_f = i - 1
                a = False
                break
        if a == False:
            break
    return 2 if ind_f == 1 else ind_f


def proposeAutoDyadicClustering (X, SHAP, n_clusters, default):
    m_kmeans = mvlearn.cluster.MultiviewKMeans(n_clusters=n_clusters, random_state=9)
    l = m_kmeans.fit_predict([X, SHAP])
    nombre_clusters = 0
    if default != False:
        X_train = pd.DataFrame(X.copy())
        recall_min = 0.8
        precision_min = 0.8
        max_ = max(l) + 1
        l_copy = deepcopy(l)
        for i in range(n_clusters):
            y_train = [1 if x == i else 0 for x in l_copy]
            indices = [i for i in range(len(y_train)) if y_train[i] == 1]
            skope_rules_clf = SkopeRules(
                feature_names=X_train.columns,
                random_state=42,
                n_estimators=5,
                recall_min=recall_min,
                precision_min=precision_min,
                max_depth_duplication=0,
                max_samples=1.0,
                max_depth=3,
            )
            skope_rules_clf.fit(X_train, y_train)
            if len(skope_rules_clf.rules_) == 0:
                k = _find_best_k(X, indices, recall_min, precision_min)
                nombre_clusters += k
                #kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=9)
                agglo = sklearn.cluster.AgglomerativeClustering(n_clusters=k)
                agglo.fit(X.iloc[indices])
                labels = np.array(agglo.labels_) + max_
                max_ += k
                l[indices] = labels
            else :
                nombre_clusters +=1
        l = _restList(l)
    l = list(np.array(l) - min(l))
    return _inverrtList(l, max(l) + 1), l


def create_save(atk, liste, name: str = "Default name"):
    '''
    Return a save file from a list of pre-defined regions.

    Function that allows to create a save file from a list of pre-defined regions.
    The save file is a json file that contains a list of dictionaries, usually generated in the interface (see antakia.interface).

    Parameters
    ---------
    liste : list
        The list of pre-defined regions to save.
    name : str
        The name of the save file.
    Returns
    -------
    retour : dict
        A dictionary containing the name of the save file, the list of pre-defined regions.
    
    Examples
    --------
    >>> import antakia
    >>> my_save = antakia.create_save([0,1,1,0], "save")
    >>> my_save
    {'name': 'save', 'liste': [[0, 1], [2, 3]]}
    '''

    l = np.array([[] for _ in range(max(liste) + 1)]).tolist()
    for i in range(len(liste)):
        l[liste[i]].append(i)
    l = [x for x in l if x != []]
    for i in range(len(l)):
        l[i] = Potato(atk, l[i])
    return {"name": name, "regions": l, "labels": liste}


def fromRules(df, rules_list):
    """Return a dataframe with specific rules applied to it.

    The rules are meant to be generated by the function antakia.results after creating a list of regions.

    Parameters
    ---------
    df : pandas dataframe
        The dataframe to which the rules will be applied
    rules_list : list
        The list of rules to apply to the dataframe. The rules must be in the form of a list of lists, each list being a rule in the following format:
        [minimum, operator1, column, operator2, maximum].
        For example : [0.5, '<=', 'cool_feature', '<=', 0.7]

    Returns
    -------
    df : pandas dataframe
        The dataframe with the rules applied to it

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 2.5], 'b': [0.1, 0.2, 0.3, 0.4, 0.5, 0.33]})
    >>> rules_list = [[1.5, '<=', 'a', '<=', 3], [0.1, '<=', 'b', '<=', 0.4]]
    >>> df = fromRules(df, rules_list)
    >>> df
         a    b
    1  2.0  0.2
    2  3.0  0.3
    3  2.5  0.33

    """
    l = rules_list
    for i in range(len(l)):
        regle1 = "df.loc[" + str(l[i][0]) + l[i][1] + "df['" + l[i][2] + "']]"
        regle2 = "df.loc[" + "df['" + l[i][2] + "']" + l[i][3] + str(l[i][4]) + "]"
        df = eval(regle1)
        df = eval(regle2)
    return df


def score(y, y_chap):
    y = np.array(y)
    y_chap = np.array(y_chap)
    return round(np.sqrt(sum((y - y_chap) ** 2) / len(y)), 3)


def models_scores_to_str(X: pd.DataFrame, y: pd.Series, models: list) -> list:
    """
    Returns  a list with a string indicating scores of the models
    """
    # function that returns a list with the name/score/perf of the different models imported for a given X and y
    list = []
    for i in range(len(models)):
        l = []
        models[i].fit(X, y)
        l.append(models[i].__class__.__name__)
        l.append(str(round(models[i].score(X, y), 3)))
        l.append("MSE")
        l.append(models[i].predict(X))
        l.append(models[i])
        list.append(l)
    return list