"""
CLass potato (selection)
"""

import pandas as pd
import numpy as np
from skrules import SkopeRules

from antakia.dataset import Dataset

from copy import deepcopy

#from antakia import Dataset


class Potato():
    """
    An AntakIA Potato is a selection of data.
    """

    # TODO : ne faudrait-il pas gérer l'état d'une potato ? lasso ? skr ? skr-tuned ? validated region ?

    def __init__(self, indexes:list = [], dataset: Dataset = None) -> None:
        """
        Constructor of the class Potato.
        """
        self.state = None
        self.indexes = indexes
        self.dataset = dataset
        if self.dataset.X is not None:
            self.data = self.dataset.X.iloc[self.indexes]
        else :
            self.data = None
        self.sub_model = None

        self.rules = None
        self.score = None

        self.rules_exp = None
        self.score_exp = None

        self.success = None

        self.y = [0]*len(self.dataset.X)
        for i in self.indexes:
            self.y[i] = 1

    def __str__(self) -> str:
        """
        Function that allows to print the Potato object.
        """
        texte = ' '.join(("Potato:\n",
                    "------------------\n",
                    "      State:", str(self.state), "\n",
                    "      Number of points:", str(len(self.indexes)), "\n",
                    "      Percentage of the dataset:", str(round(100*len(self.indexes)/len(self.dataset.X), 2))+"%", "\n"))
        return texte
    
    def __len__(self) -> int:
        """
        Function that returns the number of points in the potato.
        """
        return len(self.indexes)
    
    def shape(self) -> tuple:
        """
        Function that returns the shape of the potato.
        """
        return self.data.shape
    
    def apply_rules(self):
        self.state = "skope-ruled"
        rules = self.rules
        df = self.dataset.X
        for i in range(len(rules)):
            regle1 = "df.loc[" + str(rules[i][0]) + rules[i][1] + "df['" + rules[i][2] + "']]"
            regle2 = "df.loc[" + "df['" + rules[i][2] + "']" + rules[i][3] + str(rules[i][4]) + "]"
            df = eval(regle1)
            df = eval(regle2)
        self.data = df
        self.indexes = df.index
    
    def __transform_rules(self, rules, df):
        rules = rules[0]
        score = (round(rules[1][0], 3), round(rules[1][1], 3), rules[1][2])
        rules = rules[0]
        rules = rules.split(" and ")
        rules_list = []
        for i in range(len(rules)):
            rules[i] = rules[i].split(" ")
            l = [None]*5
            l[2] = rules[i][0]
            l[1] = "<="
            l[3] = "<="
            if "<=" in rules[i] or "<" in rules[i]:
                l[0] = round(float(min(df[l[2]])), 3)
                l[4] = round(float(rules[i][-1]), 3)
            elif ">=" in rules[i] or ">" in rules[i]:
                l[0] = round(float(rules[i][-1]), 3)
                l[4] = round(float(max(df[l[2]])),3)
            rules_list.append(l)
        return rules_list, score


    def apply_skope(self, explanation, p:float = 0.5, r:float = 0.5):
        y_train = np.zeros(len(self.dataset.X))
        y_train[self.indexes] = 1

        skope_rules_clf = SkopeRules(
            feature_names=self.dataset.X.columns,
            random_state=42,
            n_estimators=5,
            recall_min=r,
            precision_min=p,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )
        skope_rules_clf.fit(self.dataset.X, y_train)

        skope_rules_clf_exp = SkopeRules(
            feature_names=self.dataset.explain[explanation].columns,
            random_state=42,
            n_estimators=5,
            recall_min=r,
            precision_min=p,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )
        skope_rules_clf_exp.fit(self.dataset.explain[explanation], y_train)
        if skope_rules_clf.rules_ == [] or skope_rules_clf_exp.rules_ == []:
            self.rules, self.score_skope, self.rules_exp, self.score_skope_exp = None, None, None, None
            self.success = False
        else :
            self.rules, self.score = self.__transform_rules(skope_rules_clf.rules_, self.dataset.X)
            self.rules_exp, self.score_exp = self.__transform_rules(skope_rules_clf_exp.rules_, self.dataset.explain[explanation])
            self.apply_rules()
            self.success = True

    def respect_one_rule(self, indice:int):
        rules = self.rules
        df = deepcopy(self.dataset.X)
        regle1 = "df.loc[" + str(rules[indice][0]) + rules[indice][1] + "df['" + rules[indice][2] + "']]"
        regle2 = "df.loc[" + "df['" + rules[indice][2] + "']" + rules[indice][3] + str(rules[indice][4]) + "]"
        df = eval(regle1)
        df = eval(regle2)
        return df