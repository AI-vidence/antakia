"""
CLass potato (selection)
"""

import pandas as pd
import numpy as np
from skrules import SkopeRules


class Potato():
    """
    An AntakIA Potato is a selection of data.
    """

    def __init__(self, X: pd.DataFrame) -> None:
        """
        Constructor of the class Potato.
        """
        self.indexes = []
        self.data = X.iloc[self.indexes]
        self.sub_model = None
        self.rules = None
        self.explanatory_rules = None
        self.__X = X
        self.score_skope = None

    def __str__(self) -> str:
        """
        Function that allows to print the Potato object.
        """
        texte = ' '.join(("Potato:\n",
                    "------------------\n",
                    "      Number of points:", str(len(self.indexes)), "\n",
                    "      Percentage of the dataset:", str(round(100*len(self.indexes)/len(self.__X), 2))+"%", "\n"))
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
    
    def apply_rules(self, rules):
        df = self.__X
        for i in range(len(rules)):
            regle1 = "df.loc[" + str(rules[i][0]) + rules[i][1] + "df['" + rules[i][2] + "']]"
            regle2 = "df.loc[" + "df['" + rules[i][2] + "']" + rules[i][3] + str(rules[i][4]) + "]"
            df = eval(regle1)
            df = eval(regle2)
        self.data = df
        self.indexes = df.index
    
    def __transform_rules(self, rules):
        rules = rules[0]
        score = (round(rules[1][0], 3), round(rules[1][1], 3), rules[1][2])
        rules = rules[0]
        rules = rules.split(" and ")
        print(rules)
        rules_list = []
        for i in range(len(rules)):
            rules[i] = rules[i].split(" ")
            l = [None]*5
            l[2] = rules[i][0]
            l[1] = "<="
            l[3] = "<="
            if "<=" in rules[i] or "<" in rules[i]:
                l[0] = round(float(min(self.__X[l[2]])), 3)
                l[4] = round(float(rules[i][-1]), 3)
            elif ">=" in rules[i] or ">" in rules[i]:
                l[0] = round(float(rules[i][-1]), 3)
                l[4] = round(float(max(self.__X[l[2]])),3)
            rules_list.append(l)
        return rules_list, score


    def apply_skope(self, p:float = 0.5, r:float = 0.5):
        y_train = np.zeros(len(self.__X))
        y_train[self.indexes] = 1
        skope_rules_clf = SkopeRules(
            feature_names=self.__X.columns,
            random_state=42,
            n_estimators=5,
            recall_min=r,
            precision_min=p,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )
        skope_rules_clf.fit(self.__X, y_train)
        self.rules, self.score_skope = self.__transform_rules(skope_rules_clf.rules_)
        self.data = self.apply_rules(self.__X, self.rules)
