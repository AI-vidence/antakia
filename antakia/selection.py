"""
    Class Selection

"""
import typing

import pandas as pd
import numpy as np
from skrules import SkopeRules

import json as JSON
from copy import deepcopy

from antakia.data import DimReducMethod, ExplanationMethod

class Selection():
    """
    A Selection object!
    A Selection is a selection of points from the dataset, on wich the user can apply a surrogate-model.

    Attributes
    ----------
    indexes : list
        The list of the indexes of the points
    type : int
        The type of the Selection.
    ymask_list : ???
    theVSRules
    theESRules

    """

    # Class constants
    UNKNOWN=-1
    LASSO=0 # Manually defined
    SELECTION=1
    SKR=2 # Defined with Skope Rules
    REFINED_SKR=3 # Rules have been manually refined by the user
    REGION=4 # validated / to be stored in Regions
    JSON=5 # imported from JSON

    def __init__(self, indexes:list, type:int) :
        if indexes is None:
            self.indexes = []
        else:
            self.indexes = indexes
        self.ymask_list = []
        self.type = type
        self.theVSRules = self.theESRules  = None

    def get_size(self) -> int:
        return len(self.indexes)

    def is_empty(self) -> bool:
        return self.get_size() == 0

    @staticmethod
    def is_valid_type(type:int) -> bool:
        return type in [Selection.UNKNOWN, Selection.LASSO, Selection.SELECTION, Selection.SKR, Selection.REFINED_SKR, Selection.REGION, Selection.JSON] 


    def compute_skope_rules(self, p:float = 0.7, r:float = 0.7):
        pass

    def set_indexes_with_rules(self) -> list :
        return []
    
    def reveal_intervals(self):
        features = [self.theVSRules[i][2] for i in range(len(self.rules))]
        features_alone = list(set(features))
        if len(features) == len(features_alone):
            return
        else :
            for feature in features:
                if features.count(feature) > 1:
                    a=0
                    for i in range(len(self.theVSRules)):
                        min_feature = -10e99
                        max_feature = 10e99
                        if self.theVSRules[i-a][2] == feature:
                            if self.theVSRules[i-a][0] > min_feature:
                                min_feature = self.rules[i-a][0]
                            if self.theVSRules[i-a][4] < max_feature:
                                max_feature = self.theVSRules[i-a][4]
                            self.theVSRules.pop(i-a)
                            a+=1
                    self.theVSRules.append([min_feature, "<=", feature, "<=", max_feature])


        features = [self.theESRules[i][2] for i in range(len(self.theESRules))]
        features_alone = list(set(features))
        if len(features) == len(features_alone):
            return
        else :
            for feature in features:
                if features.count(feature) > 1:
                    a=0
                    for i in range(len(self.theESRules)):
                        min_feature = -10e99
                        max_feature = 10e99
                        if self.theESRules[i-a][2] == feature:
                            if self.theESRules[i-a][0] > min_feature:
                                min_feature = self.theESRules[i-a][0]
                            if self.theESRules[i-a][4] < max_feature:
                                max_feature = self.theESRules[i-a][4]
                            self.theESRules.pop(i-a)
                            a+=1
                    self.theESRules.append([min_feature, "<=", feature, "<=", max_feature])

    def prettyPrint(self, table, ch1="-", ch2="|", ch3="+"):
        le_max = 0
        for i in range(len(table)):
            table[i][2] = table[i][2].replace("_", " ")
            if len(table[i][2]) > le_max:
                le_max = len(table[i][2])
            for j in range(len(table[i])):
                table[i][j] = str(table[i][j])
        
        for i in range(len(table)):
            if len(table[i][2]) < le_max:
                table[i][2] = " "*round((le_max - len(table[i][2]))/2 - 1) + table[i][2]
        
        if len(table) == 0:
            return

        max_lengths = [
            max(column)
            for column in zip(*[[len(cell) for cell in row] for row in table])
        ]

        for row in table:
            print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))
            print(
                ch2.join(
                    [
                        "",
                        *[
                            ("{:<" + str(l) + "}").format(c)
                            for l, c in zip(max_lengths, row)
                        ],
                        "",
                    ]
                )
            )
        print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))

    def printRules(self):
        """
        Function that prints the rules of the Selection.
        Note that we use VS Rules only
        """
        if self.theVSRules is None:
            print("No rules")
        else :
            self.prettyPrint(self.rules, ch3 = '-', ch2=" ")

    def respectOneRule(self, index:int):
        """
        Function that returns the points of the dataset that respect only one rule of the list of rules.

        Parameters
        ----------
        index : int
            The index of the rule to respect.

        Returns
        -------
        pandas dataframe
            The dataframe containing the points of the dataset that respect only one rule of the list of rules.
        """
        rules = self.theVSRules
        df = deepcopy(self.X)
        rule1 = "df.loc[" + str(rules[index][0]) + rules[index][1] + "df['" + rules[index][2] + "']]"
        rule2 = "df.loc[" + "df['" + rules[index][2] + "']" + rules[index][3] + str(rules[index][4]) + "]"
        df = eval(rule1)
        df = eval(rule2)
        return df