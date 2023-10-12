"""
    Class Selection

"""
import typing

import pandas as pd
import numpy as np
from skrules import SkopeRules

import json as JSON
from copy import deepcopy

from antakia.data import DimReducMethod, ExplanationMethod, Model, Dataset, ExplanationDataset

class Selection():
    """
    A Selection object!
    A Selection is a selection of points from the dataset, on wich the user can apply a surrogate-model.

    Attributes
    ----------
    _indexes : list
        The list of the indexes of the points
    _X : pandas dataframe
        A reference to the dataset of __atk.
    _type : int
        The type of the Selection.
    """

    # Class constants
    UNKNOWN=-1
    LASSO=0 # Manually defined
    SELECTION=1
    SKR=2 # Defined with Skope Rules
    REFINED_SKR=3 # Rules have been manually refined by the user
    REGION=4 # validated / to be stored in Regions
    JSON=5 # imported from JSON

    def __init__(self,  X: pd.DataFrame, indexes:list, type:int) :
        """
        Constructor of the class Selection.

        Parameters
        ----------
        X : a pd.dataframe
        indexes : list
            The list of the indexes of the points in the dataset.
        type : int
            May be UNKNOWN, LASSO, SKR, REFINED_SKR, REGION or JSON.
        """

        self.X = X
        if not ExplanationMethod.isValidExplanationType(explain_method):
            raise ValueError(explain_method, " is, not a valid explanation method")
        
        self._indexes = indexes
        self._type = type

        # if json_path is not None and indexes != []:
        #     raise ValueError("You can't provide a list and a json file")
        
        # if json_path is not None:
        #     self._type = Selection.JSON
        #     if json_path[-5:] != ".json":
        #         json_path += ".json"
        #     fileObject = open(json_path, "r")
        #     jsonContent = fileObject.read()
        #     self._indexes = JSON.loads(jsonContent)
        # else :
        #     self._indexes = indexes

        # We compute the Y mask list from the indexes
        self._ymask_list = []
        for i in range(self._ds.get_length()):
            if i in self._indexes:
                self._ymask_list.append(1)
            else :
                self._ymask_list.append(0)




    def is_empty(self) -> bool:
        return len(self._indexes) == 0


    def size(self) -> int:
        """
        Function that returns the shape of the Selection.

        Returns
        -------
        tuple
            The shape of the Selection.
        """
        return len(self._indexes) # TODO : we're supposed to return the shape of the data, not the number of points  
    
    @staticmethod
    def is_valid_type(type:int) -> bool:
        """
        Function that checks if the type is valid.

        Parameters
        ----------
        type : int
            The type to check.

        Returns
        -------
        bool
            True if the type is valid, False otherwise.
        """
        return type in [Selection.UNKNOWN, Selection.LASSO, Selection.SELECTION, Selection.SKR, Selection.REFINED_SKR, Selection.REGION, Selection.JSON] 


    # def ruleListToStr(self, valueSpaceRules : bool = True) -> str:
    #     """ Transcribes the rules of the Selection into a string
    #         valueSpaceRules : if True, the rules are in the value space, else they are in the explanation space
    #     """
    #     string = ""
    #     theList = self._theVSRules if valueSpaceRules else self._theESRules
    #     for rule in theList:
    #         for i in range(len(rule)):
    #             if type(rule[i]) == float:
    #                 string += str(np.round(float(rule[i]), 2))
    #             elif rule[i] is None:
    #                 string += "None"
    #             elif type(rule[i]) == list:
    #                 string+="{"
    #                 string += str(rule[i][0])
    #                 for j in range(1, len(rule[i])):
    #                     string += "," + str(rule[i][j])
    #                 string+="}"
    #             else:
    #                 string += str(rule[i])
    #             string += " "
    #     return string

    def set_type(self, type : int) :
        """
        Function that sets the type of the Selection.

        Parameters
        ----------
        type : int
            The new type of the Selection.
        """
        if not Selection.is_valid_type(type):
            raise ValueError("You must provide a valid Selection type")
        self._type = type

    def get_type(self) -> int :
        """
        Function that returns the type of the Selection.

        Returns
        -------
        int
            The type of the Selection.
        """
        return self._type

    def get_ymask_list(self) -> list:
        return self._ymask_list

    def setYLMaskList(self, y_mmask_list : list) -> None:
        self._ymask_list = y_mmask_list


    def get_indexes(self) -> list:
        """
        Function that returns the indexes of the Selection.

        Returns
        -------
        list
            The indexes of the Selection.
        """
        return self._indexes
    
    
    def set_new_indexes(self, indexes:list) -> None :
        """
        Function that sets the indexes of the Selection.

        Parameters
        ----------
        indexes : list
            The new indexes of the Selection.
        """
        self._indexes = indexes


    def set_indexes_with_json(self, json_path:str) -> None:
        if json_path[-5:] != ".json":
            json_path += ".json"
        fileObject = open(json_path, "r")
        jsonContent = fileObject.read()
        self._indexes = JSON.loads(jsonContent)

        self._type = Selection.JSON




    # def setAndApplySubModel(self, model : Model, score : float) :
    #     """
    #     Function that sets the sub-model of the Selection and apply it to the Selection.

    #     Parameters
    #     ----------
    #     model : model object
    #         The new sub-model of the Selection.
    #     score : float
    #         if not None, will be set otherwise will be computed
    #     """
    #     self._sub_model["model"] = model
    #     model.fit(self.getVSValuesX(), self.getVSValuesY())
    #     if score is None:
    #         self._sub_model["score"] = model.score(self.getVSValuesX(), self.getVSValuesY())
    #     else :
    #         self._sub_model["score"] = score

    def get_submodel(self):
    #     """
    #     Function that returns the sub-model of the Selection : a dict "model" and "score".

    #     Returns
    #     -------
    #     dict
    #         The sub-model of the Selection. Is the following format : {"model": model object, "score": score of the model}.
    #     """
    #     return self._sub_model
    
    # @staticmethod
    # def get_clean_rules_and_score(rules, df):
    #     """ Transforms a raw rules list from Skope Ryles into a pretty list
    #     and returns the score of the rules.
    #     """
    #     rules = rules[0] # Only the 1st list is relevant
    #     scoreTuple = (round(rules[1][0], 3), round(rules[1][1], 3), rules[1][2])
    #     rules = rules[0]
    #     rules = rules.split(" and ")
    #     rules_list = []
    #     for i in range(len(rules)):
    #         rules[i] = rules[i].split(" ")
    #         l = [None]*5
    #         l[2] = rules[i][0]
    #         l[1] = "<="
    #         l[3] = "<="
    #         if "<=" in rules[i] or "<" in rules[i]:
    #             l[0] = round(float(min(df[l[2]])), 3)
    #             l[4] = round(float(rules[i][-1]), 3)
    #         elif ">=" in rules[i] or ">" in rules[i]:
    #             l[0] = round(float(rules[i][-1]), 3)
    #             l[4] = round(float(max(df[l[2]])),3)
    #         rules_list.append(l)
    #     return rules_list, scoreTuple


    def compute_skope_rules(self, p:float = 0.7, r:float = 0.7): #317
        """
        Computes the Skope-rules algorithm for the Selection, in both VS and ES spaces.

        Parameters
        ----------
        p : float
            The minimum precision of the rules, defaults to 0.7
        r : float 
            The minimum recall of the rules, defaults to 0.7
        """
        
        # We reintialize our Y mask list
        self._ymask_list = np.zeros(self._dataset.__len__())
        # We set it to 1 whenever the index is in the Selection
        self._ymask_list[self._indexes] = 1

        # We fit the classifier on the whole Dataset
        vs_skr_classifier = SkopeRules(
            feature_names=self._dataset.getFullValues(Dataset.REGULAR).columns,
            random_state=42,
            n_estimators=5,
            recall_min=r, # the only value we set
            precision_min=p, # the only value we set
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )
        vs_skr_classifier.fit(self._dataset.getFullValues(Dataset.REGULAR)()[self._ymask_list])

        # Idem for ES space : we fit the classifier on the whole ExplainaitionsDataset
        es_skr_classifier = SkopeRules(
            feature_names=self._explanations.getFullValues(self.explain_method).columns,
            random_state=42,
            n_estimators=5,
            recall_min=r,
            precision_min=p,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )
        es_skr_classifier.fit(self._explanations.getFullValues(self.explain_method), self._ymask_list)

        if vs_skr_classifier.rules_ == [] or es_skr_classifier.rules_ == []:
            self._theVSRules, self._theESRules = [], []
            self._theVSScores, self._theESScores = 0, 0
            self._rulesIdentified = False
        else :
            self._theVSRules, self._theVSScores = Selection.get_clean_rules_and_score(vs_skr_classifier.rules_, self._dataset.getFullValues(Dataset.REGULAR))
            self._theESRules, self._theESScores = Selection.get_clean_rules_and_score(es_skr_classifier.rules_, self._explanations.getFullValues(self._explanationMethod))
            self.reveal_intervals()
            self.set_indexes_with_rules()
            self._rulesIdentified = True
            self._type = Selection.SKR



    def set_indexes_with_rules(self) -> list :
        # TODO : understand
        # solo_features = list(set([self._theVSRules[i][2] for i in range(len(self._theVSRules))]))
        # number_features_rules = []
        # for i in range(len(solo_features)):
        #     number_features_rules.append([])
        # for i in range(len(self._theVSRules)):
        #     number_features_rules[solo_features.index(self._theVSRules[i][2])].append(self._theVSRules[i])

        # new_indexes = self._indexes
        # for i in range(len(number_features_rules)):
        #     temp_new_indexes = []
        #     for j in range(len(number_features_rules[i])):
        #         X_temp = self.atk.dataset.X[
        #             (self.atk.dataset.X[number_features_rules[i][j][2]] >= number_features_rules[i][j][0])
        #             & (self.atk.dataset.X[number_features_rules[i][j][2]] <= number_features_rules[i][j][4])
        #         ].index
        #         temp_new_indexes = list(temp_new_indexes) + list(X_temp)
        #         temp_new_indexes = list(set(temp_new_indexes))
        #     new_indexes = [g for g in new_indexes if g in temp_new_indexes]

        # # TODO : I don't understand this map thing
        # if self._mapIndexes is not None:
        #     new_indexes = [g for g in new_indexes if g in self._mapIndexes]

        # self.set_new_indexes(new_indexes)
        # return new_indexes
        return None


    def reveal_intervals(self):
        """
        Function that checks if there are duplicates in the rules.
        A duplicate is a rule that has the same feature as another rule, but with a different threshold.
        """
        features = [self._theVSRules[i][2] for i in range(len(self.rules))]
        features_alone = list(set(features))
        if len(features) == len(features_alone):
            return
        else :
            for feature in features:
                if features.count(feature) > 1:
                    a=0
                    for i in range(len(self._theVSRules)):
                        min_feature = -10e99
                        max_feature = 10e99
                        if self._theVSRules[i-a][2] == feature:
                            if self._theVSRules[i-a][0] > min_feature:
                                min_feature = self.rules[i-a][0]
                            if self._theVSRules[i-a][4] < max_feature:
                                max_feature = self._theVSRules[i-a][4]
                            self._theVSRules.pop(i-a)
                            a+=1
                    self._theVSRules.append([min_feature, "<=", feature, "<=", max_feature])

        # Same for ES
        features = [self._theESRules[i][2] for i in range(len(self._theESRules))]
        features_alone = list(set(features))
        if len(features) == len(features_alone):
            return
        else :
            for feature in features:
                if features.count(feature) > 1:
                    a=0
                    for i in range(len(self._theESRules)):
                        min_feature = -10e99
                        max_feature = 10e99
                        if self._theESRules[i-a][2] == feature:
                            if self._theESRules[i-a][0] > min_feature:
                                min_feature = self._theESRules[i-a][0]
                            if self._theESRules[i-a][4] < max_feature:
                                max_feature = self._theESRules[i-a][4]
                            self._theESRules.pop(i-a)
                            a+=1
                    self._theESRules.append([min_feature, "<=", feature, "<=", max_feature])

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
        if self._theVSRules is None:
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
        rules = self._theVSRules
        df = deepcopy(self._dataset.getFullValues(Dataset.REGULAR))
        rule1 = "df.loc[" + str(rules[index][0]) + rules[index][1] + "df['" + rules[index][2] + "']]"
        rule2 = "df.loc[" + "df['" + rules[index][2] + "']" + rules[index][3] + str(rules[index][4]) + "]"
        df = eval(rule1)
        df = eval(rule2)
        return df
    
    def toJSON(self):
        """
        Function that returns the Selection in the form of a json file.

        Returns
        -------
        json
            The Selection in the form of a json file.
        """
        return {"indexes": self._indexes, "type": self._type, "VSrules": self._theVSRules, "VSScore": self._theVSScores, "ESrules": self._theESRules, "ESScore": self._theESScores, "sub_model": self.__sub_models, "success": self._rulesIdentified}
    
def SelectionFromJson(atk, json:dict) -> Selection:
    """
    Function that loads a Selection from a json file.

    Parameters
    ----------
    atk : AntakIA object
        The AntakIA object linked to the Selection.
    json : json
        The json file containing the Selection.
    """
    Selection = Selection(atk, [])
    Selection.set_new_indexes(json["indexes"])
    Selection.state = json["state"]
    Selection.rules = json["rules"]
    Selection.score = json["score"]
    Selection.rules_exp = json["rules_exp"]
    Selection.score_exp = json["score_exp"]
    Selection.sub_model = json["sub_model"]
    Selection.success = json["success"]
    return Selection


def loadBackup(local_path):
    '''
    Return a backup file from a JSON file.

    Function that allows to load a save file.
    The save file is a json file that contains a list of dictionaries, usually generated in the interface (see antakia.gui).

    Parameters
    ----------
    local_path :str
        The path to the save file. If None, the function will return a message saying that no save file was loaded.

    Returns
    ----------
    dataList : list
        A list of dictionaries, each dictionary being a save file. This list can directly be passed to the AntakIA object so as to load the save file.
    
    Examples
    --------
    >>> import antakia
    >>> dataList = antakia.load_save("save.json")
    >>> dataList
    [{'name': 'Antoine's save', 'regions': [0, 1], [2, 3] 'labels': [0, 0, 1, 1]}]
    '''
    with open(local_path) as json_file:
        dataList = json.load(json_file)

    for temp in dataList:
        for i in range(len(temp["regions"])):
            temp["regions"][i] = Selection.SelectionFromJson(atk, temp["regions"][i])

    return dataList