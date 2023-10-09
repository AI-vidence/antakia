"""
    Class Potato

"""
import typing

import pandas as pd
import numpy as np
from skrules import SkopeRules

import json as JSON
from copy import deepcopy

from antakia.data import DimReducMethod, ExplanationMethod, Model, Dataset, ExplanationDataset

class Potato():
    """
    A Potato object!
    A Potato is a selection of points from the dataset, on wich the user can apply a surrogate-model.

    Attributes
    ----------
    _indexes : list
        The list of the indexes of the points in __atk's dataset.
    _dataset : Dataset object
        A reference to the dataset of __atk.
    _explanations : ExplanationDataset
    __explain_method : int
    _sub_model : a dict
        The surrogate-model ("model" key) of the selection. Could be None. And its score ("score" key).
    _theVSScore : tuple
        The score of the surrogate-model. Is the following format : (precision, recall, extract of the tree).
    _theESScore : tuple
        The score of the surrogate-model in the explanation space. Is the following format : (precision, recall, extract of the tree).
    _theVSRules : list
        The list of the rules that defines the selection.
    _theESRules : list
        The list of the rules that defines the selection in the explanation space.
    _success : bool
        True if the rules have been found, False otherwise.
    _type : int
        The type of the potato.
    _mapIndexes : ??
    """

    # Class constants
    UNKNOWN=-1
    LASSO=0 # Manually defined
    SELECTION=1
    SKR=2 # Defined with Skope Rules
    REFINED_SKR=3 # Rules have been manually refined by the user
    REGION=4 # validated / to be stored in Regions
    JSON=5 # imported from JSON

    def __init__(self,  ds: Dataset, xds: ExplanationDataset, explain_method: int, indexes:list, type:int, json_path: str = None) -> None:
        """
        Constructor of the class Potato.

        Parameters
        ----------
        ds : a Dataset object
        xds : an ExplanationDataset object
        currentExplanationMethod : the explanation currently used by the caller
        indexes : list
            The list of the indexes of the points in the dataset.
        type : int
            May be UNKNOWN, LASSO, SKR, REFINED_SKR, REGION or JSON.
        json_path : str
            The name of the json file containing the indexes of the points in the dataset.
        """

        self._dataset = ds
        self._explanations = xds
        if not ExplanationMethod.isValidExplanationType(explain_method):
            raise ValueError(explain_method, " is, not a valid explanation method")
        
        self._explain_method = explain_method # could be ExplanationMethod.SHAP or LIME for ex
        self._type = type

        if json_path is not None and indexes != []:
            raise ValueError("You can't provide a list and a json file")
        
        if json_path is not None:
            self._type = Potato.JSON
            if json_path[-5:] != ".json":
                json_path += ".json"
            fileObject = open(json_path, "r")
            jsonContent = fileObject.read()
            self._indexes = JSON.loads(jsonContent)
        else :
            self._indexes = indexes

        # We compute the Y mask list from the indexes
        self._yMaskList = []
        for i in range(len(self._dataset.getFullValues(Dataset.REGULAR))):
            if i in self._indexes:
                self._yMaskList.append(1)
            else :
                self._yMaskList.append(0)

        self._sub_model = {"model": None, "score": None} # model is a Model object, score an int ?


        self._theVSRules : list = []
        self._theVSScores : Tuple (int, int, int) # ?, recall and precision

        self._theESRules : list = []
        self._theESScores : Tuple (int, int, int) # ?,recall and precision

        self._rulesIdentified : list = []
        self._mapIndexes : int = 0


    def __str__(self) -> str:
        text = ' '.join(("Potato:\n",
                    "------------------\n",
                    "      Type:", self.typeToString(), "\n",
                    "      Number of points:", str(len(self._indexes)), "\n",
                    "      Percentage of the dataset:", str(round(100*len(self._indexes)/len(self._dataset.getFullValues(Dataset.REGULAR)), 2))+"%", "\n",
                    "      Sub-model:", str(self._sub_model["model"].__class__.__name__))) 
        return text
    
    def __len__(self) -> int:
        """
        The length of the potato.

        Returns
        -------
        int
            The length of the potato.
        """
        return len(self._indexes)

    def getVSValuesX(self, flavour : int = Dataset.REGULAR) -> list:
        """
        Returns the VS records of the Potato

        Returns
        -------
        list
            The VS records of the Potato
        """
        if flavour is not  None and not Dataset.isValidXFlavour(flavour) :
            raise ValueError("You must provide a valid flavour")
        return self._dataset.getFullValues(Dataset.REGULAR)[self._indexes]
    
    def getVSValuesY(self, flavour : int = Dataset.TARGET) -> list:
        """
        Returns the VS records of the Potato

        Returns
        -------
        list
            The VS records of the Potato
        """
        if flavor is not None and not Dataset.isValidYFlavour(flavour) :
            raise ValueError("You must provide a valid flavour")
        return self._dataset.getYValues(self._indexes, flavour)

    def getESvalues(self) -> list :
        """
        Returns the ES records of the Potato

        Returns
        -------
        list
            The ES records of the Potato
        """
        
        return self._explanations.getFullValues(Dataset.REGULAR)[self._indexes]
    
    def getMapIndexes(self):
        return self._mapIndexes

    def setMapIndexes(self, mapIndexes):
        self._mapIndexes = mapIndexes

    def is_empty(self) -> bool:
        return len(self._indexes) == 0


    def size(self) -> int:
        """
        Function that returns the shape of the potato.

        Returns
        -------
        tuple
            The shape of the potato.
        """
        return len(self._indexes) # TODO : we're supposed to return the shape of the data, not the number of points  
    
    @staticmethod
    def isValidType(type:int) -> bool:
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
        return type in [Potato.UNKNOWN, Potato.LASSO, Potato.SELECTION, Potato.SKR, Potato.REFINED_SKR, Potato.REGION, Potato.JSON] 


    def ruleListToStr(self, valueSpaceRules : bool = True) -> str:
        """ Transcribes the rules of the potato into a string
            valueSpaceRules : if True, the rules are in the value space, else they are in the explanation space
        """
        string = ""
        theList = self._theVSRules if valueSpaceRules else self._theESRules
        for rule in theList:
            for i in range(len(rule)):
                if type(rule[i]) == float:
                    string += str(np.round(float(rule[i]), 2))
                elif rule[i] is None:
                    string += "None"
                elif type(rule[i]) == list:
                    string+="{"
                    string += str(rule[i][0])
                    for j in range(1, len(rule[i])):
                        string += "," + str(rule[i][j])
                    string+="}"
                else:
                    string += str(rule[i])
                string += " "
        return string

    def setType(self, type : int) :
        """
        Function that sets the type of the potato.

        Parameters
        ----------
        type : int
            The new type of the potato.
        """
        if not Potato.isValidType(type):
            raise ValueError("You must provide a valid Potato type")
        self._type = type

    def getType(self) -> int :
        """
        Function that returns the type of the potato.

        Returns
        -------
        int
            The type of the potato.
        """
        return self._type

    def getYMaskList(self) -> list:
        return self._yMaskList

    def setYLMaskList(self, yMaskList : list) -> None:
        self._yMaskList = yMaskList


    def get_indexes(self) -> list:
        """
        Function that returns the indexes of the potato.

        Returns
        -------
        list
            The indexes of the potato.
        """
        return self._indexes
    
    
    def set_new_indexes(self, indexes:list) -> None :
        """
        Function that sets the indexes of the potato.

        Parameters
        ----------
        indexes : list
            The new indexes of the potato.
        """
        self._indexes = indexes
        self._rulesIdentified = False


    def set_indexes_with_json(self, json_path:str) -> None:
        if json_path[-5:] != ".json":
            json_path += ".json"
        fileObject = open(json_path, "r")
        jsonContent = fileObject.read()
        self._indexes = JSON.loads(jsonContent)

        self._type = Potato.JSON
        self.rulesIdentified = None

    def get_vs_rules(self):
        """
        Function that returns the rules of the potato.

        Returns
        -------
        list
            The rules of the potato.
        """
        return self._theVSRules
    

    def set_vs_rules(self, newRules : list):
        """
        Function that sets the rules of the potato.

        Parameters
        ----------
        newRules : list
            The new rules of the potato.
        """
        self._theVSRules = newRules 

    def get_es_rules(self):
        """
        Function that returns the rules of the potato in the explanation space.

        Returns
        -------
        list
            The rules of the potato in the explanation space.
        """
        return self._theESRules
    
    def get_vs_scores(self):
        """
        Function that returns the score of the potato.

        Returns
        -------
        tuple
            The score of the potato.
        """
        return self._theVSScores
    
    def get_es_scores(self):
        """
        Function that returns the score of the potato in the explanation space.

        Returns
        -------
        tuple
            The score of the potato in the explanation space.
        """
        return self._theESScores
    
    def has_rules_defined(self) -> bool:
        """
        Function that checks if the potato has rules defined.

        Returns
        -------
        bool
            True if the potato has rules defined, False otherwise.
        """
        return self._rulesIdentified

    def setAndApplySubModel(self, model : Model, score : float) :
        """
        Function that sets the sub-model of the potato and apply it to the potato.

        Parameters
        ----------
        model : model object
            The new sub-model of the potato.
        score : float
            if not None, will be set otherwise will be computed
        """
        self._sub_model["model"] = model
        model.fit(self.getVSValuesX(), self.getVSValuesY())
        if score is None:
            self._sub_model["score"] = model.score(self.getVSValuesX(), self.getVSValuesY())
        else :
            self._sub_model["score"] = score

    def get_submodel(self):
        """
        Function that returns the sub-model of the potato : a dict "model" and "score".

        Returns
        -------
        dict
            The sub-model of the potato. Is the following format : {"model": model object, "score": score of the model}.
        """
        return self._sub_model
    
    @staticmethod
    def get_clean_rules_and_score(rules, df):
        """ Transforms a raw rules list from Skope Ryles into a pretty list
        and returns the score of the rules.
        """
        rules = rules[0] # Only the 1st list is relevant
        scoreTuple = (round(rules[1][0], 3), round(rules[1][1], 3), rules[1][2])
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
        return rules_list, scoreTuple


    def apply_skope_rules(self, p:float = 0.7, r:float = 0.7): #317
        """
        Computes the Skope-rules algorithm for the Potato, in both VS and ES spaces.

        Parameters
        ----------
        p : float
            The minimum precision of the rules, defaults to 0.7
        r : float 
            The minimum recall of the rules, defaults to 0.7
        """
        
        # We reintialize our Y mask list
        self._yMaskList = np.zeros(self._dataset.__len__())
        # We set it to 1 whenever the index is in the potato
        self._yMaskList[self._indexes] = 1

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
        vs_skr_classifier.fit(self._dataset.getFullValues(Dataset.REGULAR)()[self._yMaskList])

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
        es_skr_classifier.fit(self._explanations.getFullValues(self.explain_method), self._yMaskList)

        if vs_skr_classifier.rules_ == [] or es_skr_classifier.rules_ == []:
            self._theVSRules, self._theESRules = [], []
            self._theVSScores, self._theESScores = 0, 0
            self._rulesIdentified = False
        else :
            self._theVSRules, self._theVSScores = Potato.get_clean_rules_and_score(vs_skr_classifier.rules_, self._dataset.getFullValues(Dataset.REGULAR))
            self._theESRules, self._theESScores = Potato.get_clean_rules_and_score(es_skr_classifier.rules_, self._explanations.getFullValues(self._explanationMethod))
            self.reveal_intervals()
            self.set_indexes_with_rules()
            self._rulesIdentified = True
            self._type = Potato.SKR



    def set_indexes_with_rules(self) -> list :
        # TODO : it seems we're dealing with VS rules ? To understand !!
        solo_features = list(set([self._theVSRules[i][2] for i in range(len(self._theVSRules))]))
        number_features_rules = []
        for i in range(len(solo_features)):
            number_features_rules.append([])
        for i in range(len(self._theVSRules)):
            number_features_rules[solo_features.index(self._theVSRules[i][2])].append(self._theVSRules[i])

        new_indexes = self._indexes
        for i in range(len(number_features_rules)):
            temp_new_indexes = []
            for j in range(len(number_features_rules[i])):
                X_temp = self.atk.dataset.X[
                    (self.atk.dataset.X[number_features_rules[i][j][2]] >= number_features_rules[i][j][0])
                    & (self.atk.dataset.X[number_features_rules[i][j][2]] <= number_features_rules[i][j][4])
                ].index
                temp_new_indexes = list(temp_new_indexes) + list(X_temp)
                temp_new_indexes = list(set(temp_new_indexes))
            new_indexes = [g for g in new_indexes if g in temp_new_indexes]

        # TODO : I don't understand this map thing
        if self._mapIndexes is not None:
            new_indexes = [g for g in new_indexes if g in self._mapIndexes]

        self.set_new_indexes(new_indexes)
        return new_indexes


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
        Function that prints the rules of the potato.
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
        Function that returns the potato in the form of a json file.

        Returns
        -------
        json
            The potato in the form of a json file.
        """
        return {"indexes": self._indexes, "type": self._type, "VSrules": self._theVSRules, "VSScore": self._theVSScores, "ESrules": self._theESRules, "ESScore": self._theESScores, "sub_model": self.__sub_models, "success": self._rulesIdentified}
    
def potatoFromJson(atk, json:dict) -> Potato:
    """
    Function that loads a potato from a json file.

    Parameters
    ----------
    atk : AntakIA object
        The AntakIA object linked to the potato.
    json : json
        The json file containing the potato.
    """
    potato = Potato(atk, [])
    potato.set_new_indexes(json["indexes"])
    potato.state = json["state"]
    potato.rules = json["rules"]
    potato.score = json["score"]
    potato.rules_exp = json["rules_exp"]
    potato.score_exp = json["score_exp"]
    potato.sub_model = json["sub_model"]
    potato.success = json["success"]
    return potato


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
            temp["regions"][i] = Potato.potatoFromJson(atk, temp["regions"][i])

    return dataList