"""
Class potato (selection) and functions to create a potato from a json file.
"""

import pandas as pd
import numpy as np
from skrules import SkopeRules

import json as JSON
from copy import deepcopy

# from antakia.antakia import AntakIA
from antakia.dataset import Dataset

class Potato():
    """
    A Potato object!
    A Potato is a selection of points from the dataset, on wich the user can apply a surrogate-model.

    Attributes
    ----------
    atk : AntakIA object
        The AntakIA object linked to the potato.
    indexes : list
        The list of the indexes of the points in the dataset.
    dataset : Dataset object
        The Dataset object containing the data of the selection.
    data : pandas dataframe
        The dataframe containing the data of the selection.
    y : list
        The list of the target values of the selection.
    sub_model : model object
        The surrogate-model of the selection. Could be None.
    rules : list
        The list of the rules that defines the selection.
    score : tuple
        The score of the surrogate-model. Is the following format : (precision, recall, extract of the tree).
    rules_exp : list
        The list of the rules that defines the selection in the explanation space.
    score_exp : tuple
        The score of the surrogate-model in the explanation space. Is the following format : (precision, recall, extract of the tree).
    success : bool
        True if the rules have been found, False otherwise.
    y_train : list
        The list of the target values of the dataset.
    explain : dict
        The dict containing the explanations of the selection. Is the following format : {"Imported": imported explanations, "SHAP": SHAP explanations, "LIME": LIME explanations}.
    state : int
        The state of the potato.

    """
    UNKNOWN=-1
    LASSO=0 # Manually defined
    SKR=1 # Defined with Skope Rules
    REFINED_SKR=2 # Rules have been manually refined by the user
    REGION=3 # validated / to be stored in Regions
    JSON=4 # imported from JSON

    def __init__(self,  atk, array:list = [], json_path: str = None) -> None:
        """
        Constructor of the class Potato.

        Parameters
        ----------
        atk : AntakIA object
            The AntakIA object linked to the potato.
        array : list
            The list of the indexes of the points in the dataset.
        json_path : str
            The name of the json file containing the indexes of the points in the dataset.
        """
        import antakia
        if not isinstance(atk, antakia.AntakIA):
            raise ValueError("You must provide an AntakIA object")
        self.atk = atk
        self.state = Potato.UNKNOWN

        if json_path is not None and array != []:
            raise ValueError("You can't provide a list and a json file")
        
        if json_path is not None:
            self.state = Potato.JSON
            if json_path[-5:] != ".json":
                json_path += ".json"
            fileObject = open(json_path, "r")
            jsonContent = fileObject.read()
            self.indexes = JSON.loads(jsonContent)
        else :
            self.indexes = array

        self.dataset = atk.dataset
        if self.dataset.X is not None:
            self.data = self.dataset.X.iloc[self.indexes]
        else :
            self.data = None
        if self.dataset.y is not None:
            self.y = self.dataset.y.iloc[self.indexes]
        else :
            self.y = None
        self.sub_model = {"model": None, "score": None}

        self.rules = None
        self.score = None

        self.rules_exp = None
        self.score_exp = None

        self.success = None

        self.y_train = []
        for i in range(len(self.dataset.X)):
            if i in self.indexes:
                self.y_train.append(1)
            else :
                self.y_train.append(0)

        self.indexes_from_map = None

        self.explain = {"Imported": None, "SHAP": None, "LIME": None}
        if self.atk.explain["Imported"] is not None:
            self.explain["Imported"] = self.atk.explain["Imported"].iloc[self.indexes]
        if self.atk.explain["SHAP"] is not None:
            self.explain["SHAP"] = self.atk.explain["SHAP"].iloc[self.indexes]
        if self.atk.explain["LIME"] is not None:
            self.explain["LIME"] = self.atk.explain["LIME"].iloc[self.indexes]

    def __str__(self) -> str:
        texte = ' '.join(("Potato:\n",
                    "------------------\n",
                    "      State:", self.stateToSring(), "\n",
                    "      Number of points:", str(len(self.indexes)), "\n",
                    "      Percentage of the dataset:", str(round(100*len(self.indexes)/len(self.dataset.X), 2))+"%", "\n",
                    "      Sub-model:", str(self.sub_model["model"].__class__.__name__))) 
        return texte
    
    def __len__(self) -> int:
        """
        The length of the potato.

        Returns
        -------
        int
            The length of the potato.
        """
        return len(self.indexes)
    
    def size(self) -> int:
        """
        Function that returns the shape of the potato.

        Returns
        -------
        tuple
            The shape of the potato.
        """
        return len(self.indexes)
    
    def stateToSring(self)-> str :
        """
        Returns the state of the potato

        Returns
        -------
        str
            The name of the state
        """
        if self.state == Potato.UNKNOWN : return "unknown"
        elif self.state == Potato.LASSO : return "lasso"
        elif self.state == Potato.SKR : return "skope ruled"
        elif self.state == Potato.REFINED_SKR : return "refined skope rules"
        elif self.state == Potato.REGION : return "region"
        elif self.state == Potato.JSON : return "json importation"
        else : raise ValueError("unknown state for a potato")

    def getIndexes(self) -> list:
        """
        Function that returns the indexes of the potato.

        Returns
        -------
        list
            The indexes of the potato.
        """
        return self.indexes
    
    # TODO : le nom de la méthode devrait être + explicite (cf. lasso)
    def setIndexes(self, indexes:list) -> None:
        """
        Function that sets the indexes of the potato.

        Parameters
        ----------
        indexes : list
            The new indexes of the potato.
        """
        self.indexes = indexes
        self.data = self.dataset.X.iloc[self.indexes]
        self.y = self.dataset.y.iloc[self.indexes]

        self.y_train = []
        for i in range(len(self.dataset.X)):
            if i in self.indexes:
                self.y_train.append(1)
            else :
                self.y_train.append(0)

        self.state = Potato.LASSO
        self.success = None

        self.explain = {"Imported": None, "SHAP": None, "LIME": None}
        if self.atk.explain["Imported"] is not None:
            self.explain["Imported"] = self.atk.explain["Imported"].iloc[self.indexes]
        if self.atk.explain["SHAP"] is not None:
            self.explain["SHAP"] = self.atk.explain["SHAP"].iloc[self.indexes]
        if self.atk.explain["LIME"] is not None:
            self.explain["LIME"] = self.atk.explain["LIME"].iloc[self.indexes]

    def setJsonPath(self, json_path:str) -> None:
        if json_path[-5:] != ".json":
            json_path += ".json"
        fileObject = open(json_path, "r")
        jsonContent = fileObject.read()
        self.indexes = JSON.loads(jsonContent)
        self.data = self.dataset.X.iloc[self.indexes]
        self.y = self.dataset.y.iloc[self.indexes]

        self.y_train = []
        for i in range(len(self.dataset.X)):
            if i in self.indexes:
                self.y_train.append(1)
            else :
                self.y_train.append(0)

        self.state = Potato.JSON
        self.success = None

        self.explain = {"Imported": None, "SHAP": None, "LIME": None}
        if self.atk.explain["Imported"] is not None:
            self.explain["Imported"] = self.atk.explain["Imported"].iloc[self.indexes]
        if self.atk.explain["SHAP"] is not None:
            self.explain["SHAP"] = self.atk.explain["SHAP"].iloc[self.indexes]
        if self.atk.explain["LIME"] is not None:
            self.explain["LIME"] = self.atk.explain["LIME"].iloc[self.indexes]


    def getVSdata(self) -> pd.DataFrame:
        """
        Function that returns the data of the potato.

        Returns
        -------
        pandas dataframe
            The data of the potato.
        """
        return self.data
    
    def getESdata(self, explanation="Imported") -> list:
        """
        Function that returns the data of the potato in the explanation space.

        Parameters
        ----------
        explanation : str
            The name of the explanation space.

        Returns
        -------
        pandas dataframe
            The data of the potato in the explanation space.
        """
        return self.explain[explanation]
    
    def getVSrules(self):
        """
        Function that returns the rules of the potato.

        Returns
        -------
        list
            The rules of the potato.
        """
        return self.rules
    
    def getESrules(self):
        """
        Function that returns the rules of the potato in the explanation space.

        Returns
        -------
        list
            The rules of the potato in the explanation space.
        """
        return self.rules_exp
    
    def getVSscore(self):
        """
        Function that returns the score of the potato.

        Returns
        -------
        tuple
            The score of the potato.
        """
        return self.score
    
    def getESscore(self):
        """
        Function that returns the score of the potato in the explanation space.

        Returns
        -------
        tuple
            The score of the potato in the explanation space.
        """
        return self.score_exp
    
    def applyRules(self, to_return:bool=False):
        """
        Function that applies the rules to the dataset, in order to create a new selection.

        Examples
        --------
        >>> from antakia import AntakIA, Dataset, Potato
        >>> X = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns = ["col1", "col2", "col3"], index = [1, 3, 4])
        >>> atk = AntakIA(Dataset(X, model))
        >>> potato = Potato(indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], atk)
        >>> potato.rules = [[0, "<=", "col1", "<=", 9], [3, "<=", "col2", "<=", 8]]
        >>> potato.applyRules()
        >>> potato.indexes
        [3, 4] # only the two last points respect the rules !

        """
        self.state = Potato.SKR
        """
        rules = self.rules
        df = self.dataset.X
        for i in range(len(rules)):
            regle1 = "df.loc[" + str(rules[i][0]) + rules[i][1] + "df['" + rules[i][2] + "']]"
            regle2 = "df.loc[" + "df['" + rules[i][2] + "']" + rules[i][3] + str(rules[i][4]) + "]"
            df = eval(regle1)
            df = eval(regle2)
        self.data = df
        self.setIndexes(df.index)
        """
        solo_features = list(set([self.rules[i][2] for i in range(len(self.rules))]))
        nombre_features_rules = []
        for i in range(len(solo_features)):
            nombre_features_rules.append([])
        for i in range(len(self.rules)):
            nombre_features_rules[solo_features.index(self.rules[i][2])].append(self.rules[i])

        nouvelle_tuile = self.atk.dataset.X.index
        for i in range(len(nombre_features_rules)):
            nouvelle_tuile_temp = []
            for j in range(len(nombre_features_rules[i])):
                X_temp = self.atk.dataset.X[
                    (self.atk.dataset.X[nombre_features_rules[i][j][2]] >= nombre_features_rules[i][j][0])
                    & (self.atk.dataset.X[nombre_features_rules[i][j][2]] <= nombre_features_rules[i][j][4])
                ].index
                nouvelle_tuile_temp = list(nouvelle_tuile_temp) + list(X_temp)
                nouvelle_tuile_temp = list(set(nouvelle_tuile_temp))
            nouvelle_tuile = [g for g in nouvelle_tuile if g in nouvelle_tuile_temp]

        if self.indexes_from_map is not None:
            nouvelle_tuile = [g for g in nouvelle_tuile if g in self.indexes_from_map]
        if to_return:
            return nouvelle_tuile
        self.setIndexes(nouvelle_tuile)

    def setIndexesFromMap(self, indexes:list) -> None:
        """
        Function that sets the indexes of the potato from a map.

        Parameters
        ----------
        indexes : list
            The new indexes of the potato.
        """
        self.indexes_from_map = indexes
        self.applyRules()

    def setSubModel(self, model) -> None:
        """
        Function that sets the sub-model of the potato.

        Parameters
        ----------
        model : model object
            The new sub-model of the potato.
        """
        self.sub_model["model"] = model
        model.fit(self.data, self.y)
        self.sub_model["score"] = model.score(self.data, self.y)

    def getSubModel(self):
        """
        Function that returns the sub-model of the potato.

        Returns
        -------
        dict
            The sub-model of the potato. Is the following format : {"model": model object, "score": score of the model}.
        """
        return self.sub_model
    
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
    
    def __error_message(self, message:str):
        print("AntakIA ERROR : " + message)

    def applySkope(self, explanation, p:float = 0.7, r:float = 0.7):
        """
        Function that applies the skope-rules algorithm to the dataset, in order to create a new selection.
        Must be connected to the AntakIA object (for the explanation space).

        Parameters
        ----------
        explanation : str
            The name of the explanation to use.
        p : float = 0.7
            The minimum precision of the rules.
        r : float = 0.7
            The minimum recall of the rules.
        """
        if self.atk.explain[explanation] is None:
            raise ValueError("You must provide a valid explanation space")
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
            feature_names=self.atk.explain[explanation].columns,
            random_state=42,
            n_estimators=5,
            recall_min=r,
            precision_min=p,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )
        skope_rules_clf_exp.fit(self.atk.explain[explanation], y_train)
        if skope_rules_clf.rules_ == [] or skope_rules_clf_exp.rules_ == []:
            self.rules, self.score_skope, self.rules_exp, self.score_skope_exp = None, None, None, None
            self.success = False
            self.__error_message("No rules found for this precision and recall")
        else :
            self.rules, self.score = self.__transform_rules(skope_rules_clf.rules_, self.dataset.X)
            self.rules_exp, self.score_exp = self.__transform_rules(skope_rules_clf_exp.rules_, self.atk.explain[explanation])
            self.checkForDuplicates()
            self.applyRules()
            self.success = True
            self.state = Potato.SKR

    def checkForDuplicates(self):
        """
        Function that checks if there are duplicates in the rules.
        A duplicate is a rule that has the same feature as another rule, but with a different threshold.
        """
        features = [self.rules[i][2] for i in range(len(self.rules))]
        features_alone = list(set(features))
        if len(features) == len(features_alone):
            return
        else :
            for feature in features:
                if features.count(feature) > 1:
                    a=0
                    for i in range(len(self.rules)):
                        min_feature = -10e99
                        max_feature = 10e99
                        if self.rules[i-a][2] == feature:
                            if self.rules[i-a][0] > min_feature:
                                min_feature = self.rules[i-a][0]
                            if self.rules[i-a][4] < max_feature:
                                max_feature = self.rules[i-a][4]
                            self.rules.pop(i-a)
                            a+=1
                    self.rules.append([min_feature, "<=", feature, "<=", max_feature])

        # same thing for the explanation space
        features = [self.rules_exp[i][2] for i in range(len(self.rules_exp))]
        features_alone = list(set(features))
        if len(features) == len(features_alone):
            return
        else :
            for feature in features:
                if features.count(feature) > 1:
                    a=0
                    for i in range(len(self.rules_exp)):
                        min_feature = -10e99
                        max_feature = 10e99
                        if self.rules_exp[i-a][2] == feature:
                            if self.rules_exp[i-a][0] > min_feature:
                                min_feature = self.rules_exp[i-a][0]
                            if self.rules_exp[i-a][4] < max_feature:
                                max_feature = self.rules_exp[i-a][4]
                            self.rules_exp.pop(i-a)
                            a+=1
                    self.rules_exp.append([min_feature, "<=", feature, "<=", max_feature])

    def pretty_print(self, table, ch1="-", ch2="|", ch3="+"):
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
        """
        if self.rules is None:
            print("No rules")
        else :
            self.pretty_print(self.rules, ch3 = '-', ch2=" ")

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
        rules = self.rules
        df = deepcopy(self.dataset.X)
        regle1 = "df.loc[" + str(rules[index][0]) + rules[index][1] + "df['" + rules[index][2] + "']]"
        regle2 = "df.loc[" + "df['" + rules[index][2] + "']" + rules[index][3] + str(rules[index][4]) + "]"
        df = eval(regle1)
        df = eval(regle2)
        return df
    
    def toJson(self):
        """
        Function that returns the potato in the form of a json file.

        Returns
        -------
        json
            The potato in the form of a json file.
        """
        return {"indexes": self.indexes, "state": self.state, "rules": self.rules, "score": self.score, "rules_exp": self.rules_exp, "score_exp": self.score_exp, "sub_model": self.sub_model, "success": self.success}
    
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
    potato.setIndexes(json["indexes"])
    potato.state = json["state"]
    potato.rules = json["rules"]
    potato.score = json["score"]
    potato.rules_exp = json["rules_exp"]
    potato.score_exp = json["score_exp"]
    potato.sub_model = json["sub_model"]
    potato.success = json["success"]
    return potato