"""
    Class Selection

"""
import logging

import pandas as pd
import numpy as np
from skrules import SkopeRules

import json as JSON
from copy import copy

from antakia.data import Variable, vars_to_string, vars_to_sym_list

from antakia.utils import confLogger

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()

class Rule():
    """ 
    A Rule is defined by a Variable, two operators and two range values
    variable : Variable
    min : float
    operator_min : int # index in operators
    operator_max : int # index in operators
    max : float
    """

    OPERATORS = ["<", "<=", "=",">=",">"]

    def __init__(self, min:float, operator_min:int, variable:Variable, operator_max: int, max:float):
        self.min = min
        self.operator_min = operator_min
        self.variable = variable
        self.operator_max = operator_max
        self.max = max

    def __str__(self) -> str:
        return f"{self.variable.symbol} {self.operators[self.operator_min]} {self.min} and {self.variable.symbol} {self.operators[self.operator_max]} {self.max}"

    @staticmethod
    def get_operator_index(operator:str) -> int:
        return Rule.operators.index(operator)
    

    def get_positives(self, X :pd.DataFrame) -> pd.DataFrame:
        """
        Returns the points of X that respect the rule
        """
        df = copy(X)
        left_condition = "df.loc[" + str(self.min) + OPERATORS[self.operator_min] + "df['" + self.variable.symbol + "']]"
        right_condition = "df.loc[" + "df['" + self.variable.symbol + "']" + OPERATORS[self.operator_max] + str(self.max) + "]"
        df = eval(left_condition)
        df = eval(right_condition)
        return df

class Rules():
    """
    rules : A list of Rule
    is_skope_rules : bool
    scores : dict {"precision":float, "recall":float, "f1":float}
    """

    def __init__(self, rules : list, is_skope_rules : bool, scores:float = None):
        self.rules = rules
        self.is_skope_rules = is_skope_rules
        self.scores = scores

    
    def handle_duplicate_rules(self):
        """
        Checks if there are duplicates in the rules.
        A duplicate is a rule that has the same feature as another rule, but with a different threshold.
        """
        variables = [rule.variable for rule in (self.rules)]
        unique_variables = list(set(variables))
        if len(variables) == len(variables):
            return # No duplicates
        else :
            for current_variable in variables:
                if variables.count(variable) > 1:
                    a=0
                    for i in range(len(self.rules)):
                        current_min = -10e99
                        current_max = 10e99
                        if self.rules[i-a].variable == current_variable:
                            if self.rules[i-a].min > current_min:
                                current_min = self.rules[i-a].min
                            if self.rules[i-a].max < current_max:
                                current_max = self.rules[i-a].max
                            self.rules.pop(i-a)
                            a+=1
                    self.rules.rules.append(Rule(current_min, Rule.get_operator_index("<="), current_variable, Rule.get_operator_index("<="), current_max))

    def __str__(self) -> str:
        text = ""
        for rule in self.rules:
            text += str(rule) + "\n"
        return text


def str_to_rules(skrules, X: pd.DataFrame, variables: list) -> Rules:
    rules = skrules.rules[0]
    scores = {
        "precision":round(rules[1][0],3),
        "recall":round(rules[1][1],3),
        "f1":round(rules[1][2],3),
        }

    token_list = rules[0].split(" and ")
    rule_list = []
    for i in range(len(token_list)):
        token_list[i] = token_list[i].split(" ") 
        variable = Variable.from_symbol(variables, token_list[i][0])
        operator_min = "<="
        operator_max = "<="
        if "<=" in token_list[i] or "<" in token_list[i]:
            min = round(float(min(df[l[2]])), 3)
            max = round(float(token_list[i][-1]), 3)
        elif ">=" in token_list[i] or ">" in token_list[i]:
            min = round(float(token_list[i][-1]), 3)
            max = round(float(max(X[l[2]])),3)
        rule_list.append(Rule(min, operator_min, variable, operator_max, max))
    return Rules(rule_list, True, scores)

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
    rules : list [Rules for VS, Rules for ES]

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
        self.rules = None

    def get_size(self) -> int:
        return len(self.indexes)

    def is_empty(self) -> bool:
        return self.get_size() == 0

    def get_type_as_str(self) -> str:
        if self.type == Selection.UNKNOWN:
            return "UNKNOWN"
        elif self.type == Selection.LASSO:
            return "LASSO"
        elif self.type == Selection.SELECTION:
            return "SELECTION"
        elif self.type == Selection.SKR:
            return "SKR"
        elif self.type == Selection.REFINED_SKR:
            return "REFINED_SKR"
        elif self.type == Selection.REGION:
            return "REGION"
        elif self.type == Selection.JSON:
            return "JSON"
        else:
            return "UNKNOWN"

    def __str__(self) -> str:
        text = f"Selection type: {self.get_type_as_str()}, size = {self.get_size()}"
        if self.rules is not None:
            if self.rules.count > 0:
                text += f", {self.rules.count} rules"
        else:
            text += ", no rules"
        return text

    @staticmethod
    def is_valid_type(type:int) -> bool:
        return type in [Selection.UNKNOWN, Selection.LASSO, Selection.SELECTION, Selection.SKR, Selection.REFINED_SKR, Selection.REGION, Selection.JSON] 



    def compute_skope_rules(self, X1 : pd.DataFrame, X2 : pd.DataFrame, y: pd.Series, variables :list, p:float = 0.7, r:float = 0.7):
        
        logger.debug(f"Entering compute_skope_rules X1={X1.shape}, X2={X2.shape} and y={y.shape}")

        self.rules = [self._compute_skope_rules_oneside(X1, y, variables, p, r),  self._compute_skope_rules_oneside(X2, y, variables, p, r)]
        self.type = Selection.SKR

    def _compute_skope_rules_oneside(self, X: pd.DataFrame, y: pd.Series, variables: list, p: float = 0.7, r: float = 0.7) -> Rules:
        """
        Computes the skope-rules algorithm to the selection
        # TODO : we need to allow the user to undo.

        Parameters
        ----------
        X : pd.DataFrame
        variables : list of X Variables
        p : float = 0.7
            The minimum precision of the rules.
        r : float = 0.7
            The minimum recall of the rules.

        Returns : a Rules object or None if no rules were found
        """
        
        y_train = np.zeros(y.size)
        y_train[self.indexes] = 1

        sk_classifier = SkopeRules(
            feature_names=vars_to_sym_list(variables),
            random_state=42,
            n_estimators=5,
            recall_min=r,
            precision_min=p,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )
        sk_classifier.fit(X, y_train)

        if sk_classifier.rules_ == []:
            logger.debug(f"No Skope rule found")
            return None
        else :
            rules = str_to_rules(sk_classifier.rules_, X, variables)
            logger.debug(f"We found these rules : \n{rules}")
            rules.handle_duplicate_rules()
            self.apply_rules()
            return rules


    def apply_rules(self, X: pd.DataFrame, rules: Rules):
        """
        Applies the rules to the selection
        """
        temp_indexes = []
        for rule in rules.rules:
            temp_indexes = temp_indexes.append(rule.get_positives(X).index.tolist())
        self.indexes = list(dict.fromkeys(temp_indexes)) # Remove duplicates

    
    # def reveal_intervals(self):
    #     features = [self.theVSRules[i][2] for i in range(len(self.rules))]
    #     features_alone = list(set(features))
    #     if len(features) == len(features_alone):
    #         return
    #     else :
    #         for feature in features:
    #             if features.count(feature) > 1:
    #                 a=0
    #                 for i in range(len(self.theVSRules)):
    #                     min_feature = -10e99
    #                     max_feature = 10e99
    #                     if self.theVSRules[i-a][2] == feature:
    #                         if self.theVSRules[i-a][0] > min_feature:
    #                             min_feature = self.rules[i-a][0]
    #                         if self.theVSRules[i-a][4] < max_feature:
    #                             max_feature = self.theVSRules[i-a][4]
    #                         self.theVSRules.pop(i-a)
    #                         a+=1
    #                 self.theVSRules.append([min_feature, "<=", feature, "<=", max_feature])


    #     features = [self.theESRules[i][2] for i in range(len(self.theESRules))]
    #     features_alone = list(set(features))
    #     if len(features) == len(features_alone):
    #         return
    #     else :
    #         for feature in features:
    #             if features.count(feature) > 1:
    #                 a=0
    #                 for i in range(len(self.theESRules)):
    #                     min_feature = -10e99
    #                     max_feature = 10e99
    #                     if self.theESRules[i-a][2] == feature:
    #                         if self.theESRules[i-a][0] > min_feature:
    #                             min_feature = self.theESRules[i-a][0]
    #                         if self.theESRules[i-a][4] < max_feature:
    #                             max_feature = self.theESRules[i-a][4]
    #                         self.theESRules.pop(i-a)
    #                         a+=1
    #                 self.theESRules.append([min_feature, "<=", feature, "<=", max_feature])
