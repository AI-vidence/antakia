import logging

import pandas as pd
import numpy as np


from copy import copy
import math

from antakia.data import Variable, var_from_symbol
from antakia.utils import conf_logger
import antakia.config as config

logger = logging.getLogger(__name__)
conf_logger(logger)


class Rule():
    """ 
    A Rule is defined by a Variable, two operators and two range values (min, max)
    It may be of the form :
    - simple rules:
        1. variable < (or <=) max (then min is None and only operator_max is considered)
        2. variable > (or >=) min (then max is None and only operator_min is considered)
    - interval rules:
        3. min < (or <=) variable < (or <=) max, ie. variable element of [min, max]
        4. var < (or <=) min and var > (or >=) max, ie. variable not element of [min, max]

    Interval rules are "combined rules", ie. they were 2 rules when genrated by SKR

    The 'create_categorical_rule' method of the module makes it easier to instantiate a categorical rule: it only needs a variable anda  list of allowed categorical values

    variable : Variable
    min : float
    operator_min : int # index in operators
    operator_max : int # index in operators
    max : float
    cat_values: list # list of values for categorical variables
    """

    OPERATORS = ["<", "<=", "=", ">=", ">"]

    def __init__(self, min: float, operator_min, variable: Variable, operator_max, max: float, cat_values: list = None):
        self.min = min
        # operarors may be int or str or None (simple rule).
        if isinstance(operator_min, int):
            self.operator_min = operator_min
        elif isinstance(operator_min, str):
            self.operator_min = self.OPERATORS.index(operator_min)
        else:
            self.operator_min = None
        self.variable = variable
        if isinstance(operator_max, int):
            self.operator_max = operator_max
        elif isinstance(operator_max, str):
            self.operator_max = self.OPERATORS.index(operator_max)
        else:
            self.operator_max = None
        self.max = max
        self.cat_values = cat_values

    def is_categorical_rule(self) -> bool:
        return self.cat_values is not None and len(self.cat_values)

    def is_interval_rule(self) -> bool:
        return self.min is not None and self.max is not None
    
    def is_inner_interval_rule(self) -> bool:
        return ((self.operator_min == 0 or self.operator_min == 1) and (self.operator_max == 0 or self.operator_max == 1)) or ((self.operator_min == 3 or self.operator_min == 4) and (self.operator_max == 3 or self.operator_max == 4))

    def __repr__(self) -> str:
        txt = ""
        if self.is_categorical_rule():
            txt =  f"{self.variable.symbol} \u2208  \u27E6"
            for i, cat_value in enumerate(self.cat_values):
                txt += cat_value
                if i < len(self.cat_values) - 1:
                    txt += ", "
            txt += "\u27E7"
        else:
            if self.min is None:
                # Rule type 1
                txt = self.variable.symbol + " "
                txt += "<" if self.operator_min == 0 else "\u2264" # lesser (or equal) than
                txt += " " + str(self.max)
            elif self.max is None:
                # Rule type 2
                txt = self.variable.symbol + " "
                txt += ">" if self.operator_min == 0 else "\u2265" # geater (or equal) than
                txt += " " + str(self.min)
            else:
                if self.is_inner_interval_rule():
                    # Rule type 3 : the rule is of the form : variable included in [min, max] interval, or min < variable < max
                    if config.USE_INTERVALS_FOR_RULES:
                        txt = self.variable.symbol + " \u2208 " # element of
                        txt += "[" if self.operator_min==0 else "\u27E6" # exclusive or inclusive left square bracket
                        txt += f"{self.min}, {self.max}"
                        txt += "]" if self.operator_max==0 else "\u27E7" # exclusive or inclusive right square bracket
                    else:
                        txt = str(self.min) + " "
                        txt += "<" if self.operator_min == 0 else "\u2264" # lesser (or equal) than
                        txt += " " + self.variable.symbol + " "
                        txt += "<" if self.operator_max == 0 else "\u2264" # lesser (or equal) than
                        txt += " " + str(self.max)
                else:
                    # Rule type 4 : the rule is of the form : variable not included in [min, max] interval or variable < min and variable > max
                    if config.USE_INTERVALS_FOR_RULES:
                        txt = self.variable.symbol + "\u2209" # not an element of
                        txt += "[" if self.operator_min==0 else "\u27E6" # exclusive or inclusive left square bracket
                        txt += f"{self.min}, {self.max}"
                        txt += "]" if self.operator_max==0 else "\u27E7" # exclusive or inclusive right square bracket
                    else :
                        txt = self.variable.symbol + " "
                        txt += "<" if self.operator_min == 0 else "\u2264" # lesser (or equal) than
                        txt += " " + self.variable.symbol + " "
                        txt += ">" if self.operator_max == 0 else "\u2265" # greater (or equal) than
                        txt += " " + str(self.max)
        return txt

    def get_matching_indexes(self, X: pd.DataFrame) -> list:
        """
        Returns a list of indexes of the points of X that satisfy the rule
        """
        # We're going to modify X, so we make a copy
        df = copy(X)

        query_var = "df['" + self.variable.symbol + "']"

        if self.is_categorical_rule():
            query = f"df[ + {query_var} .isin({self.cat_values})]"
            df = eval(query)
        else:
            if not self.is_interval_rule():
                if self.min is None:
                    # Rule type 1
                    query = "df.loc[" + query_var + " "
                    query += "<" if self.operator_max == 0 else "<="
                    query += " " + str(self.max) + "]"    
                if self.max is None:
                    # Rule type 2
                    query = "df.loc[" + " " + query_var + " "
                    query += ">" if self.operator_min == 0 else ">="
                    query += " " + str(self.min) + "]"
                logger.debug(f"Query : {query}")
                df = eval(query)
            else:
                # We have an interval rule -> 2 queries
                if self.is_inner_interval_rule():
                    # Rule type 3 
                    min_query = "df.loc[" + query_var
                    min_query += ">" if self.operator_min == 0 else ">="
                    min_query += " " + str(self.min) + "]"
                    df = eval(min_query)
                    max_query = "df.loc[" + query_var
                    max_query += "<" if self.operator_max == 0 else "<="
                    max_query += " " + str(self.max) + "]"
                    df = eval(max_query)
                else: 
                    # Rule type 4
                    min_query = "df.loc[" + query_var
                    min_query += "<" if self.operator_min == 0 else "<="
                    min_query += " " + str(self.min) + "]"
                    df = eval(min_query)
                    max_query = "df.loc[" + query_var
                    max_query += ">" if self.operator_max == 0 else ">="
                    max_query += " " + str(self.max) + "]"
                    df = eval(max_query)
        return df.index.tolist() if df is not None else []


    @staticmethod
    def rules_to_indexes(rules_list: list, base_space_df: pd.DataFrame) -> list:
        """"
        Returns a list of indexes of base_space_df that comply with the rules
        We assume rules_list contains rules, not list of rules
        """

        if rules_list is None or len(rules_list) == 0:
            return []

        indexes = []
        for rule in rules_list:
            matching_indexes = rule.get_matching_indexes(base_space_df)
            for index in matching_indexes:
                if index not in indexes:
                    indexes.append(index)

        return indexes
    

    @staticmethod
    def _combine_rule_pair(rule1, rule2) -> list:
        """ 
        Try to combine 2 rules on the same variable into one. Possible when their intervals overlap
        Returns a list of 1 combined rule or the 2 original rules when no comnbination are possible
        Only used by the _combine_rule_list method
        """
        if rule1.variable == rule2.variable:    
            if rule1.min < rule2.max or rule2.min < rule1.max:
                # We have an interval, we may combine the rules
                min_ = max(rule1.min, rule2.min)
                max_ = min(rule1.max, rule2.max)
                rule3 = [Rule(min_, "<=", rule1.variable, "<=", max_)]
                return rule3
            else:
                # No overlap, no combination possible
                return [rule1, rule2]
        else:
            #  Not the same variable, no combination possible
            return [rule1, rule2]
        
    @staticmethod
    def _combine_rule_list(rule_list: list):
        """
        Try to combine all rules of the list into a smaller list of rules
        """
        variables_checked = []

        for i in range(len(rule_list)):
            # We might be out of range (since we pop), so we check:
            if i < len(rule_list):
                rule_i = rule_list[i]
                variable = rule_i.variable
                if variable not in variables_checked:
                    # Let's see if we find another rule with the same variable
                    for j in range(i+1, len(rule_list)):
                        if j < len(rule_list):
                            rule_j = rule_list[j]
                            if rule_j.variable == variable:
                                # We found another rule with the same variable, let's try to combine them
                                combination_result_list = Rule._combine_rule_pair(rule_i, rule_j)
                                if len(combination_result_list) == 1:
                                    # We managed to combine the rules
                                    # 1) we remove rule_j from tule_list :
                                    rule_list.pop(j)
                                    # We replace rule_i by the combined rule:
                                    rule_list[i] = combination_result_list[0]
                                    # we need to decrement j because we removed an element from the list
                                    j-=1 
            variables_checked.append(variable)


    @staticmethod
    def _extract_rules(skrules, X: pd.DataFrame, variables: list) -> list:
        """
        Transforms a string into a list of rules
        """
        tokens = skrules[0]
        score_dict = {
            "precision": round(tokens[1][0], 3),
            "recall": round(tokens[1][1], 3),
            "f1": round(tokens[1][2], 3),
        }

        tokens = tokens[0].split(" and ")
        
        rule_list = []
        for i in range(len(tokens)):
            tokens[i] = tokens[i].split(" ")
            variable = var_from_symbol(variables, tokens[i][0])

            if "<=" in tokens[i] or "<" in tokens[i]:
                min = - math.inf
                max = round(float(tokens[i][2]), 3)

            elif ">=" in tokens[i] or ">" in tokens[i]:
                min = round(float(tokens[i][2]), 3)
                max = math.inf

            temp_rule = Rule(min, "<=", variable, "<=", max)
            rule_list.append(temp_rule)

        return rule_list, score_dict


    @staticmethod
    def rows_to_indexes(X: pd.DataFrame, rows_list: list) -> list:
        """
        Converts DataFrame row numbers to Index numbers
        """
        return [X.index[row_number] for row_number in rows_list]
    
    @staticmethod
    def indexes_to_rows(X: pd.DataFrame, indexes_list: list) -> list:
        """
        Converts DataFrame Index numbers to row numbers
        """
        row_ids_list = []
        for index in indexes_list:
            if index in X.index:
                row_ids_list.append(X.index.get_loc(index))
            else:
                raise KeyError(f"Index {index} not found in DataFrame")

        return row_ids_list

    @staticmethod
    def rules_to_dict_list(rules_list: list) -> dict:
        """""
        Returns a dict rep compatible with the v.DataTable widget
        """

        if rules_list is None or len(rules_list) == 0:
            return []

        def rule_to_dict(rule: Rule) -> dict:
            temp = dict()
            temp['Variable'] = rule.variable.symbol
            temp['Unit'] = rule.variable.unit
            temp['Desc'] = rule.variable.descr
            temp['Critical'] = rule.variable.critical
            temp['Rule'] = rule.__repr__()
            return temp
        
        return [rule_to_dict(rule) for rule in rules_list]
    
    @staticmethod
    def regions_to_rules(region_list: list) -> list:
        """
        Transforms a list of regions into a list of rules
        """
        rules = []
        for region in region_list:
            region_rule_list = region['rules']
            for rule in region_rule_list:
                rules.append(rule)   
        return rules
    
    @staticmethod
    def resulting_df(X: pd.DataFrame, rules_list:list) -> pd.DataFrame:
        """
        Returns X minus rules_list(X)
        """
        return X.drop(Rule.rules_to_indexes(rules_list, X))
    
    @staticmethod
    def multi_rules_to_string(rules_list: list) -> str:
        """
        Returns a string representation of a list of rules
        """
        if rules_list is None or len(rules_list) == 0:
            return ""
        else:
            return " and ".join([rule.__repr__() for rule in rules_list])
    
    
def create_categorical_rule(variable: Variable, cat_values: list) -> Rule:
    """
    Creates a categorical rule for the given variable and list of values
    """
    return Rule(None, None, variable, None, None, cat_values)
