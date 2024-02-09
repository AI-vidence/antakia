from __future__ import annotations

import logging
from typing import List, Dict
from numbers import Number

import pandas as pd

import os
import math

from antakia.utils.utils import boolean_mask, mask_to_index
from antakia.utils.variable import Variable, DataVariables
from antakia.utils.logging import conf_logger


class Rule:
    """ 
    A Rule is defined by a Variable, two operators and two range values (min, max)
    It may be of the form :
    - simple rules:
        1. variable < (or <=) max (then min is None and only operator_max is considered)
        2. variable > (or >=) min (then max is None and only operator_min is considered)
        3.1 variable = x - encoded as var in [x,x]
    - interval rules:
        3. min < (or <=) variable < (or <=) max, ie. variable element of [min, max]
        4. var < (or <=) min or var > (or >=) max, ie. variable not element of [min, max]

    Interval rules are "combined rules", ie. they were 2 rules when genrated by SKR

    The 'create_categorical_rule' method of the module makes it easier to instantiate a categorical rule: it only needs a variable and a list of allowed categorical values

    variable : Variable
    min : float
    operator_min : int # index in operators
    operator_max : int # index in operators
    max : float
    cat_values: list # list of values for categorical variables
    """

    OPERATORS = ["<", "<=", "=", ">=", ">"]
    PYTHON_OPERATORS = ['__lt__', '__le__', '__eq__', '__ge__', '__gt__']
    PRETTY_OPERATORS = ["<", "\u2264", "=", "\u2265", ">"]
    PRETTY_BRAKET = ["[", "\u27E6", "", "\u27E7", "]"]

    def __init__(self, variable_min: float | None, operator_min: str | int | None, variable: Variable,
                 operator_max: str | int | None, variable_max: float | None, cat_values: list[str] | None = None):
        self.variable = variable
        self.cat_values = cat_values
        assert variable_min is None or isinstance(variable_min, Number)
        assert variable_max is None or isinstance(variable_max, Number)

        # operators may be int or str or None (simple rule).
        if isinstance(operator_min, int):
            self.operator_min = operator_min
            self.min = variable_min
        elif isinstance(operator_min, str):
            self.operator_min = self.OPERATORS.index(operator_min)
            self.min = variable_min
        if variable_min == -math.inf or variable_min is None:
            self.operator_min = 2  # convention -> invarient par inversion
            self.min = -math.inf

        if isinstance(operator_max, int):
            self.operator_max = operator_max
            self.max = variable_max
        elif isinstance(operator_max, str):
            self.operator_max = self.OPERATORS.index(operator_max)
            self.max = variable_max
        if variable_max == math.inf or variable_max is None:
            self.operator_max = 2  # convention -> invarient par inversion
            self.max = math.inf

        self.clean()

        # todo handle double rule min<var>max ans min>var<max

    def _same_direction(self):
        return min(1, max(-1, (self.operator_max - 2) * (self.operator_min - 2)))

    def _invert(self):
        max_, min_ = self.max, self.min
        if max_ == math.inf:
            self.min = -math.inf
        else:
            self.min = max_
        if min_ == -math.inf:
            self.max = math.inf
        else:
            self.max = min_
        self.operator_min, self.operator_max = 4 - self.operator_max, 4 - self.operator_min

    def rectify(self):
        if self._same_direction() == 0:
            if (self.operator_min - 2) + (self.operator_max - 2) > 0:
                self._invert()
        else:
            if self.min > self.max:
                self._invert()

    def clean(self):
        self.rectify()

        if self.min in (math.inf, -math.inf):
            self.operator_min = 2
        if self.max in (math.inf, -math.inf):
            self.operator_max = 2

        if self.operator_max > 2 and self.operator_min <= 2:
            if self.max > self.min:
                self.min = self.max
                self.operator_min = 4 - self.operator_max
            if self.max == self.min:
                self.operator_min = min(self.operator_min, 4 - self.operator_max)
            self.operator_max = 2
            self.max = math.inf
        elif self.operator_min > 2 and self.operator_max <= 2:
            if self.max > self.min:
                self.max = self.min
                self.operator_max = 4 - self.operator_min
            if self.max == self.min:
                self.operator_max = min(self.operator_max, 4 - self.operator_min)
            self.operator_min = 2
            self.min = -math.inf

    def __eq__(self, other):
        if self.rule_type != other.rule_type or self.variable != other.variable:
            return False
        if self.rule_type == 0:
            return (
                (self.cat_values == other.cat_values)
            )
        if self.rule_type == 1:
            return (
                    (self.max == other.max) &
                    (self.operator_max == other.operator_max)
            )
        if self.rule_type == 2:
            return (
                    (self.min == other.min) &
                    (self.operator_min == other.operator_min)
            )
        return (
                (self.min == other.min) &
                (self.max == other.max) &
                (self.operator_min == other.operator_min) &
                (self.operator_max == other.operator_max) &
                (self.variable == other.variable)
        )

    @property
    def rule_type(self):
        if self.is_categorical_rule:
            return 0
        if self._same_direction() == 0:
            if self.min == -math.inf:
                return 1
            if self.max == math.inf:
                return 2
        if self.max == math.inf and self.min == -math.inf:
            return -1

        if self.operator_max < 2 and self.operator_min < 2:
            return 3
        if self.operator_min > 2 and self.operator_max > 2:
            return 4
        return 5

    @property
    def is_categorical_rule(self) -> bool:
        return self.cat_values is not None and len(self.cat_values)

    @property
    def is_interval_rule(self) -> bool:
        return self.rule_type > 2

    @property
    def is_inner_interval_rule(self) -> bool:
        return self.rule_type == 3

    @property
    def include_equals(self):
        if self.rule_type == 0:
            return True
        if self.rule_type <= 2:
            return self.operator_min in (1, 3) or self.operator_max in (1, 3)
        return abs(self.operator_min - 2) == 1 and abs(self.operator_max - 2) == 1

    def __repr__(self) -> str:
        if self.is_categorical_rule:
            txt = f"{self.variable.display_name} \u2208  \u27E6"
            txt += ', '.join(self.cat_values)
            txt += "\u27E7"
            return txt
        if self.rule_type == 1:
            # Rule type 1
            txt = f"{self.variable.display_name} {self.PRETTY_OPERATORS[self.operator_max]} {self.max}"
            return txt
        if self.rule_type == 2:
            # Rule type 2
            txt = f"{self.variable.display_name} {self.PRETTY_OPERATORS[4 - self.operator_min]} {self.min}"
            return txt
        if self.rule_type == 3:
            # Rule type 3 : the rule is of the form : variable included in [min, max] interval, or min < variable < max
            if os.environ.get("USE_INTERVALS_FOR_RULES"):
                txt = f"{self.variable.display_name} \u2208 {self.PRETTY_BRAKET[self.operator_min]} {self.min},"
                txt += f" {self.max} {self.PRETTY_BRAKET[4 - self.operator_max]}"  # element of
                return txt
            txt = f'{self.min} {self.PRETTY_OPERATORS[self.operator_min]} {self.variable.display_name} '
            txt += f'{self.PRETTY_OPERATORS[self.operator_max]} {self.max}'
            return txt
        # Rule type 4 : the rule is of the form : variable not included in [min, max] interval or variable < min and variable > max
        if os.environ.get("USE_INTERVALS_FOR_RULES"):
            txt = f"{self.variable.display_name} \u2209 {self.PRETTY_BRAKET[self.operator_min]} {self.min},"
            txt += f" {self.max} {self.PRETTY_BRAKET[4 - self.operator_max]}"  # element of
            return txt
        else:
            txt = f'{self.variable.display_name} {self.PRETTY_OPERATORS[4 - self.operator_min]} {self.min} or '
            txt += f'{self.variable.display_name} {self.PRETTY_OPERATORS[self.operator_max]} {self.max}'
        return txt

    def get_matching_mask(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns a mask of indices matching the rule
        """
        # We're going to modify X, so we make a copy
        col = X.loc[:, self.variable.column_name]

        if self.is_categorical_rule:
            return col.isin(self.cat_values)
        if self.rule_type == 1:
            return getattr(col, self.PYTHON_OPERATORS[self.operator_max])(self.max)
        if self.rule_type == 2:
            return getattr(col, self.PYTHON_OPERATORS[4 - self.operator_min])(self.min)
        max_ = getattr(col, self.PYTHON_OPERATORS[self.operator_max])(self.max)
        min_ = getattr(col, self.PYTHON_OPERATORS[4 - self.operator_min])(self.min)
        if self.rule_type == 3:
            # Rule type 3
            return max_ & min_
        else:
            # Rule type 4
            return max_ | min_

    def combine(self, rule: Rule) -> Rule | None:
        """ 
        Try to combine 2 rules on the same variable into one. Possible when their intervals overlap
        Returns a list of 1 combined rule or the 2 original rules when no comnbination are possible
        Only used by the _combine_rule_list method
        """
        if self.variable == rule.variable:
            if self.rule_type == 4 or rule.rule_type == 4:
                # combine with type 4 not yet handled
                return None
            if self.min == rule.max:
                if self.operator_min == 1 and rule.operator_max == 1:
                    return Rule(self.min, 1, self.variable, 1, self.min)
                else:
                    return None
            if rule.min == self.max:
                if self.operator_max == 1 and rule.operator_min == 1:
                    return Rule(self.max, 1, self.variable, 1, self.max)
                else:
                    return None
            if self.min <= rule.max and rule.min <= self.max:
                # We have an interval, we may combine the rules
                if self.min == rule.min:
                    min_ = self.min
                    if self.operator_min is None:
                        min_op = rule.operator_min
                    elif rule.operator_min is None:
                        min_op = self.operator_min
                    else:
                        min_op = min(rule.operator_min, self.operator_min)
                elif self.min < rule.min:
                    min_ = rule.min
                    min_op = rule.operator_min
                else:
                    min_ = self.min
                    min_op = self.operator_min
                if self.max == rule.max:
                    max_ = self.max
                    if self.operator_max is None:
                        max_op = rule.operator_max
                    elif rule.operator_max is None:
                        max_op = self.operator_max
                    else:
                        max_op = min(rule.operator_max, self.operator_max)
                elif self.max < rule.max:
                    max_ = self.max
                    max_op = self.operator_max
                else:
                    max_ = rule.max
                    max_op = rule.operator_max
                return Rule(min_, min_op, self.variable, max_op, max_)
            else:
                # No overlap, no combination possible
                return None
        else:
            #  Not the same variable, no combination possible
            return None

    def to_dict(self) -> dict:
        return {
            'Variable': self.variable.display_name,
            'Unit': self.variable.unit,
            'Desc': self.variable.descr,
            'Critical': self.variable.critical,
            'Rule': self.__repr__()
        }


class RuleSet:
    """
    set of rules
    """

    def __init__(self, rules: list[Rule] | RuleSet | None = None):

        if isinstance(rules, RuleSet):
            rules = rules.rules
        if rules:
            self.rules = rules
        else:
            self.rules = []

    def append(self, value: Rule):
        """
        add a new rule
        Parameters
        ----------
        value

        Returns
        -------

        """
        self.rules.append(value)

    def pop(self, index: int = None) -> Rule:
        """
        remove and return index rule
        Parameters
        ----------
        index

        Returns
        -------

        """
        return self.rules.pop(index)

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        if not self.rules:
            return ""
        return " and ".join([rule.__repr__() for rule in self.rules])

    def to_dict(self):
        if not self.rules:
            return []
        return [rule.to_dict() for rule in self.rules]

    def copy(self):
        return RuleSet(self.rules.copy())

    def set(self, new_rule: Rule):
        """
        edit a rule in the ruleset
        Parameters
        ----------
        new_rule

        Returns
        -------

        """
        for rule in self.rules:
            if rule.variable == new_rule.variable:
                self.rules[self.rules.index(rule)] = new_rule
                return
        # append if no rule on the variable
        self.append(new_rule)

    def get_matching_mask(self, X: pd.DataFrame) -> pd.Series:
        """
        get the mask of samples validating the rule
        Parameters
        ----------
        X: dataset to get the mask from

        Returns
        -------

        """
        res = boolean_mask(X, True)
        if self.rules is not None:
            for rule in self.rules:
                res &= rule.get_matching_mask(X)
        return res

    def get_matching_indexes(self, X):
        """
        get the list indexes of X validating the rule
        Parameters
        ----------
        X

        Returns
        -------

        """
        res = self.get_matching_mask(X)
        return mask_to_index(res)

    @classmethod
    def sk_rules_to_rule_set(cls, skrules, variables: DataVariables):
        """
        transform skope rules to a RuleSet
        Parameters
        ----------
        skrules
        variables

        Returns
        -------

        """
        tokens = skrules[0]
        precision = tokens[1][0]
        recall = tokens[1][1]
        f1 = precision * recall * 2 / (precision + recall)
        score_dict = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

        tokens = tokens[0].split(" and ")

        rule_list = RuleSet()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].split(" ")
            variable = variables.get_var(tokens[i][0])

            if "<=" in tokens[i] or "<" in tokens[i]:
                min = - math.inf
                max = round(float(tokens[i][2]), 3)

            elif ">=" in tokens[i] or ">" in tokens[i]:
                min = round(float(tokens[i][2]), 3)
                max = math.inf
            else:
                raise ValueError('Rule not recognized')

            temp_rule = Rule(min, "<=", variable, "<=", max)
            rule_list.append(temp_rule)

        return rule_list, score_dict

    @staticmethod
    def combine_rules_var(rule_list: list[Rule]) -> list[Rule]:
        """
        reduce rule list
        all rules should share the same variable
        Parameters
        ----------
        rule_list

        Returns
        -------

        """
        rule_list = rule_list[:]
        i = 0
        while i < len(rule_list):
            j = i + 1
            while j < len(rule_list):
                combined_rule = rule_list[i].combine(rule_list[j])
                if combined_rule is not None:
                    rule_list[i] = combined_rule
                    rule_list.pop(j)
                else:
                    j += 1
            i += 1
        return rule_list

    def combine(self) -> None:
        """
        Try to combine all rules of the list into a smaller list of rules
        """
        rules_per_var = {}
        for rule in self.rules:
            if rule.variable not in rules_per_var:
                rules_per_var[rule.variable] = [rule]
            else:
                rules_per_var[rule.variable].append(rule)

        new_rules = []
        for rules in rules_per_var.values():
            if len(rules) == 1:
                new_rules.append(rules[0])
            else:
                combined_rules = self.combine_rules_var(rules)
                new_rules.extend(combined_rules)
        self.rules = new_rules

    def find_rule(self, var: Variable) -> Rule | None:
        """
        find a rule on the variable
        Parameters
        ----------
        var

        Returns
        -------

        """
        for rule in self.rules:
            if rule.variable.column_name == var.column_name:
                return rule
