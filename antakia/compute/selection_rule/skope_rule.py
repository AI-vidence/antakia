import math
import warnings

import numpy as np
import pandas as pd
from skrules import SkopeRules

from antakia.data import Variable
from antakia.rules import Rule


def skope_rules(df_indexes: list, base_space_df: pd.DataFrame, variables: list = None, precision: float = 0.7,
                recall: float = 0.7) -> list:
    """
    variables : list of Variables of the app
    df_indexes : list of (DataFrame) indexes for the points selected in the GUI
    base_space_df : the dataframe on which the rules will be computed / extracted. May be VS or ES values
    precision for SKR binary classifer : defaults to 0.7
    recall for SKR binary classifer : defaults to 0.7
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_train = np.zeros(len(base_space_df))
        y_train[df_indexes] = 1  # our target

        if variables is None:
            variables = Variable.guess_variables(base_space_df)

        sk_classifier = SkopeRules(
            feature_names=Variable.vars_to_sym_list(variables),
            random_state=42,
            n_estimators=5,
            recall_min=recall,
            precision_min=precision,
            max_depth_duplication=0,
            max_samples=1.0,
            max_depth=3,
        )

        sk_classifier.fit(base_space_df, y_train)

    if sk_classifier.rules_ != []:
        rules_list, score_dict = Rule._extract_rules(sk_classifier.rules_, base_space_df, variables)

        if len(rules_list) >= 0:
            Rule._combine_rule_list(rules_list)

        # We remove infinity in rules : we convert in simple rule if inf present
        # We had to wait for _combine_rule_list to proceed
        for rule in rules_list:
            if rule.min == -math.inf:
                rule.min = None
                rule.operator_min = None
            if rule.max == math.inf:
                rule.max = None
                rule.operator_max = None
        return rules_list, score_dict

    else:
        return [], {}
