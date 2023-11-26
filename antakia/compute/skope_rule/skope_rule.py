import math
import warnings

import numpy as np
import pandas as pd
from skrules import SkopeRules

from antakia.utils.variable import Variable, DataVariables
from antakia.data_handler.rules import Rule
import antakia.utils.utils as utils

import logging as logging
from antakia.utils.logging import conf_logger
logger = logging.getLogger(__name__)
conf_logger(logger)

def skope_rules(df_mask: pd.Series, base_space_df: pd.DataFrame, variables: DataVariables = None, precision: float = 0.7,
                recall: float = 0.7, random_state=42) -> (list, dict):
    """
    variables : list of Variables of the app
    df_indexes : list of (DataFrame) indexes for the points selected in the GUI
    base_space_df : the dataframe on which the rules will be computed / extracted. May be VS or ES values
    precision for SKR binary classifer : defaults to 0.7
    recall for SKR binary classifer : defaults to 0.7
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.info('skope_rules in')
        # We convert df_indexes in row_indexes
        y_train = df_mask.astype(int)
        if variables is None:
            variables = Variable.guess_variables(base_space_df)

        sk_classifier = SkopeRules(
            feature_names=variables.sym_list(),
            random_state=random_state,
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
            rules_list = Rule.combine_rule_list(rules_list)

        # We remove infinity in rules : we convert in simple rule if inf present
        # We had to wait for _combine_rule_list to proceed
        logger.info('skope_rules out')
        return rules_list, score_dict

    else:
        logger.info('skope_rules out')
        return [], {}
