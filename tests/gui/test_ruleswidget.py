import mock
import numpy as np
import pandas as pd

from antakia.data_handler.rules import Rule
from antakia.gui.ruleswidget import RuleWidget
from antakia.utils.variable import Variable


def test_rule_widget():
    var = Variable(0, 'var1', 'float')
    rule1 = Rule(None, None, var, '<', 10)
    rule2 = Rule(10, '<=', var, None, None)
    rule3 = Rule(10, '<=', var, '<', 40)
    var2 = Variable(0, 'var2', 'float')
    rule4 = Rule(10, '<=', var2, None, None)

    data = pd.DataFrame(np.arange(100).reshape((-1, 2)), columns=['var1', 'var2'])
    rules1 = [rule1, rule4]
    mask1 = rule1.rules_to_mask(rules1, data)
    rules2 = [rule2, rule4]
    mask2 = rule1.rules_to_mask(rules2, data)

    rw = RuleWidget(rule1, data, True, mask1, lambda x: None)

    # add tests

    rw.update(mask2, rule1)

    # add tests



