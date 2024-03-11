import numpy as np
import pandas as pd

from antakia_core.data_handler.rules import Rule, RuleSet
from antakia.gui.tabs.ruleswidget import RuleWidget
from antakia_core.utils.variable import Variable


def test_rule_widget():
    var = Variable(0, 'var1', 'float')
    rule1 = Rule(var, max=10, includes_max=False)  # None, None, var, '<', 10)
    rule2 = Rule(var, min=10, includes_min=True)  # 10, '<=', var, None, None)
    rule3 = Rule(var, min=10, includes_min=True, max=40, includes_max=False)  # 10, '<=', var, '<', 40)
    var2 = Variable(0, 'var2', 'float')
    rule4 = Rule(var2, min=10, includes_min=True)  # 10, '<=', var2, None, None)

    data = pd.DataFrame(np.arange(150).reshape((-1, 3)), columns=['var1', 'var2', 'y'])
    rules1 = RuleSet([rule1, rule4])
    mask1 = rules1.get_matching_mask(data)
    rules2 = RuleSet([rule2, rule4])
    mask2 = rules2.get_matching_mask(data)

    rw = RuleWidget(rule1, data.iloc[:, :2], data.iloc[:, 2], True, mask1, mask1, lambda x: None)

    # add tests

    rw.update(mask2, rule1)

    # add tests
