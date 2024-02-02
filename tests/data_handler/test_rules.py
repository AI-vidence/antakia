import numpy as np
import pandas as pd

from antakia.data_handler.rules import Rule, Variable


def test_type_1():
    var = Variable(0, 'type1', 'float')
    rule1_1 = Rule(None, None, var, '<', 10)
    rule1_2 = Rule(None, 2, var, '<=', 10)
    rule1_3 = Rule(None, None, var, '<', 20)
    rule1_4 = Rule(20, '>', var, 2, None)
    rule1_5 = Rule(10, '>=', var, None, None)
    rule1_6 = Rule(10, '>=', var, '<', 10)
    rule1_7 = Rule(10, '>', var, '<=', 10)
    rule1_8 = Rule(10, '>', var, '<=', 20)
    rule1_9 = Rule(20, '>', var, '<=', 10)

    assert rule1_4 == rule1_3
    assert rule1_5 == rule1_2
    assert repr(rule1_6) == 'type1 < 10'
    assert repr(rule1_7) == 'type1 < 10'
    assert repr(rule1_8) == 'type1 < 10'
    assert repr(rule1_9) == 'type1 ≤ 10'

    assert rule1_1.rule_type == 1
    assert rule1_2.rule_type == 1
    assert rule1_3.rule_type == 1

    assert not rule1_1.is_categorical_rule
    assert not rule1_1.is_interval_rule
    assert not rule1_1.is_inner_interval_rule

    assert repr(rule1_1) == 'type1 < 10'
    assert repr(rule1_2) == 'type1 ≤ 10'
    assert repr(rule1_3) == 'type1 < 20'

    data = pd.DataFrame(np.arange(30).reshape((-1, 1)), columns=['type1'])
    assert rule1_1.get_matching_mask(data).sum() == 10
    assert rule1_2.get_matching_mask(data).sum() == 11
    assert rule1_3.get_matching_mask(data).sum() == 20

    assert rule1_1.combine(rule1_2) == rule1_1
    assert rule1_1.combine(rule1_3) == rule1_1

    r1 = Rule(rule1_1.min, rule1_1.operator_min, rule1_1.variable, rule1_1.operator_max, rule1_1.max)
    assert r1 == rule1_1


def test_type_2():
    var = Variable(0, 'type2', 'float')
    rule2_1 = Rule(10, '<', var, None, None)
    rule2_2 = Rule(10, '<=', var, None, None)
    rule2_3 = Rule(20, '<', var, None, None)
    rule2_4 = Rule(None, 2, var, '>', 20)
    rule2_5 = Rule(None, 2, var, '>=', 10)
    rule2_6 = Rule(10, '<=', var, '>', 10)
    rule2_7 = Rule(10, '<', var, '>=', 10)
    rule2_8 = Rule(10, '<', var, '>=', 20)
    rule2_9 = Rule(20, '<', var, '>=', 10)

    assert rule2_4 == rule2_3
    assert rule2_5 == rule2_2
    assert repr(rule2_6) == 'type2 > 10'
    assert repr(rule2_7) == 'type2 > 10'
    assert repr(rule2_8) == 'type2 ≥ 20'
    assert repr(rule2_9) == 'type2 > 20'

    assert rule2_1.rule_type == 2
    assert rule2_2.rule_type == 2
    assert rule2_3.rule_type == 2

    assert not rule2_1.is_categorical_rule
    assert not rule2_1.is_interval_rule
    assert not rule2_1.is_inner_interval_rule

    assert repr(rule2_1) == 'type2 > 10'
    assert repr(rule2_2) == 'type2 ≥ 10'
    assert repr(rule2_3) == 'type2 > 20'

    data = pd.DataFrame(np.arange(30).reshape((-1, 1)), columns=['type2'])
    assert rule2_1.get_matching_mask(data).sum() == 19
    assert rule2_2.get_matching_mask(data).sum() == 20
    assert rule2_3.get_matching_mask(data).sum() == 9

    assert rule2_1.combine(rule2_2) == rule2_1
    assert rule2_1.combine(rule2_3) == rule2_3
    r1 = Rule(rule2_1.min, rule2_1.operator_min, rule2_1.variable, rule2_1.operator_max, rule2_1.max)
    assert r1 == rule2_1


def test_type_3():
    var = Variable(0, 'type3', 'float')
    rule3_1 = Rule(10, '<', var, '<', 40)
    rule3_2 = Rule(10, '<=', var, '<=', 40)
    rule3_3 = Rule(20, '<', var, '<', 30)
    rule3_4 = Rule(30, '>', var, '>', 20)
    rule3_5 = Rule(40, '>=', var, '>=', 10)

    assert rule3_4 == rule3_3
    assert rule3_5 == rule3_2

    assert rule3_1.rule_type == 3
    assert rule3_2.rule_type == 3
    assert rule3_3.rule_type == 3

    assert not rule3_1.is_categorical_rule
    assert rule3_1.is_interval_rule
    assert rule3_1.is_inner_interval_rule

    assert repr(rule3_1) == '10 < type3 < 40'
    assert repr(rule3_2) == '10 ≤ type3 ≤ 40'
    assert repr(rule3_3) == '20 < type3 < 30'

    data = pd.DataFrame(np.arange(50).reshape((-1, 1)), columns=['type3'])
    assert rule3_1.get_matching_mask(data).sum() == 29
    assert rule3_2.get_matching_mask(data).sum() == 31
    assert rule3_3.get_matching_mask(data).sum() == 9

    assert rule3_1.combine(rule3_2) == rule3_1
    assert rule3_1.combine(rule3_3) == rule3_3
    r1 = Rule(rule3_1.min, rule3_1.operator_min, rule3_1.variable, rule3_1.operator_max, rule3_1.max)
    assert r1 == rule3_1


def test_type_4():
    var = Variable(0, 'type4', 'float')
    rule4_1 = Rule(10, '>', var, '>', 40)
    rule4_2 = Rule(10, '>=', var, '>=', 40)
    rule4_3 = Rule(20, '>', var, '>', 30)
    rule4_4 = Rule(30, '<', var, '<', 20)
    rule4_5 = Rule(40, '<=', var, '<=', 10)

    assert rule4_4 == rule4_3
    assert rule4_5 == rule4_2

    assert rule4_1.rule_type == 4
    assert rule4_2.rule_type == 4
    assert rule4_3.rule_type == 4

    assert not rule4_1.is_categorical_rule
    assert rule4_1.is_interval_rule
    assert not rule4_1.is_inner_interval_rule

    assert repr(rule4_1) == 'type4 < 10 or type4 > 40'
    assert repr(rule4_2) == 'type4 ≤ 10 or type4 ≥ 40'
    assert repr(rule4_3) == 'type4 < 20 or type4 > 30'

    data = pd.DataFrame(np.arange(50).reshape((-1, 1)), columns=['type4'])
    assert rule4_1.get_matching_mask(data).sum() == 19
    assert rule4_2.get_matching_mask(data).sum() == 21
    assert rule4_3.get_matching_mask(data).sum() == 39

    assert rule4_1.combine(rule4_2) is None
    assert rule4_1.combine(rule4_3) is None
    r1 = Rule(rule4_1.min, rule4_1.operator_min, rule4_1.variable, rule4_1.operator_max, rule4_1.max)
    assert r1 == rule4_1


def test_combine():
    var1 = Variable(0, 'comb1', 'float')
    rule1_1 = Rule(None, None, var1, '<', 20)
    rule1_2 = Rule(None, None, var1, '<', 10)
    rule1_3 = Rule(None, None, var1, '<=', 10)
    rule1_4 = Rule(None, None, var1, '<', 5)
    rule2_1 = Rule(10, '<=', var1, None, None)
    rule3_1 = Rule(10, '<=', var1, '<', 40)
    rule4_1 = Rule(10, '>', var1, '>', 40)

    assert repr(rule2_1.combine(rule1_1)) == '10 ≤ comb1 < 20'
    assert rule2_1.combine(rule1_2) is None
    assert repr(rule2_1.combine(rule1_3)) == '10 ≤ comb1 ≤ 10'
    assert rule2_1.combine(rule1_4) is None

    assert repr(rule3_1.combine(rule1_1)) == '10 ≤ comb1 < 20'
    assert rule3_1.combine(rule1_2) is None
    assert repr(rule3_1.combine(rule1_3)) == '10 ≤ comb1 ≤ 10'
    assert rule3_1.combine(rule1_4) is None

    assert rule4_1.combine(rule1_1) is None
    assert rule4_1.combine(rule1_2) is None
    assert rule4_1.combine(rule1_3) is None
    assert rule4_1.combine(rule1_4) is None