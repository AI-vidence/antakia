from antakia_core.data_handler import Rule, RuleSet

from antakia.gui.tabs.ruleswidget import RuleWidget
from tests.antakia_test_case import AntakiaTestCase


class TestRulesWidget(AntakiaTestCase):
    def test_rule_widget(self):
        col_name = self.data_store.X.columns[0]
        var = self.data_store.variables.get_var(col_name)
        col_name = self.data_store.X.columns[1]
        var2 = self.data_store.variables.get_var(col_name)
        rule1 = Rule(var, max=10, includes_max=False)  # None, None, var, '<', 10)
        rule2 = Rule(var, min=10, includes_min=True)  # 10, '<=', var, None, None)
        rule3 = Rule(
            var, min=10, includes_min=True, max=40, includes_max=False
        )  # 10, '<=', var, '<', 40)
        rule4 = Rule(var2, min=10, includes_min=True)  # 10, '<=', var2, None, None)

        rules1 = RuleSet([rule1, rule4])
        mask1 = rules1.get_matching_mask(self.data_store.X)
        rules2 = RuleSet([rule2, rule4])
        mask2 = rules2.get_matching_mask(self.data_store.X)

        # TODO test with X_exp
        rw = RuleWidget(
            rule1, self.data_store, self.data_store.X, True, lambda x: None, lambda x: None
        )

        # add tests

        rw.update(mask2, rule1)

        # add tests
