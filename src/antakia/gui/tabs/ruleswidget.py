from __future__ import annotations
from functools import partial
from typing import Callable

import pandas as pd
import ipyvuetify as v

from antakia_core.data_handler import Rule, RuleSet

from antakia_core.utils import boolean_mask
from antakia_core.utils import DataVariables, Variable

from antakia.gui.tabs.rule_wgt import RuleWidget
from antakia.utils.stats import log_errors


class RulesWidget:
    """
    A RulesWidget is a piece of GUI that allows the user to refine a set of rules.
    it is composed by a title section that recaps the rule info and a set of RuleWidgets

    The user can use the slider to change the rules.
    There are 2 RW : VS and ES slides

    init attributes
    X : pd.DataFrame, values or explanations Dataframe depending on the context
    y : pd.DataFrame, target values
    variables : list of Variable (DataVariables)
    values_space : bool : is the widget the value space one
    update_callback: function called after a rule is edited - used to sync ruleswidgets and hdes

    other attributes :
    rules_history : a list of RuleSet maintaining all versions of the ruleset
    init_selection_mask : reference selection mask to compare the rule against - initialized in reinit function
    rule_widget_collection : the collection of RuleWidget objects

    graphical elements :
    rule_card :
    _region_stat_card: widget with region info
        _title_wgt : the title
        _stats_wgt : the rule stats
        _rules_txt_wgt : the rule in plain text

    _rules_widgets : the expansion panel with all RuleWidgets
    widget: the complete widget
    """

    def __init__(
        self,
        X: pd.DataFrame | None,
        y: pd.Series,
        variables: DataVariables,
        values_space: bool,
        update_callback: Callable | None = None,
    ):
        """
        widget to manage rule edition and display
        Parameters
        ----------
        X: pd.Dataframe reference dataset
        y: pd.Series test dataset
        variables
        values_space: is the widget on the value space
        update_callback: callback on rule update
        """
        self.X = X
        self.y = y
        self.variables: DataVariables = variables
        self.reference_mask: pd.Series | None = None
        if self.X is not None:
            self.rule_mask = boolean_mask(X, True)
        self.is_value_space = values_space
        if update_callback is not None:
            self.update_callback: Callable | None = partial(
                update_callback, self, 'rule_updated')
        else:
            self.update_callback = None

        self.rules_history: list[RuleSet] = []

        self.rule_widget_collection: dict[Variable, RuleWidget] = {}

        # At startup, we are disabled :
        self.is_disabled = True

        self._build_widget()
        self._create_rule_widgets()
        self.refresh()

    # ------------------- widget init -------------#
    def _build_widget(self):
        self._title_wgt = v.Html(class_="ml-3",
                                 tag="h2",
                                 children=["not initialized"])
        self._stats_wgt = v.Html(  # 431001 / 01
            class_="ml-7",
            tag="li",
            children=["not initialized"])
        self._rules_txt_wgt = v.Html(  # 431002 / 02
            class_="ml-7", tag="li", children=['N/A'])

        self._region_stat_card = v.Col(  # 43100 / 0
            children=[
                v.Row(  # 431000 / 00
                    children=[
                        v.Icon(children=["mdi-target"]),  #
                        self._title_wgt,
                    ]),
                self._stats_wgt,
                self._rules_txt_wgt,
            ])
        self._rules_widgets = v.ExpansionPanels(  # Holds VS RuleWidgets  # 43101 / 1
            style_="max-width: 95%",
            children=[
                rw.widget for rw in self.rule_widget_collection.values()
            ])
        self.widget = v.Col(  # placeholder for the VS RulesWidget (RsW) # 4310
            class_="col-6",
            children=[self._region_stat_card, self._rules_widgets],
        )
        self.disable()

    def _create_rule_widgets(self):
        """
        creates all rule widgets
                Called by self.init_rules
                If init_rules_mask is None, we clear all our RuleWidgets

        """
        # We set new RuleWidget list and put it in our ExpansionPanels children
        if self.X is not None:
            for variable in self.variables.variables.values():
                self.rule_widget_collection[variable] = RuleWidget(
                    Rule(variable), self.X, self.is_value_space,
                    self.sync_rule_widgets)
            self._rules_widgets.children = [
                rw.widget for rw in self.rule_widget_collection.values()
            ]
        else:
            self._rules_widgets.children = []

    def _reorder_rule_widgets(self, all: bool = False):
        rule_widgets = list(self.rule_widget_collection.values())
        if all:
            rule_widgets.sort(
                key=lambda x: x.rule.get_matching_mask(self.X).mean())
            for i, rw in enumerate(rule_widgets):
                if rw.rule.rule_type >= 0:
                    rw.idx = i / len(rule_widgets)
                else:
                    break
        else:
            rule_widgets.sort(key=lambda x: x.idx if x.idx is not None else 1 +
                              x.rule.get_matching_mask(self.X).mean())
            for i, rw in enumerate(rule_widgets):
                if rw.rule.rule_type >= 0 and not rw.idx:
                    rw.idx = i / len(rule_widgets)
                else:
                    break
        self._rules_widgets.children = [rw.widget for rw in rule_widgets]

    # ------------------ widget update --------------------#

    def _refresh_title(self):
        if len(self.current_rules_set) == 0 or self.is_disabled:
            title = f"No rule to display for the {'VS' if self.is_value_space else 'ES'} space"
            css = "ml-3 grey--text" if self.is_disabled else "ml-3"
        else:
            title = f"Rule(s) applied to the {'values' if self.is_value_space else 'explanations'} space"
            css = "ml-3"
        self._title_wgt.children = [title]
        self._title_wgt.class_ = css

    def _refresh_score(self):
        """
        show rule score
        Returns
        -------

        """
        current_scores_dict = self.current_scores_dict()
        if self.reference_mask is None or self.is_disabled:
            scores_txt = "Precision = n/a, recall = n/a, f1_score = n/a"
            css = "ml-7 grey--text"
        elif current_scores_dict['num_points'] == 0:
            scores_txt = "No point of the dataset matches the new rules"
            css = "ml-7 red--text"
        else:
            precision, recall, f1, target_avg = (
                current_scores_dict['precision'],
                current_scores_dict['recall'],
                current_scores_dict['f1'],
                current_scores_dict['target_avg'],
            )
            scores_txt = (
                f"Precision : {precision:.2f}, recall :{recall:.2f} ," +
                f" f1_score : {f1:.2f}, target_avg : {target_avg:.2f}")
            css = "ml-7 black--text"
        self._stats_wgt.children = [scores_txt]
        self._stats_wgt.class_ = css

    def _refresh_rule_txt(self):
        """
        show rules as text
        Returns
        -------

        """
        if (len(self.current_rules_set) == 0 or self.is_disabled):
            rules_txt = "N/A"
            css = "ml-7 grey--text"
        else:
            rules_txt = repr(self.current_rules_set)
            css = "ml-7 blue--text"
        self._rules_txt_wgt.children = [rules_txt]
        self._rules_txt_wgt.class_ = css

    def _refresh_macro_info(self):
        """
        Sets macro widget info and Datatable: scores and rule details int the DataTable
        Returns
        -------

        """
        # We set the title
        self._refresh_title()

        # We set the scores
        self._refresh_score()

        # We set the rules
        self._refresh_rule_txt()

    def _refresh_rule_widgets(self):
        """
        update all rules widgets to match the current rule and selection_mask
        """
        selectable_masks = self._get_selectable_masks(
        )  # for rule, mask_wo_rule in mask_wo_rule:
        for var, rule_wgt in self.rule_widget_collection.items():
            rule_wgt.update(self.rule_mask,
                            selectable_masks.get(var, self.rule_mask),
                            self.current_rules_set.get_rule(var))
        self._reorder_rule_widgets()

    def refresh(self):
        if self.X is not None:
            self._refresh_macro_info()
            self._refresh_rule_widgets()

    # ------------- widget macro method (disable/refresh) ------------- #
    def disable(self, is_disabled: bool = True):
        """
        disables (True) or enables (False) the widget
        Disabled : card in grey with dummy text + no ExpansionPanels
        Enabled : card in light blue / ready to display RuleWidgets
        """

        self.is_disabled = is_disabled

        self.refresh()

    def show_msg(self, msg: str, css: str = ""):
        """
        print a message for the user
        Parameters
        ----------
        msg : message to be printed
        css : css class to apply

        Returns
        -------

        """
        css = "ml-7 " + css
        self._rules_txt_wgt.children = [msg]
        self._rules_txt_wgt.class_ = css

    # ---------------- update logic -------------------------- #
    def change_underlying_dataframe(self, X: pd.DataFrame):
        """
        change the reference dataset
        Parameters
        ----------
        X: pd.DataFrame

        Returns
        -------

        """
        if id(X) == id(self.X):
            return
        self.X = X
        self.update_reference_mask(boolean_mask(X, True))
        self.update_rule_mask(boolean_mask(X, True), False)
        self._create_rule_widgets()
        self.refresh()

    def change_rules(self,
                     rules_set: RuleSet,
                     reference_mask: pd.Series | None = None,
                     reset: bool = False):
        """
        initialize the widget with the rule list and
        the reference_mask (selection_mask)

        """
        # we start wih a fresh widget
        if reset:
            self._reset_widget()
        self._put_in_db(rules_set)
        self.disable(False)
        if reference_mask is not None:
            self.update_reference_mask(reference_mask)
        self.update_rule_mask(rules_set.get_matching_mask(self.X), False)

    def _get_selectable_masks(self) -> dict[Variable, pd.Series]:
        """
        computes for each rule the selection mask, ignoring the rule
        used for update each rule widget
        Returns
        -------

        """
        res = {}
        for rule in self.current_rules_set.rules.values():
            rules_minus_r = [
                r for r in self.current_rules_set.rules.values() if r != rule
            ]
            mask = RuleSet(rules_minus_r).get_matching_mask(self.X)
            res[rule.variable] = mask
        return res

    def _reset_widget(self):
        """
        a reseted widget is
        - disabled
        - history is erased
        - reference mask is None
        Returns
        -------

        """
        self.disable()
        self.rules_history = []
        for var, rule_wgt in self.rule_widget_collection.items():
            rule_wgt.reinit_rule(Rule(var), boolean_mask(self.X, True))
        self.reference_mask = None
        self.refresh()

    @log_errors
    def sync_rule_widgets(self, caller, event, new_rule: Rule):
        """
        callback to synchronize rules in the widget
        called by the edition of a single rule
        """
        if new_rule.rule_type == -1:
            caller.idx = None
        # set rule
        new_rules_set = self.current_rules_set.copy()
        new_rules_set.replace(new_rule)
        new_rules_mask = new_rules_set.get_matching_mask(self.X)
        self._put_in_db(new_rules_set)
        self.update_rule_mask(new_rules_mask=new_rules_mask, sync=True)

    def update_reference_mask(self, reference_mask: pd.Series):
        self.reference_mask = reference_mask
        for var, rule_wgt in self.rule_widget_collection.items():
            rule = self.current_rules_set.get_rule(var)
            if rule is not None:
                rule_wgt.reinit_rule(rule, self.reference_mask)
            else:
                rule_wgt.reinit_rule(Rule(var), self.reference_mask)
        self.refresh()

    def update_rule_mask(self, new_rules_mask: pd.Series, sync=True):
        """
        updates the widget with the new rule_mask and rule list - the reference_mask is kept for comparison
        called by update_rule and sync RulesWidgets
        Parameters
        ----------
        new_rules_mask : mask to display
        new_rule : rule to edit
        sync : whether or not to call the sync callback (self.new_rules_defined) to update the other rules widget
            necessary to avoid call loops

        Returns
        -------

        """

        self.rule_mask = new_rules_mask
        # We refresh the widget
        self.refresh()
        # We notify the GUI and tell there are new rules to draw
        if self.update_callback is not None and sync:
            self.update_callback(new_rules_mask)

    def undo(self):
        """
        Restore the previous rules
        """
        # We remove last rules item from the db:
        if len(self.rules_history) > 1:
            self.rules_history.pop(-1)
            self.rule_mask = self.current_rules_set.get_matching_mask(self.X)

        # We compute again the rules mask

        # we refresh the widget macro info
        self.refresh()

        # We notify the GUI and tell there are new rules to draw if necessary
        if self.update_callback is not None:
            self.update_callback(self.rule_mask)

    def _put_in_db(self, rule_set: RuleSet):
        """
        add the rule list to memory
        Parameters
        ----------
        rule_set

        Returns
        -------

        """
        self.rules_history.append(rule_set)

    @property
    def current_rules_set(self) -> RuleSet:
        """
        get the current rule set
        Returns
        -------

        """
        if len(self.rules_history) == 0:
            return RuleSet()
        else:
            return self.rules_history[-1]

    def current_scores_dict(self) -> dict:
        """
        computes the current score dict
        Returns
        -------

        """
        if self.reference_mask is None or self.is_disabled:
            return {}
        rules_mask = self.current_rules_set.get_matching_mask(self.X)
        precision = (rules_mask & self.reference_mask).sum() / rules_mask.sum()
        recall = (rules_mask
                  & self.reference_mask).sum() / self.reference_mask.sum()
        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'num_points': rules_mask.sum(),
            'target_avg': self.y[rules_mask].mean(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    @property
    def history_size(self) -> int:
        """
        get the size of the db
        Returns
        -------

        """
        return len(self.rules_history)
