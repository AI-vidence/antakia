import math

import numpy as np
import pandas as pd

import ipyvuetify as v
from plotly.graph_objects import FigureWidget, Histogram

from antakia.data_handler.rules import Rule
from antakia.gui.widgets import change_widget, get_widget, app_widget

from copy import copy

from antakia.utils.variable import DataVariables


class RuleWidget:
    """
    A piece of UI handled by a RulesExplorer
    It allows the user to modify one rule

    Attributes :
    rule : Rule
    values_space : bool
    root_widget : its widget representation
    select_widget : the widget that allows the user to modify the rule
    figure : FigureWidget
    X
    rule_updated : callable of RulesWidget
    """

    def __init__(self, rule: Rule, X: pd.DataFrame, y, values_space: bool, init_rules_mask: pd.Series,
                 rule_updated: callable):
        self.rule = rule
        self.X = X
        self.values_space = values_space
        self.rule_updated = rule_updated
        self.display_sliders = self.values_space
        self.root_widget = None

        self.build_widget(init_rules_mask)

    def build_widget(self, init_rules_mask):
        # root_widget is an ExpansionPanel
        self.root_widget: v.ExpansionPanel = v.ExpansionPanel(
            children=[
                v.ExpansionPanelHeader(
                    class_="blue lighten-4",
                    children=[
                        "Placeholder for variable symbol"  # 1000
                    ]
                ),
                v.ExpansionPanelContent(
                    children=[
                        v.Col(
                            children=[
                                v.Spacer(),
                                v.Slider(  # placeholder for select widget
                                ),
                            ],
                        ),
                        FigureWidget(  # Placeholder for figure
                        ),
                    ]
                ),
            ]
        )

        # The variable name bg (ExpansionPanelHeader) is light blue
        get_widget(self.root_widget, "0").class_ = "blue lighten-4"
        self._set_panel_title()
        # We set the select widget (slider, rangeslider ...)
        if self.display_sliders:
            self.select_widget = self._build_select_widget()
            self._set_select_widget_values()
            self.select_widget.on_event("change", self._widget_value_changed)
            change_widget(self.root_widget, "101", self.select_widget)
        else:
            change_widget(self.root_widget, "101", None)

        # Now we build the figure with 3 histograms :
        self._draw_histograms(init_rules_mask)

    def _draw_histograms(self, init_rules_mask: pd.Series):
        base_args = {
            'bingroup': 1,
            'nbinsx': 50,
        }
        # 1) X[var] : all values
        self.figure = FigureWidget(
            data=[
                Histogram(
                    x=self.X[self.rule.variable.symbol],
                    marker_color="grey",
                    **base_args
                )
            ]
        )
        # 2) X[var] with only INITIAL SKR rule 'matching indexes'
        X_skr = self.X.loc[init_rules_mask, self.rule.variable.symbol]
        self.figure.add_trace(
            Histogram(
                x=X_skr,
                marker_color="LightSkyBlue",
                opacity=0.6,
                **base_args
            )
        )
        # 3) X[var] with only CURRENT rule 'matching indexes'
        self.figure.add_trace(
            Histogram(
                x=X_skr,
                marker_color="blue",
                **base_args
            )
        )
        self.figure.update_layout(
            barmode="overlay",
            bargap=0.1,
            # width=600,
            showlegend=False,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            height=200,
        )
        # display
        change_widget(self.root_widget, "11", self.figure)

    def _set_panel_title(self):
        if self.rule.is_categorical_rule:
            title = f"{self.rule.variable.symbol} possible values :"
        elif self.rule.rule_type == 1:  # var < max
            title = f"{self.rule.variable.symbol} lesser than {'or equal to ' if self.rule.include_equals else ''}:"
        elif self.rule.rule_type == 2:  # var > min
            title = f"{self.rule.variable.symbol} greater than {'or equal to ' if self.rule.include_equals else ''}:"
        elif self.rule.is_inner_interval_rule:
            title = f"{self.rule.variable.symbol} inside the interval:"
        else:
            title = f"{self.rule.variable.symbol} outside the interval:"
        return change_widget(self.root_widget, "00", title)

    def _build_select_widget(self):
        if self.rule.is_categorical_rule:
            return v.Select(
                label=self.rule.variable.symbol,
                items=self.X[self.rule.variable.symbol].unique().tolist(),
                style_="width: 150px",
                multiple=True,
            )
        min_ = float(self.X[self.rule.variable.symbol].min())
        max_ = float(self.X[self.rule.variable.symbol].max())
        step = (max_ - min_) / 100
        round_value = round(math.log(step / 2) / math.log(10)) - 1
        min_ = np.round(min_, -round_value)
        max_ = np.round(max_, -round_value)
        step = np.round(step, -round_value)
        slider_args = {
            'min': min_,
            'max': max_,
            'step': step,
            'thumb_label': "always",
            'thumb_size': 30,
            'thumb_color': 'blue'
        }
        if self.rule.rule_type == 1:  # var < max
            slider = v.Slider
            slider_args['color'] = 'green'
            slider_args['track_color'] = 'red'
        elif self.rule.rule_type == 2:  # var > min
            slider = v.Slider
            slider_args['color'] = 'red'
            slider_args['track_color'] = 'green'
        elif self.rule.is_inner_interval_rule:
            slider = v.RangeSlider
            slider_args['color'] = 'green'
            slider_args['track_color'] = 'red'
        else:
            slider = v.RangeSlider
            slider_args['color'] = 'red'
            slider_args['track_color'] = 'green'
        return slider(**slider_args)

    def _set_select_widget_values(self):
        if self.rule.is_categorical_rule:
            self.select_widget.v_model = self.rule.cat_values
        elif self.rule.rule_type == 1:  # var < max
            self.select_widget.v_model = [self.rule.max],
        elif self.rule.rule_type == 2:  # var > min
            self.select_widget.v_model = [self.rule.min]
        else:
            self.select_widget.v_model = [self.rule.min, self.rule.max]

    def _widget_value_changed(self, widget, event, data):
        cat_values = None
        if self.rule.is_categorical_rule:
            cat_values = data
            min_ = max_ = None
        else:
            if isinstance(data, list):  # Interval rule
                min_ = data[0]
                max_ = data[1]
            elif self.rule.rule_type == 1:  # var < max
                min_ = None
                max_ = data
            else:
                min_ = data
                max_ = None
        new_rule = Rule(min_, self.rule.operator_min, self.rule.variable, self.rule.operator_max, max_, cat_values)
        self.rule = new_rule
        self.rule_updated(new_rule)

    def update(self, new_rules_mask: pd.Series, previous_rule: Rule = None):
        """ 
        Called by the RulesWidget. Each RuleWidget must now use these indexes
        Also used to restore previous rule
        """

        if previous_rule is not None:
            self.rule = previous_rule

        # We update the selects
        if self.display_sliders:
            self._set_select_widget_values()

        # We update the third histogram only
        with self.figure.batch_update():
            self.figure.data[2].x = self.X.loc[new_rules_mask, self.rule.variable.symbol]


class RulesWidget:
    """
    A RulesWidget is a piece of GUI that allows the user to refine a set of rules.
    The user can use the slider to change the rules.
    There are 2 RW : VS and ES slides

    rules_db : a dict of list :
        [[rules_list, scores]]], so that at iteration i we have :
        [i][0] : the list of rules
        [i][1] : the scores as a dict {"precision": ?, "recall": ?, "f1": ?}
    X : pd.DataFrame, values or explanations Dataframe depending on the context
    variables : list of Variable
    rules_mask : list of indexes of the GUI selection (from X.index)
    is_value_space : bool
    is_disabled : bool
    rules_mask : Dataframe mask of the rules
    rules_updated : callable of the GUI parent
    root_wiget : its widget representation
    """

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            variables: DataVariables,
            values_space: bool,
            new_rules_defined: callable = None,
    ):

        self.X = X
        self.y = y
        self.variables: DataVariables = variables
        self.rules_mask = None
        self.is_value_space = values_space
        self.new_rules_defined = new_rules_defined

        # The root widget is a v.Col - we get it from app_widget
        self.root_widget = get_widget(app_widget, "4310" if values_space else "4311")

        self.rules_db = []
        self.rule_widget_list = []

        # At startup, we are disabled :
        self.disable()

    @property
    def space(self):
        return 'values' if self.is_value_space else 'explanations'

    def enable(self):
        return self._set_disable(False)

    def disable(self):
        return self._set_disable(True)

    def _set_disable(self, is_disabled: bool = True):
        """
        Disabled : card in grey with dummy text + no ExpansionPanels
        Enabled : card in light blue / ready to display RuleWidgets
        """

        self.is_disabled = is_disabled

        header_html = get_widget(self.root_widget, "001")
        header_html.children = [
            f"No rule to display for the {self.space} space"
        ]
        header_html.class_ = "ml-3 grey--text" if self.is_disabled else "ml-3"

        # We set an empty ExpansionPanels :
        change_widget(self.root_widget, "1", v.ExpansionPanels())

    def init_rules(self, rules_list: list, score_dict: dict, rules_mask: pd.Series):
        """
        Called to set rules or clear them. To update, use update_rule
        """
        self.rules_mask = rules_mask

        self.is_disabled = False
        # We populate the db and ask to erase past rules
        self._put_in_db(rules_list, score_dict, True)
        # We set our card info
        self.refresh_widget()

        # We create the RuleWidgets
        current_rules_mask = Rule.rules_to_mask(self.current_rules_list, self.X)
        self._create_rule_widgets(current_rules_mask)

    def reset_widget(self):
        self.disable()
        self.rules_db = []
        self.rule_widget_list = []
        self.rules_mask = None
        self.refresh_widget()

    def update_rule(self, new_rule: Rule):
        """
        Called by a RuleWidget when a slider has been modified
        """

        # We update the rule in the db
        new_rules_list = copy(self.current_rules_list)
        for i in range(len(new_rules_list)):
            if new_rules_list[i].variable.symbol == new_rule.variable.symbol:
                new_rules_list[i] = new_rule
                break

        # The list of our 'rules model' 'predicted positives'
        new_rules_mask = Rule.rules_to_mask(new_rules_list, self.X)
        self.update_from_mask(new_rules_mask, new_rules_list)

    def update_from_mask(self, new_rules_mask, new_rules_list):

        try:
            precision = (new_rules_mask & self.rules_mask).sum() / new_rules_mask.sum()
            recall = (new_rules_mask & self.rules_mask).sum() / self.rules_mask.sum()
            f1 = 2 * (precision * recall) / (precision + recall)
            target_avg = self.y[new_rules_mask].mean()
        except ZeroDivisionError:
            precision = recall = f1 = -1

        new_score_dict = {"precision": precision, "recall": recall, "f1": f1, 'target_avg': target_avg}

        self._put_in_db(new_rules_list, new_score_dict)

        # We update our card info
        self.refresh_widget()

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(new_rules_mask)

        # We notify the GUI and tell there are new rules to draw
        if self.new_rules_defined is not None:
            self.new_rules_defined(self, new_rules_mask)

    def undo(self):
        """
        Restore the previous rules
        """
        # We remove last rules item from the db:
        if len(self.rules_db) > 1:
            self.rules_db.pop(-1)

        # We compute again the rules mask
        rules_mask = Rule.rules_to_mask(self.current_rules_list, self.X)

        def find_rule(rule_widget: RuleWidget) -> Rule:
            var = rule_widget.rule.variable
            for rule in self.current_rules_list:
                if rule.variable.symbol == var.symbol:
                    return rule

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(rules_mask, find_rule(rw))

        # we refresh the widget
        self.refresh_widget()
        # We notify the GUI and tell there are new rules to draw
        if self.new_rules_defined is not None:
            self.new_rules_defined(self, rules_mask)

    def _put_in_db(self, rules_list: list, score_dict: dict, erase_past: bool = False):
        if erase_past:
            self.rules_db = []
        self.rules_db.append([rules_list, score_dict])

    def _dump_db(self) -> str:
        txt = ""
        for i in range(len(self.rules_db)):
            rules_list = self.rules_db[i][0]
            scores_dict = self.rules_db[i][1]

            txt += f"({i}) : {len(rules_list)} rules:"
            "\n    ".join(rules_list)
            txt += f"\n   scores = {scores_dict}\n"
        return txt

    def _create_rule_widgets(self, init_rules_mask: pd.Series):
        """
        Called by self.init_rules
        Creates RuleWidgest, one per rule
        If init_rules_mask is None, we clear all our RuleWidgets
        """
        if init_rules_mask is None:
            return

        # We set new RuleWidget list and put it in our ExpansionPanels children
        if len(self.current_rules_list) == 0:
            self.rule_widget_list = []
        else:
            self.rule_widget_list = [
                RuleWidget(rule, self.X, self.y, self.is_value_space, init_rules_mask, self.update_rule)
                for rule in self.current_rules_list]
            get_widget(self.root_widget, "1").children = [rule_widget.root_widget for rule_widget in
                                                          self.rule_widget_list]

    def refresh_widget(self):
        """
        Sets scores and rule details int the DataTable
        """
        # We set the title
        if len(self.current_rules_list) == 0:
            title = f"No rule to display for the {'VS' if self.is_value_space else 'ES'} space"
        else:
            title = f"Rule(s) applied to the {'values' if self.is_value_space else 'explanations'} space"
        change_widget(self.root_widget, "001", v.Html(class_="ml-3", tag="h2", children=[title]))

        # We set the scores
        self._show_score()

        # We set the rules
        self._show_rules()

        # We set the rules in the DataTable
        change_widget(
            self.root_widget,
            "011", v.DataTable(
                v_model=[],
                show_select=False,
                headers=[{"text": column, "sortable": False, "value": column} for column in
                         ['Variable', 'Unit', 'Desc', 'Critical', 'Rule']],
                items=Rule.rules_to_dict_list(self.current_rules_list),
                hide_default_footer=True,
                disable_sort=True,
            )
        )

    def _show_score(self):
        if len(self.current_scores_dict) == 0 or self.is_disabled:
            scores_txt = "Precision = n/a, recall = n/a, f1_score = n/a"
            css = "ml-7 grey--text"
        elif self.current_scores_dict['precision'] == -1:
            scores_txt = "No point of the dataset matches the new rules"
            css = "ml-7 red--text"
        else:
            precision, recall, f1, target_avg = (
                self.current_scores_dict['precision'], self.current_scores_dict['recall'],
                self.current_scores_dict['f1'], self.current_scores_dict['target_avg'],
            )
            scores_txt = (f"Precision : {precision:.2f}, recall :{recall:.2f} ," +
                          f" f1_score : {f1:.2f}, target_avg : {target_avg:.2f}")
            css = "ml-7 black--text"
        change_widget(self.root_widget, "01", v.Html(class_=css, tag="li", children=[scores_txt]))

    def _show_rules(self):
        if (
                len(self.current_rules_list) == 0
                or self.is_disabled
        ):
            rules_txt = "N/A"
            css = "ml-7 grey--text"
        else:
            rules_txt = Rule.multi_rules_to_string(self.current_rules_list)
            css = "ml-7 blue--text"

        change_widget(self.root_widget, "02", v.Html(class_=css, tag="li", children=[rules_txt]))

    def show_msg(self, msg: str, css: str = ""):
        css = "ml-7 " + css
        change_widget(self.root_widget, "02", v.Html(class_=css, tag="li", children=[msg]))

    @property
    def current_rules_list(self):
        if len(self.rules_db) == 0:
            return []
        else:
            return self.rules_db[-1][0]

    @property
    def current_scores_dict(self) -> dict:
        if len(self.rules_db) == 0:
            return {}
        else:
            return self.rules_db[-1][1]

    @property
    def rules_num(self):
        return len(self.rules_db)
