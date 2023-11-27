import pandas as pd

import ipyvuetify as v
from plotly.graph_objects import FigureWidget, Histogram

from antakia.data_handler.rules import Rule
from antakia.gui.widgets import change_widget, get_widget, app_widget

from antakia.utils.logging import conf_logger

from copy import copy
import logging

from antakia.utils.variable import DataVariables

logger = logging.getLogger(__name__)
conf_logger(logger)


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

    def __init__(self, rule: Rule, X: pd.DataFrame, values_space: bool, init_rules_mask: pd.Series,
                 rule_updated: callable):
        self.rule = rule
        self.X = X
        self.values_space = values_space
        self.rule_updated = rule_updated

        # root_widget is an ExpansionPanel
        self.root_widget = v.ExpansionPanel(
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

        # We set the select widget (slider, rangeslider ...)
        self.select_widget = self._get_select_widget()
        self._set_select_widget_values()
        self.select_widget.on_event("change", self._widget_value_changed)
        change_widget(self.root_widget, "101", self.select_widget)

        # Now we build the figure with 3 histograms :
        self._draw_histograms(init_rules_mask)

        change_widget(self.root_widget, "11", self.figure)
        self.update(init_rules_mask)

    def _draw_histograms(self, init_rules_mask: pd.Series):
        # 1) X[var] : all values
        self.figure = FigureWidget(
            data=[
                Histogram(
                    x=self.X[self.rule.variable.symbol],
                    bingroup=1,
                    nbinsx=50,
                    marker_color="grey")
            ]
        )
        # 2) X[var] with only INITIAL SKR rule 'matching indexes'
        X_skr = self.X.loc[init_rules_mask, self.rule.variable.symbol]
        self.figure.add_trace(
            Histogram(
                x=X_skr,
                bingroup=1,
                nbinsx=50,
                marker_color="LightSkyBlue",
                opacity=0.6,
            )
        )
        # 3) X[var] with only CURRENT rule 'matching indexes'
        self.figure.add_trace(
            Histogram(
                bingroup=1,
                nbinsx=50,
                marker_color="blue"
            )
        )
        self.figure.update_layout(
            barmode="overlay",
            bargap=0.1,
            # width=600,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
        )

    def _get_select_widget(self):
        if self.rule.is_categorical_rule:
            change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} possible values :")
            return v.Select(
                label=self.rule.variable.symbol,
                items=self.X[self.rule.variable.symbol].unique().tolist(),
                style_="width: 150px",
                multiple=True,
            )
        min_ = float(self.X[self.rule.variable.symbol].min())
        max_ = float(self.X[self.rule.variable.symbol].max())
        step = (max_ - min_) / 100
        if self.rule.rule_type == 1:  # var < max
            change_widget(self.root_widget, "00",
                          f"{self.rule.variable.symbol} lesser than {'or equal to ' if self.rule.operator_max == 0 else ''}:")
            return v.Slider(
                # class_="ma-3",
                min=min_,
                max=max_,
                color='green',  # outside color
                track_color='red',  # inside color
                thumb_color='blue',  # marker color
                step=step,  # TODO we could divide the spread by 50 ?
                thumb_label="always"
            )
        if self.rule.rule_type == 2:  # var > min
            change_widget(self.root_widget, "00",
                          f"{self.rule.variable.symbol} greater than {'or equal to ' if self.rule.operator_min == 4 else ''}:")
            return v.Slider(
                # class_="ma-3",
                min=min_,
                max=max_,
                color='red',  # greater color
                track_color='green',  # lesser color
                thumb_color='blue',  # marker color
                step=step,  # TODO set according to the variable distribution
                thumb_label="always"
            )
        if self.rule.is_inner_interval_rule:
            change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} inside the interval:")
            return v.RangeSlider(
                min=min_,
                max=max_,
                step=step,
                color='green',  # outside color
                track_color='red',  # inside color
                thumb_color='blue',  # marker color
                thumb_label="always",
                thumb_size=30,
            )
        change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} outside the interval:")
        return v.RangeSlider(
            min=min_,
            max=max_,
            step=step,
            color='red',  # inside color
            track_color='green',  # outside color
            thumb_color='blue',
            thumb_label="always",
            thumb_size=30,
        )

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
        logger.info(str((min_, self.rule.operator_min, self.rule.variable, self.rule.operator_max, max_, cat_values)))
        new_rule = Rule(min_, self.rule.operator_min, self.rule.variable, self.rule.operator_max, max_, cat_values)
        logger.info(str(new_rule))
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
    selection_ids : list of indexes of the GUI selection (from X.index)
    is_value_space : bool
    is_disabled : bool
    rules_mask : Dataframe mask of the rules
    rules_updated : callable of the GUI parent
    root_wiget : its widget representation
    """

    def __init__(
            self,
            X: pd.DataFrame,
            variables: DataVariables,
            values_space: bool,
            new_rules_defined: callable = None,
    ):

        self.X = X
        self.variables: DataVariables = variables
        self.rules_mask = None
        self.is_value_space = values_space
        self.new_rules_defined = new_rules_defined

        # The root widget is a v.Col - we get it from app_widget
        self.root_widget = get_widget(app_widget, "4310") if values_space else get_widget(app_widget, "4311")

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

        self._show_score()
        self._show_rules()

        # We set an empty ExpansionPanels :
        change_widget(self.root_widget, "1", v.ExpansionPanels())

    def init_rules(self, rules_list: list, score_dict: dict, selection_mask: pd.Series):
        """
        Called to set rules or clear them. To update, use update_rule
        """
        logger.info('init_rules in')
        self.selection_mask = selection_mask

        self.is_disabled = False
        # We populate the db and ask to erase past rules
        self._put_in_db(rules_list, score_dict, True)
        # We set our card info
        self.set_rules_info()
        self._show_score()
        self._show_rules()

        # We create the RuleWidgets
        rules_mask = Rule.rules_to_mask(self.get_current_rules_list(), self.X)
        self._create_rule_widgets(rules_mask)

        # We notify the GUI and ask to draw the rules on HDEs
        self.new_rules_defined(self, rules_mask, True)
        logger.info('init_rules out')

    def update_rule(self, new_rule: Rule):
        """
        Called by a RuleWidget when a slider has been modified
        """

        # We update the rule in the db
        new_rules_list = copy(self.get_current_rules_list())
        for i in range(len(new_rules_list)):
            if new_rules_list[i].variable.symbol == new_rule.variable.symbol:
                new_rules_list[i] = new_rule
                break

        # The list of our 'rules model' 'predicted positives'
        new_rules_mask = Rule.rules_to_mask(new_rules_list, self.X)
        # self.selection_ids = the list of the true positives (i.e. in the selection)

        try:
            precision = (new_rules_mask & self.selection_mask).sum() / new_rules_mask.sum()
            recall = (new_rules_mask & self.selection_mask).sum() / self.selection_mask.sum()
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            precision = recall = f1 = -1

        new_score_dict = {"precision": precision, "recall": recall, "f1": f1}

        self._put_in_db(new_rules_list, new_score_dict)

        # We update our card info
        self.set_rules_info()

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(new_rules_mask)

        # We notify the GUI and tell there are new rules to draw
        self.new_rules_defined(self, new_rules_mask)

    def undo(self):
        """
        Restore the previous rules
        """
        # We remove last rules item from the db:
        if len(self.rules_db)>1:
            self.rules_db.pop(-1)
        self.set_rules_info()

        # We compute again the rules mask
        rules_mask = Rule.rules_to_mask(self.get_current_rules_list(), self.X)

        def find_rule(rule_widget: RuleWidget) -> Rule:
            var = rule_widget.rule.variable
            for rule in self.get_current_rules_list():
                if rule.variable.symbol == var.symbol:
                    return rule

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(rules_mask, find_rule(rw))

        # We notify the GUI and tell there are new rules to draw
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
            txt = txt + f"({i}) : {len(rules_list)} rules:\n"
            for rule in rules_list:
                txt = txt + f"    {rule}\n"
            txt = txt + f"   scores = {scores_dict}\n"
        return txt

    def _create_rule_widgets(self, init_rules_mask: pd.Series):
        """
        Called by self.init_rules
        Creates RuleWidgest, one per rule
        If init_rules_mask is None, we clear all our RuleWidgets
        """
        if init_rules_mask is None:
            return

        # We remove existing RuleWidgets
        for i in reversed(range(len(self.rule_widget_list))):
            self.rule_widget_list[i].root_widget.close()
            self.rule_widget_list.pop(i)

        # We set new RuleWidget list and put it in our ExpansionPanels children
        if self.get_current_rules_list() is None or len(self.get_current_rules_list()) == 0:
            self.rule_widget_list = []
        else:
            self.rule_widget_list = [RuleWidget(rule, self.X, self.is_value_space, init_rules_mask, self.update_rule)
                                     for rule in self.get_current_rules_list()]
            get_widget(self.root_widget, "1").children = [rule_widget.root_widget for rule_widget in
                                                          self.rule_widget_list]

    def set_rules_info(self):
        """
        Sets scores and rule details int the DataTable
        """
        # We set the title
        if self.get_current_rules_list() is None or len(self.get_current_rules_list()) == 0:
            title = f"No rule to display for the {'VS' if self.is_value_space else 'ES'} space"
        else:
            title = f"Rule(s) applied to the {'values' if self.is_value_space else 'explanations'} space"
        change_widget(self.root_widget, "001", v.Html(class_="ml-3", tag="h2", children=[title]))

        # We set the scores
        self._show_score()

        # We set the rules
        self._show_rules()

        # We set the rules in the DataTable
        change_widget(self.root_widget, "011", v.DataTable(
            v_model=[],
            show_select=False,
            headers=[{"text": column, "sortable": False, "value": column} for column in
                     ['Variable', 'Unit', 'Desc', 'Critical', 'Rule']],
            items=Rule.rules_to_dict_list(self.get_current_rules_list()),
            hide_default_footer=True,
            disable_sort=True,
        )
                      )

    def _show_score(self):
        if (
                self.get_current_scores_dict() is None
                or len(self.get_current_scores_dict()) == 0 or self.is_disabled
        ):
            scores_txt = "Precision = n/a, recall = n/a, f1_score = n/a"
            css = "ml-7 grey--text"
        else:
            if self.get_current_scores_dict()['precision'] == -1:
                scores_txt = "No point of the dataset matches the new rules"
                css = "ml-7 red--text"
            else:
                scores_txt = "Precision : " + "{:.2f}".format(
                    self.get_current_scores_dict()['precision']) + ", recall : " + "{:.2f}".format(
                    self.get_current_scores_dict()['recall']) + ", f1_score : " + "{:.2f}".format(
                    self.get_current_scores_dict()['f1'])
                css = "ml-7 black--text"
        change_widget(self.root_widget, "01", v.Html(class_=css, tag="li", children=[scores_txt]))

    def _show_rules(self):
        if (
                self.get_current_rules_list() is None
                or len(self.get_current_rules_list()) == 0 or self.is_disabled
        ):
            rules_txt = "N/A"
            css = "ml-7 grey--text"
        else:
            rules_txt = f"{self.get_current_rules_list()[0]}"
            for i in range(1, len(self.get_current_rules_list())):
                rules_txt += f", {self.get_current_rules_list()[i]}"
            css = "ml-7 blue--text"

        change_widget(self.root_widget, "02", v.Html(class_=css, tag="li", children=[rules_txt]))

    def show_msg(self, msg: str, css: str = ""):
        css = "ml-7 " + css
        change_widget(self.root_widget, "02", v.Html(class_=css, tag="li", children=[msg]))

    def get_current_rules_list(self):
        if len(self.rules_db) == 0:
            return []
        else:
            return self.rules_db[-1][0]

    def get_current_scores_dict(self):
        if len(self.rules_db) == 0:
            return []
        else:
            return self.rules_db[-1][1]

    @property
    def rules_num(self):
        return len(self.rules_db)
