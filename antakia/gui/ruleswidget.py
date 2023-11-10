import pandas as pd
import numpy as np

import ipyvuetify as v
from ipywidgets import Layout, widgets
from plotly.graph_objects import FigureWidget, Histogram, Scatter, Scatter3d

from antakia.rules import Rule
from antakia.gui.widgets import change_widget, get_widget, app_widget

from antakia.utils import conf_logger

from copy import copy
import logging

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

    def __init__(self, rule: Rule, X: pd.DataFrame, values_space: bool, init_rules_indexes:list, rule_updated: callable):
        self.rule = rule
        self.X = X
        self.values_space = values_space
        self.rule_updated = rule_updated

        logger.debug(f"RuleWidget.init : rule = {rule}")
        
        # root_widget is an ExpansionPanel
        self.root_widget = v.ExpansionPanel( # PH for VS RuleWidget #421010 10
            children=[
                v.ExpansionPanelHeader(
                    class_="blue lighten-4",
                    children=[
                        "Placeholder for variable symbol" # 1000 
                    ]
                ),
                v.ExpansionPanelContent( 
                    children=[
                        v.Col( 
                            children=[
                                v.Spacer(), 
                                v.Slider( # placeholder for select widget
                                ),
                            ],
                        ),
                        FigureWidget( # Placeholder for figure
                        ),
                    ]
                ),
            ]
        )


        # The variable name bg (ExpansionPanelHeader) is light blue
        get_widget(self.root_widget,"0").class_= "blue lighten-4"
        
        # We set the select widget (slider, rangeslider ...)
        if self.rule.is_categorical_rule():
            change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} possible values :")
            select_widget = v.Select(
                label=self.rule.variable.symbol,
                items=self.X[self.rule.variable.symbol].unique().tolist(),
                style_="width: 150px",
                multiple=True,
                )
        else:
            min_ = min(self.X[self.rule.variable.symbol])
            max_ = max(self.X[self.rule.variable.symbol])            
            if self.rule.min is None: # var < max
                change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} lesser than {'or equal to ' if self.rule.operator_max == 0 else ''}:")
                select_widget = v.Slider(
                    # class_="ma-3",
                    min=min_,
                    max=max_,
                    color='green', # outside color
                    track_color='red', # inside color
                    thumb_color='blue', # marker color
                    step=0.1, # TODO we could divide the spread by 50 ?
                    thumb_label="always"
                    )
            elif self.rule.max is None: # var > min
                change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} greater than {'or equal to ' if self.rule.operator_min == 4 else ''}:")
                select_widget = v.Slider(
                    # class_="ma-3",
                    min=min_,
                    max=max_,
                    color='red', # greater color
                    track_color='green', # lesser color
                    thumb_color='blue', # marker color
                    step=0.1, # TODO set according to the variable distribution
                    thumb_label="always"
                    )
            else:
                if self.rule.is_inner_interval_rule():
                    change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} inside the interval:")
                    select_widget = v.RangeSlider(
                        min=min_,
                        max=max_,
                        step=0.1,
                        color='green', # outside color
                        track_color='red', # inside color
                        thumb_color='blue', # marker color
                        thumb_label="always",
                        thumb_size=30,
                        )
                else: 
                    change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} outside the interval:")
                    select_widget = v.RangeSlider(
                        min=min_,
                        max=max_,
                        step=0.1,
                        color='red', # inside color
                        track_color='green', # outside color
                        thumb_color='blue',
                        thumb_label="always",
                        thumb_size=30,
                    )
        self.select_widget = select_widget
        self.set_select_widget_values()
        select_widget.on_event("change", self.widget_value_changed)
        change_widget(self.root_widget, "101", select_widget)

        # Now we build the figure with 3 histograms :
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
        X_skr= self.X.loc[init_rules_indexes][self.rule.variable.symbol]
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
            # width=0.9 * int(fig_size),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            # height=400,
        )

        change_widget(self.root_widget, "11", self.figure)
        self.update(init_rules_indexes)

    def set_select_widget_values(self):
        if self.rule.is_categorical_rule():
            self.select_widget.v_model=self.rule.cat_values
        elif self.rule.min is None: # var < max
            self.select_widget.v_model=[self.rule.max],
        elif self.rule.max is None: # var > min
            self.select_widget.v_model=[self.rule.min]
        else:
            self.select_widget.v_model=[self.rule.min, self.rule.max]

    def widget_value_changed(self, widget, event, data):
        cat_values = None
        if self.rule.is_categorical_rule():
            cat_values = data
            min = max = None
        else:
            if isinstance(data, list): # Interval rule
                min = data[0]
                max = data[1]
            elif self.rule.min is None: # var < max
                min = None
                max = data
            else:
                min = data
                max = None
        new_rule = Rule(min, self.rule.operator_min, self.rule.variable, self.rule.operator_max, max, cat_values)
        self.rule = new_rule
        self.rule_updated(new_rule)


    def update(self, new_rules_indexes:list, previous_rule:Rule = None):
        """ 
        Called by the RulesWidget. Each RuleWidget must now use these indexes
        Also used to restore previous rule
        """

        if previous_rule is not None:
            self.rule = previous_rule
        
        # We update the selects
        self.set_select_widget_values()

        # We update the third histogram only
        with self.figure.batch_update():
            self.figure.data[2].x = self.X.loc[new_rules_indexes][self.rule.variable.symbol]
        
        

class RulesWidget:
    """
    A RulesWidget is a piece of GUI that allows the user to refine a set of rules.
    The user can use the slider to change the rules.
    There are 2 RW : VS and ES slides

    rules_db : a dict of list :
        [[rules_list, scores]]], so that at iteration i we have :
        [i][0] : the list of rules
        [i][1] : the scores as a dict {"precision": ?, "recall": ?, "f1": ?}
    current_index : int refers to rules_db
    X : pd.DataFrame, values or explanations Dataframe depending on the context
    variables : list of Variable
    selection_ids : list of indexes of the GUI selection (from X.index)
    is_value_space : bool
    is_disabled : bool
    rules_indexes : Dataframe indexes of the rules
    rules_updated : callable of the GUI parent
    root_wiget : its widget representation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        variables: list,
        values_space: bool,
        new_rules_defined: callable = None,
    ):
        
        self.X = X
        self.variables = variables
        self.rules_indexes = None
        self.is_value_space = values_space
        self.new_rules_defined = new_rules_defined 

        # The root widget is a v.Col - we get it from app_widget
        self.root_widget = get_widget(app_widget, "4210") if values_space else get_widget(app_widget, "4211")

        self.rules_db = {}
        self.current_index = -1
        self.rule_widget_list = []

        # At startup, we are disabled :
        self.disable(True)


    def disable(self, is_disabled:bool=True):
        """
        Disabled : card in grey with dummy text + no ExpansionPanels
        Enabled : card in light blue / ready to display RuleWidgets
        """

        self.is_disabled = is_disabled

        header_html= get_widget(self.root_widget, "001")
        header_html.children=[
            f"No rule to display for the {'values' if self.is_value_space else 'explanations'} space"
        ]
        header_html.class_= "ml-3 grey--text" if self.is_disabled else "ml-3"
        
        self._show_score()
        self._show_rules()

        # We set an empty ExpansionPanels :
        change_widget(self.root_widget, "1", v.ExpansionPanels())


    def init_rules(self, rules_list: list, score_dict: list, selection_ids:list ):
        """
        Called to set rules or clear them. To update, use update_rule
        """
        self.selection_ids=selection_ids

        self.is_disabled = False
        # We populate the db and ask to erase past rules
        self._put_in_db(rules_list, score_dict, True)
        # We set our card info
        self.set_rules_info()
        self._show_score()
        self._show_rules()

        # We create the RuleWidgets
        rules_indexes = Rule.rules_to_indexes(self.get_current_rules_list(), self.X)
        self._create_rule_widgets(rules_indexes)

        # We notify the GUI and ask to draw the rules on HDEs
        self.new_rules_defined(self,rules_indexes, True)

    def update_rule(self, new_rule:Rule):
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
        rules_indexes = Rule.rules_to_indexes(self.get_current_rules_list(), self.X)
        # self.selection_ids = the list of the true positives (i.e. in the selection)

        precision = len(set(rules_indexes).intersection(set(self.selection_ids))) / len(rules_indexes)
        recall = len(set(rules_indexes).intersection(set(self.selection_ids))) / len(self.selection_ids)
        f1 = 2 * (precision * recall) / (precision + recall)

        new_score_dict = {"precision": precision, "recall": recall, "f1": f1}

        self._put_in_db(new_rules_list, new_score_dict)

        # We update our card info
        self.set_rules_info()

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(rules_indexes)

        # We notify the GUI and tell there are new rules to draw
        self.new_rules_defined(self,rules_indexes)

    def undo(self):
        """
        Restore the previous rules
        """
        # We remove last rules item from the db:
        self.rules_db.pop(self.current_index)
        self.current_index -= 1
        self.set_rules_info()

        # We compute again the rules indexes
        rules_indexes = Rule.rules_to_indexes(self.get_current_rules_list(), self.X)
        
        def find_rule(rule_widget:RuleWidget) -> Rule:
            var = rule_widget.rule.variable
            for rule in self.get_current_rules_list():
                if rule.variable.symbol == var.symbol:
                    return rule

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(rules_indexes, find_rule(rw))

        # We notify the GUI and tell there are new rules to draw
        self.new_rules_defined(self,rules_indexes)


    def _put_in_db(self, rules_list: list, score_dict: list, erase_past:bool=False):
        if erase_past:
            self.rules_db = {}
            self.current_index = -1
        self.current_index += 1
        self.rules_db[self.current_index] = [rules_list, score_dict]

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

    def _create_rule_widgets(self, init_rules_indexes:list):
        """
        Called by self.init_rules
        Creates RuleWidgest, one per rule
        If init_rules_indexes is None, we clear all our RuleWidgets
        """
        if init_rules_indexes is None:
            return

        # We remove existing RuleWidgets
        for i in reversed(range(len(self.rule_widget_list))):
            self.rule_widget_list[i].root_widget.close()
            self.rule_widget_list.pop(i)

        # We set new RuleWidget list and put it in our ExpansionPanels children
        if self.get_current_rules_list() is None or len(self.get_current_rules_list()) == 0:
            self.rule_widget_list = []
        else:
            self.rule_widget_list = [RuleWidget(rule, self.X, self.is_value_space, init_rules_indexes, self.update_rule) for rule in self.get_current_rules_list()]
            get_widget(self.root_widget, "1").children = [rule_widget.root_widget for rule_widget in self.rule_widget_list]

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
                headers=[{"text": column, "sortable": False, "value": column } for column in ['Variable', 'Unit', 'Desc', 'Critical', 'Rule']],
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
            scores_txt = "Precision : " + "{:.2f}".format(self.get_current_scores_dict()['precision']) + ", recall : " + "{:.2f}".format(self.get_current_scores_dict()['recall']) + ", f1_score : " + "{:.2f}".format(self.get_current_scores_dict()['f1'])
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
            rules_txt = ""
            for rule in self.get_current_rules_list():
                rules_txt += f"{rule} / "
            css = "ml-7 blue--text"
            logger.debug(f"RulesWidget._show_rules : rules_txt = {rules_txt}")

        change_widget(self.root_widget, "02", v.Html(class_=css, tag="li", children=[rules_txt]))

    def show_msg(self, msg:str, css:str = ""):
         css = "ml-7 " + css
         change_widget(self.root_widget, "02", v.Html(class_=css, tag="li", children=[msg]))


    def get_current_rules_list(self):
        if len(self.rules_db) == 0:
            return []
        else:
            return self.rules_db[self.current_index][0]

    def get_current_scores_dict(self):
        if len(self.rules_db) == 0:
            return []
        else:
            return self.rules_db[self.current_index][1]


