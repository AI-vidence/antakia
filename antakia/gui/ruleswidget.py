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
    figure : FigureWidget
    X
    rule_updated : callable of RulesWidget
    """

    def __init__(self, rule: Rule, X: pd.DataFrame, values_space: bool, init_rules_indexes:list, rule_updated: callable):
        self.rule = rule
        self.X = X
        self.values_space = values_space
        self.rule_updated = rule_updated
        
        # root_widget is an ExpansionPanel
        self.root_widget = get_widget(app_widget,"30501010") if values_space else get_widget(app_widget,"30501110")
        
        # We set the select widget (slider, rangeslider ...)
        if self.rule.is_categorical_rule():
            change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} possible values :")
            select_widget = v.Select(
                label=self.rule.variable.symbol,
                items=self.X[self.rule.variable.symbol].unique().tolist(),
                v_model=self.rule.cat_values,
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
                    v_model=[self.rule.max],
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
                    v_model=[self.rule.min],
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
                        v_model=[
                            self.rule.min,
                            self.rule.max,
                        ],
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
                        v_model=[
                            self.rule.min,
                            self.rule.max,
                        ],
                        min=min_,
                        max=max_,
                        step=0.1,
                        color='red', # inside color
                        track_color='green', # outside color
                        thumb_color='blue',
                        thumb_label="always",
                        thumb_size=30,
                    )
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
        self.update_figure(init_rules_indexes)

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


    def update_figure(self, new_rules_indexes:list):
        """ 
        Called by the RulesWidget. Each RuleWidget must now use these indexes
        """
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
    is_value_space : bool
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
    


        # UI:
        self.root_widget = get_widget(app_widget, "305010") if values_space else get_widget(app_widget, "305011")

        self.rule_widget_list = []

    def _set_score(self, score_dict: dict):
        if score_dict is None or len(score_dict) == 0:
            get_widget(self.root_widget, "01").children=[f"Precision = {score_dict['precision']}, recall = {score_dict['recall']}, f1_score = {score_dict['f1']}"]

    def disable(self, is_disabled: bool):
        """
        SKR computation only takes place in one space. GUI may call this method
        to disable the other space.
        NOTE : disable = False erases existing rules
        """

        html = get_widget(self.root_widget, "001")
        html.children=[
            f"No rule to display for the {'values' if self.is_value_space else 'explanations'} space"
        ]
        html.class_= "ml-3 grey--text" if is_disabled else "ml-3"
        
        html = get_widget(self.root_widget, "01")
        html.children=["Precision = n/a, recall = n/a, f1_score = n/a"]
        html.class_= "ml-7 grey--text" if is_disabled else "ml-7"
  
        # The first ExpansionPanel
        get_widget(self.root_widget, "10").disabled = is_disabled
        xph = get_widget(self.root_widget, "100")
        xph.children=["Variable"]
        xph.class_ = "light-grey" if is_disabled else "font-weight-bold blue lighten-4"


    def init_rules(self, rules_list: list, score_dict: list):
        """
        Called by the GUI when new SKR rules are computed
        """
        if rules_list is None:
            self.rules_db = {}
            self.current_index = 0
            self.create_rule_widgets([])
            self._set_score({})
            self.disable(True)
            return
        
        # We populate our db
        self._put_in_db(rules_list, score_dict, True)
    
        self.set_rules_info()
        self._set_score(score_dict)
        rules_indexes = Rule.rules_to_indexes(self.get_current_rules_list(), self.X)

        self.create_rule_widgets(rules_indexes)

        # We notify the GUI and ask to draw the rules
        if self.new_rules_defined is not None:
            self.new_rules_defined(self,rules_indexes, True)

    def update_rule(self, new_rule:Rule):
        """
        Called by a RuleWidget when a slider has been modified
        """
        # We update the rule in the db
        rules_list = self.get_current_rules_list()
        
        for i in range(len(rules_list)):
            if rules_list[i].variable.symbol == new_rule.variable.symbol:
                rules_list[i] = new_rule
                break

        # TODO : compute new scores / use fake scores for now
        score_dict = {"precision": 0, "recall": 0, "f1": 0}
        self._set_score(score_dict)
        self._put_in_db(rules_list, score_dict)

        # We update our card info
        self.set_rules_info()

        rules_indexes = Rule.rules_to_indexes(self.get_current_rules_list(), self.X)

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update_figure(rules_indexes)

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

    def create_rule_widgets(self, init_rules_indexes:list):
        """
        Creates RuleWidgest, one per rule
        """
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
        if (
            self.get_current_scores_dict() is None
            or len(self.get_current_scores_dict()) == 0
        ):
            scores_txt = "Precision = n/a, recall = n/a, f1_score = n/a"
        else:
            scores_txt = f"Precision : {self.get_current_scores_dict()['precision']}, recall : {self.get_current_scores_dict()['recall']}, f1_score : {self.get_current_scores_dict()['f1']}"
        change_widget(self.root_widget, "010", v.Html(class_="ml-3", tag="p", children=[scores_txt]))

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

    def add_rules(self, rule_list: list, score_lsit):
        pass

    def get_current_rules_list(self):
        return self.rules_db[self.current_index][0]


    def get_current_scores_dict(self):
        return self.rules_db[self.current_index][1]


