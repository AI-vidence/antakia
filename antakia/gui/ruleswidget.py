import pandas as pd
import numpy as np

import ipyvuetify as v
from ipywidgets import Layout, widgets
from plotly.graph_objects import FigureWidget, Histogram, Scatter, Scatter3d

from antakia.rules import Rule
from antakia.gui.widgets import change_widget, get_widget

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
    """

    def __init__(self, rule: Rule, X: pd.DataFrame, values_space: bool):
        self.rule = rule
        self.X = X
        self.values_space = values_space
        
        if values_space:
            self.root_widget = v.ExpansionPanel( 
                children=[
                    v.ExpansionPanelHeader( # 0
                        class_="font-weight-bold blue lighten-4",
                        children=[
                            "Variable"
                        ]
                    ),
                    v.ExpansionPanelContent( # 1
                        children=[
                            v.Col( # 10
                                class_="ma-3 pa-3",
                                children=[
                                    v.Spacer(), # 100
                                    v.RangeSlider(  # 101
                                        class_="ma-3",
                                        v_model=[
                                            -1,
                                            1,
                                        ],
                                        min=-5,
                                        max=5,
                                        step=0.1,
                                        thumb_label="always",
                                    ),
                                ],
                            ),
                            FigureWidget( # 11
                                data=[
                                    Histogram(
                                        x=pd.DataFrame(
                                            np.random.randint(
                                                0,
                                                100,
                                                size=(
                                                    100,
                                                    4,
                                                ),
                                            ),
                                            columns=list(
                                                "ABCD"
                                            ),
                                        ),
                                        bingroup=1,
                                        nbinsx=50,
                                        marker_color="grey",
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            )
        else:
            self.root_widget = v.ExpansionPanel(
                children=[
                    v.ExpansionPanelHeader( # 0 
                        class_="font-weight-bold blue lighten-4",
                        variant="outlined",
                        children=[
                            "Another variable"
                        ]
                    ),
                    v.ExpansionPanelContent( # 1
                        children=[
                            v.Col( # 10
                                class_="ma-3 pa-3",
                                children=[
                                    v.Spacer(), # 100
                                    v.RangeSlider(  # 101 
                                        class_="ma-3",
                                        v_model=[
                                            -1,
                                            1,
                                        ],
                                        min=-5,
                                        max=5,
                                        step=0.1,
                                        thumb_label="always",
                                    ),
                                ],
                            ),
                            FigureWidget(
                                data=[
                                    Histogram(
                                        x=pd.DataFrame(
                                            np.random.randint(
                                                0,
                                                100,
                                                size=(
                                                    100,
                                                    4,
                                                ),
                                            ),
                                            columns=list(
                                                "ABCD"
                                            ),
                                        ),
                                        bingroup=1,
                                        nbinsx=50,
                                        marker_color="grey",
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            )
        
        if self.rule.is_categorical_rule():
            change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} possible values :")
            # We use a multiple select widget
            select_widget = v.Select(
                label=self.rule.variable.symbol,
                items=self.rule.cat_values,
                style_="width: 150px",
                multiple=True,
                )
        # Rules on continuous variables :
        if not self.rule.is_inner_interval_rule():
            if self.rule.min is None: # var < max
                change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} lesser than {'or equal to ' if self.rule.operator_max == 0 else ''}:")
                select_widget = v.Slider(
                    class_="ma-3",
                    v_model=[self.rule.max],
                    min=-5, # TODO : easy to set : min(var)
                    max=self.rule.max*1.5, # TODO : easy to set : max(var)
                    step=0.1, # TODO we could divide the spread by 50 ?
                    thumb_label="always"
                    )
            else: # var > min
                change_widget(self.root_widget, "00", f"{self.rule.variable.symbol} greater than {'or equal to ' if self.rule.operator_min == 4 else ''}:")
                select_widget = v.Slider(
                    class_="ma-3",
                    v_model=[self.rule.min],
                    min=-self.rule.min*1.5, # TODO set according to the variable distribution
                    max=2,
                    step=0.1, # TODO set according to the variable distribution
                    thumb_label="always"
                    )
        else: # We represent an intervel rule
            if self.rule.is_inner_interval_rule():
                select_widget = v.RangeSlider(
                    class_="ma-3",
                    v_model=[
                        self.rule.min,
                        self.rule.max,
                    ],
                    # min=-5,
                    # max=5,
                    step=0.1,
                    thumb_label="always"
                    )
            else: # We have a outter interval rule
                select_widget = v.RangeSlider(
                    class_="ma-3",
                    v_model=[
                        self.rule.min,
                        self.rule.max,
                    ],
                    # min=-5,
                    # max=5,
                    step=0.1,
                    thumb_label="always"
                )

        select_widget.on_event("change", self.widget_value_changed)
        change_widget(self.root_widget, "101", select_widget)

        self.figure = FigureWidget(
            data=[
                Histogram(
                    x=self.X[self.rule.variable.symbol], 
                    bingroup=1, 
                    nbinsx=50, 
                    marker_color="grey")
            ]
        )
        self.figure.add_trace(
            Histogram(
                x=self.X[self.rule.variable.symbol],
                bingroup=1,
                nbinsx=50,
                marker_color="LightSkyBlue",
                opacity=0.6,
            )
        )
        self.figure.add_trace(
            Histogram(
                x=self.X[self.rule.variable.symbol], 
                bingroup=1, 
                nbinsx=50, 
                marker_color="blue"
                )
        )

        self.figure.update_layout(
            barmode="overlay",
            bargap=0.1,
            width=400,
            # width=0.9 * int(fig_size),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=150,
        )

        change_widget(self.root_widget, "11", self.figure)

    def widget_value_changed(self, widget, event, data):
        if isinstance(widget, v.Select):
            logger.debug(f"RW.widget_value_changed : data = {data}")
        elif isinstance(widget, v.Slider):
            # Simple rule
            logger.debug(f"RW.widget_value_changed : data = {data}")
        else: # Interval rule
            logger.debug(f"RW.widget_value_changed : data = {data}")


class RulesWidget:
    """
    A RulesWidget is a piece of GUI that allows the user to refine a set of rules.
    The user can use the slider to change the rules.
    There are 2 RW : VS and ES slides

    rules_db : a dict of list :
        [[rules_list, scores]]], so that at iteration i we have :
        [i][0] : the list of rules
        [i][1] : the list of scores
    current_index : int refers to rules_db
    X : pd.DataFrame, values or explanations Dataframe depending on the context
    variables : list of Variable
    is_value_space : bool
    rules_updated : callable of the GUI parent
    root_wiget : its widget representation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        variables: list,
        values_space: bool,
        rules_updated: callable,
    ):
        
        self.X = X
        self.variables = variables
        self.is_value_space = values_space
        self.rules_updated = rules_updated

        # UI:
        self.root_widget = widgets.VBox(
            [
                v.Col( # 0 
                    children=[
                        v.Row( # 00
                            class_="ml-4",
                            children=[
                                v.Icon(children=["mdi-target"]), # 000
                                v.Html(class_="ml-3", tag="h2", children=[f"No rule to display for the {'VS' if self.is_value_space else 'ES'} space"]), # 001
                            ]
                            ),
                        v.Col( # 01
                            elevation=10,
                            children=[ 
                                v.Html( # 010
                                    class_="ml-3", 
                                    tag="p", 
                                    children=[
                                        "Precision = n/a, Recall = n/a, f_score = n/a"
                                        ]
                                ), 
                                v.DataTable( # 011 
                                        v_model=[],
                                        show_select=False,
                                        headers=[{"text": column, "sortable": False, "value": column } for column in ['Variable$$', 'Unit', 'Desc', 'Critical', 'Rule']],
                                        items=[], 
                                        hide_default_footer=True,
                                        disable_sort=True,
                                    ),
                                ]
                            )
                        ]
                ),
                v.ExpansionPanels() # 1 / We insert RuleWidgets here
            ]
        )

        self.rule_widget_list = []

    def set_rules(self, rules_list: list, score_list: list):
        """
        Called by the GUI when a new set of rules is computed
        """
        # We populate our db
        self._put_in_db(rules_list, score_list, True)

        self.set_rules_info()
        self.set_rule_widgets()

        indexes_for_HDE = Rule.rules_to_indexes(self.get_current_rule_list(), self.X)

        self.rules_updated(self, indexes_for_HDE)

    def _put_in_db(self, rules_list: list, score_list: list, erase_past:bool=False):
        if erase_past:
            self.rules_db = {}
            self.current_index = -1
        self.current_index += 1
        self.rules_db[self.current_index] = [rules_list, score_list]

    def _dump_db(self) -> str:
        txt = ""
        for i in range(len(self.rules_db)):
            rules_list = self.rules_db[i][0]
            scores_list = self.rules_db[i][1]
            txt = txt + f"({i}) : {len(rules_list)} rules:\n"
            for rule in rules_list:
                txt = txt + f"    {rule}\n"
            txt = txt + f"   scores = {scores_list}\n"
        return txt

    def set_rule_widgets(self):
        """
        Creates RuleWidgest, one per rule
        """

        # We remove existing RuleWidgets
        for i in reversed(range(len(self.rule_widget_list))):
            self.rule_widget_list[i].close()
            self.rule_widget_list.pop(i)
    
        # We set new RuleWidget list in our ExpansionPanels children
        get_widget(self.root_widget, "1").children = [RuleWidget(rule, self.X, self.is_value_space).root_widget for rule in self.get_current_rule_list()]

    def set_rules_info(self):
        """
        Sets scores and rule details int the DataTable
        """
        # We set the title
        if self.get_current_rule_list() is None or len(self.get_current_rule_list()) == 0:
            title = f"No rule to display for the {'VS' if self.is_value_space else 'ES'} space"
        else:
            title = f"Rule(s) applied to the {'values' if self.is_value_space else 'explanations'} space"
        change_widget(self.root_widget, "001", v.Html(class_="ml-3", tag="h2", children=[title]))

        # We set the scores
        if (
            self.get_current_score_list() is None
            or len(self.get_current_score_list()) == 0
        ):
            scores_txt = "Precision = n/a, recall = n/a, f1_score = n/a"
        else:
            scores_txt = f"Precision : {self.get_current_score_list()['precision']}, recall : {self.get_current_score_list()['recall']}, f1_score : {self.get_current_score_list()['f1']}"
        change_widget(self.root_widget, "010", v.Html(class_="ml-3", tag="p", children=[scores_txt]))

        # We set the rules in the DataTable
        change_widget(self.root_widget, "011", v.DataTable(
                v_model=[],
                show_select=False,
                headers=[{"text": column, "sortable": False, "value": column } for column in ['Variable', 'Unit', 'Desc', 'Critical', 'Rule']],
                items=Rule.rules_to_dict_list(self.get_current_rule_list()),
                hide_default_footer=True,
                disable_sort=True,
            )
        )

    def add_rules(self, rule_list: list, score_lsit):
        pass

    def get_current_rule_list(self):
        return self.rules_db[self.current_index][0]

    def get_current_score_list(self):
        return self.rules_db[self.current_index][1]

    def hide_beeswarm(self, hide: bool):
        # We retrieve the beeswarmGrp (VBox)
        get_widget(self.root_widget, "0101").disabled = hide

    def skope_slider_changed(*change):
        # we just call skope_changed @GUI
        self.skope_changed()

    def redraw_both_graphs(self):
        # We update the refiner's histogram :
        with get_widget(self.root_widget, "01001").batch_update():
            get_widget(self.vbox_widget, "01001").data[
                0
            ].x = self._ds.get_full_values()[
                self._selection.get_vs_rules()[self._variable.get_col_index][2]
            ]

            # We update the refiner's beeswarm :
            # get_widget(self.root_widget,"01011").v_model : # TODO Why do we check ?
            with get_widget(self.vbox_widget, "01011").batch_update():
                pass
                

    def skope_rule_changed(widget, event, data):
        pass
        

    def get_class_selector(
        self, min: int = 1, max: int = -1, fig_size: int = 700
    ) -> v.Layout:
        valuesList = list(set(self._gui.get_dataset().getVariableValue(self._variable)))
        widgetList = []
        for value in valuesList:
            if value <= max and value >= min:
                inside = True
            else:
                inside = False
            widget = v.Checkbox(
                class_="ma-4",
                v_model=inside,
                label=str(value).replace("_", " "),
            )
            widgetList.append(widget)
        row = v.Row(class_="ml-6 ma-3", children=widgetList)
        text = v.Html(
            tag="h3",
            children=["Select the values of the feature " + self._variable.getSymbol()],
        )
        return v.Layout(
            class_="d-flex flex-column align-center justify-center",
            style_="width: " + str(int(fig_size) - 70) + "px; height: 303px",
            children=[v.Spacer(), text, row],
        )

    def real_time_changed(*args):
        """If changed, we invert the validate button"""
        get_widget(
            self.root_widget, "0010020"
        ).disabled = not get_widget(self.root_widget, "0010020").disabled

        # See realTimeUpdateCheck (0010021)
        get_widget(self.root_widget, "0010021").on_event(
            "change", real_time_changed
        )

    def beeswarm_color_changed(*args):
        """If changed, we invert the showScake value"""
        # See beeswarm :
        show_scale = (
            get_widget(self.root_widget, "01011").data[0].marker[showscale]
        )
        show_scale = get_widget(self.root_widget, "01011").update_traces(
            marker=dict(showscale=not show_scale)
        )

        # See bsColorChoice[,v.Switch] (0010101)
        self._widgetGraph.get_widget("010101").on_event(
            "change", beeswarm_color_changed
        )

    def continuous_check_changed(widget, event, data):
        features = [
            self._selection.getVSRules()[i][2]
            for i in range(len(self._selection.getVSRules()))
        ]
        aSet = []
        for i in range(len(features)):
            if features[i] not in aSet:
                aSet.append(features[i])

        index = features.index(aSet[2])
        if widget.v_model:
            # TODO : understand
            # We define accordion (0010) children as histoCtrl (00100) + list (accordion(0010).children[1])
            self._widget.get_widget("010").children = [
                self._widget.get_widget("0100")
            ] + list(self._widget.get_widget("010").children[1:])
            count = 0
            for i in range(len(self._gui.get_selection().getVSRules())):
                if (
                    self._gui.get_selection().getVSRules()[i - count][2]
                    == self._selection.getVSRules()[index][2]
                    and i - count != index
                ):
                    self._gui.get_selection().getVSRules().pop(i - count)
                    count += 1
            # We set skopeSlider (0010001) values
            self.selection.getVSRules()[index][0] = get_widget(
                self.root_widget, "010001"
            ).v_model[0]
            self.selection.getVSRules()[index][4] = get_widget(
                self.root_widget, "010001"
            ).v_model[1]

            self._skope_list = create_rule_card(self.selection.ruleListToStr())
        else:
            class_selector = self.get_class_selector()
            get_widget(self.root_widget, "010").children = [
                class_selector
            ] + list(get_widget(self.root_widget, "010").children[1:])
            aSet = []
            for i in range(len(self.get_class_selector().children[2].children)):
                if class_selector.children[2].children[i].v_model:
                    aSet.append(int(class_selector.children[2].children[i].label))
            if len(aSet) == 0:
                widget.v_model = True
                return
            column = deepcopy(self._gui.get_selection().getVSRules()[index][2])
            count = 0
            for i in range(len(self._gui.get_selection().getVSRules())):
                if self._gui.get_selection().getVSRules()[i - count][2] == column:
                    self._gui.get_selection().getVSRules().pop(i - count)
                    count += 1
            ascending = 0
            for item in aSet:
                self.selection.getVSRules().insert(
                    index + ascending, [item - 0.5, "<=", column, "<=", item + 0.5]
                )
                ascending += 1
            self._skope_list = create_rule_card(
                self._gui.get_selection().ruleListToStr()
            )

        # We wire the "change" event on the isContinuousChck (001021)
        get_widget(self.root_widget, "01021").on_event(
            "change", continuous_check_changed
        )
