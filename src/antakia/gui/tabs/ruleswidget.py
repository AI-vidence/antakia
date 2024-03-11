import pandas as pd

import ipyvuetify as v
from plotly.graph_objects import FigureWidget, Histogram

from antakia_core.data_handler.rules import Rule, RuleSet

from antakia_core.utils.utils import compute_step, get_mask_comparison_color
from antakia_core.utils.variable import DataVariables

from antakia.utils.stats import log_errors


class RuleWidget:
    """
    A piece of UI handled by a RulesExplorer
    It allows the user to modify one rule

    Attributes :
    """

    def __init__(self, rule: Rule, X: pd.DataFrame, y, values_space: bool, init_selection_mask: pd.Series,
                 init_rules_mask: pd.Series,
                 rule_updated: callable):
        '''

        Parameters
        ----------
        rule : the rule to be dsplayed
        X : training dataset
        y : target variable
        values_space : bool is value space ?
        init_selection_mask : reference selection mask
        init_rules_mask : reference rules mask
        rule_updated : callable called on update
        '''
        self.rule: Rule = rule
        self.X: pd.DataFrame = X
        self.X_col = X.loc[:, rule.variable.column_name]
        self.values_space: bool = values_space
        self.rule_updated: callable = rule_updated
        self.display_sliders: bool = self.values_space  # enable rule edit
        self.widget = None
        self.init_mask = init_selection_mask
        self.rule_mask = init_rules_mask

        self._build_widget()

    # --------------- build widget ------------------- #
    def _build_widget(self):
        """
        build the widget
        Returns
        -------

        """
        # build slider
        self.select_widget = self._get_select_widget()
        # build figure
        self._build_figure()

        title = self._get_panel_title()

        # root_widget is an ExpansionPanel
        self.widget: v.ExpansionPanel = v.ExpansionPanel(
            children=[
                v.ExpansionPanelHeader(
                    class_="blue lighten-4",
                    children=[title]
                ),
                v.ExpansionPanelContent(
                    children=[
                        v.Col(
                            children=[v.Spacer(), self.select_widget],
                        ),
                        self.figure,
                    ]
                ),
            ]
        )

        # The variable name bg (ExpansionPanelHeader) is light blue
        # get_widget(self.root_widget, "0").class_ = "blue lighten-4"

    def _build_figure(self):
        """
        draw the histograms
        Returns
        -------

        """
        base_args = {
            'bingroup': 1,
            'nbinsx': 50,
        }
        mask_color, colors_info = get_mask_comparison_color(self.rule_mask, self.init_mask)
        h = []
        for name, color in colors_info.items():
            h.append(Histogram(
                name=name,
                x=self.X_col[mask_color == color],
                marker_color=color,
                **base_args
            ))
        self.figure = FigureWidget(
            data=h
        )
        self.figure.update_layout(
            barmode="stack",
            bargap=0.1,
            # width=600,
            showlegend=False,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            height=200,
        )

    def _get_panel_title(self):
        """
        compute and display accordion title
        Returns
        -------

        """
        if self.rule.is_categorical_rule:
            title = f"{self.rule.variable.display_name} possible values :"
        elif self.rule.rule_type == 1:  # var < max
            title = f"{self.rule.variable.display_name} lesser than {'or equal to ' if self.rule.includes_max else ''}:"
        elif self.rule.rule_type == 2:  # var > min
            title = f"{self.rule.variable.display_name} greater than {'or equal to ' if self.rule.includes_min else ''}:"
        elif self.rule.rule_type == 3:
            title = f"{self.rule.variable.display_name} inside the interval:"
        else:
            title = f"{self.rule.variable.display_name} outside the interval:"
        return title

    def _get_select_widget(self):
        """
        builds the widget to edit the rule
        Returns
        -------

        """
        if self.display_sliders:

            if self.rule.is_categorical_rule:
                return v.Select(
                    label=self.rule.variable.display_name,
                    items=self.X[self.rule.variable.column_name].unique().tolist(),
                    style_="width: 150px",
                    multiple=True,
                )
            min_ = float(self.X[self.rule.variable.column_name].min())
            max_ = float(self.X[self.rule.variable.column_name].max())
            min_, max_, step = compute_step(min_, max_)
            min_ = min_ - step
            slider_args = {
                'min': min_,
                'max': max_,
                'step': step,
                'thumb_label': "always",
                'thumb_size': 30,
                'thumb_color': 'blue'
            }
            if self.rule.rule_type == 1:  # var < max
                slider_class = v.Slider
                slider_args['color'] = 'green'
                slider_args['track_color'] = 'red'
            elif self.rule.rule_type == 2:  # var > min
                slider_class = v.Slider
                slider_args['color'] = 'red'
                slider_args['track_color'] = 'green'
            elif self.rule.is_inner_interval_rule:
                slider_class = v.RangeSlider
                slider_args['color'] = 'green'
                slider_args['track_color'] = 'red'
            else:
                slider_class = v.RangeSlider
                slider_args['color'] = 'red'
                slider_args['track_color'] = 'green'

            slider = slider_class(**slider_args)

            slider.v_model = self._get_select_widget_values()
            slider.on_event("change", self._widget_value_changed)
            return slider
        else:
            return v.Col()

    def _get_select_widget_values(self):
        """
        sets the selection values
        Returns
        -------

        """
        if self.rule.is_categorical_rule:
            return self.rule.cat_values
        elif self.rule.rule_type == 1:  # var < max
            return [self.rule.max],
        elif self.rule.rule_type == 2:  # var > min
            return [self.rule.min]
        else:
            return [self.rule.min, self.rule.max]

    # --------------- callbacks ------------------- #
    @log_errors
    def _widget_value_changed(self, widget, event, data):
        """
        callback called when the user edits a value (called by the widget)

        should fire rule updated callback

        Parameters
        ----------
        widget
        event
        data : new value

        Returns
        -------

        """
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
        new_rule = Rule(
            self.rule.variable,
            min_,
            self.rule.includes_min,
            max_,
            self.rule.includes_max,
            cat_values
        )
        self.rule = new_rule
        self.rule_updated(new_rule)

    def update(self, new_rules_mask: pd.Series = None, rule: Rule = None):
        """ 
            used to update the display (sliders and histogram) to match the new rule
            (called from outside th object to synchronize it)
        """

        if rule is not None:
            self.rule = rule
        if new_rules_mask is not None:
            self.rule_mask = new_rules_mask
        # We update the selects
        if self.display_sliders:
            self._get_select_widget_values()
        mask_color, colors_info = get_mask_comparison_color(self.rule_mask, self.init_mask)
        with self.figure.batch_update():
            for i, color in enumerate(colors_info.values()):
                self.figure.data[i].x = self.X_col[mask_color == color]


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
        """
        widget to manage rule edition and display
        Parameters
        ----------
        X: pd.Dataframe reference dataset
        y: pd.Series test dataset
        variables
        values_space: is the widget on the value space
        new_rules_defined: callback on rule update
        """
        self.X = X
        self.y = y
        self.variables: DataVariables = variables
        self.init_selection_mask = None
        self.is_value_space = values_space
        self.new_rules_defined = new_rules_defined

        self.rules_db: list[tuple[RuleSet, dict]] = []
        self.rule_widget_list: list[RuleWidget] = []

        # At startup, we are disabled :
        self.is_disabled = True

        self._build_widget()

    # ---------------- widget management -------------------- #

    def _build_widget(self):
        self.title = "not initialized"
        self.stats = "not initialized"
        self.rules = 'N/A'
        self.region_stat_card = v.Col(  # 43100 / 0
            children=[
                v.Row(  # 431000 / 00
                    children=[
                        v.Icon(children=["mdi-target"]),  #
                        v.Html(class_="ml-3", tag="h2",
                               children=[self.title]),
                    ]
                ),
                v.Html(  # 431001 / 01
                    class_="ml-7",
                    tag="li",
                    children=[
                        self.stats
                    ]
                ),
                v.Html(  # 431002 / 02
                    class_="ml-7",
                    tag="li",
                    children=[self.rules]
                ),
            ]
        )
        self.rules_widgets = v.ExpansionPanels(  # Holds VS RuleWidgets  # 43101 / 1
            style_="max-width: 95%",
            children=[rw.widget for rw in self.rule_widget_list]
        )
        self.widget = v.Col(  # placeholder for the VS RulesWidget (RsW) # 4310
            children=[self.region_stat_card, self.rules_widgets],
        )
        self.refresh_widget()
        self.disable()

    def set_title(self, title: str, css: str):
        self.title = title
        title_wgt = self.region_stat_card.children[0].children[1]
        title_wgt.children = [self.title]
        title_wgt.class_ = css

    def set_stats(self, stats: str, css: str):
        self.stats = stats

        stats_wgt = self.region_stat_card.children[1]
        stats_wgt.children = [self.stats]
        stats_wgt.class_ = css

    def set_rules(self, rules: str, css: str):
        self.rules = rules

        rules_wgt = self.region_stat_card.children[2]
        rules_wgt.children = [self.rules]
        rules_wgt.class_ = css

    def set_rules_widgets(self, rule_widget_list: list[RuleWidget]):
        self.widget.children[1].children = [rule_widget.widget for rule_widget in rule_widget_list]

    def _create_rule_widgets(self):
        """
        creates all rule widgets
                Called by self.init_rules
                If init_rules_mask is None, we clear all our RuleWidgets

        Parameters
        ----------
        init_mask : reference_mask

        Returns
        -------

        """
        init_rule_mask = self.current_rules_list.get_matching_mask(self.X)
        # We set new RuleWidget list and put it in our ExpansionPanels children
        if len(self.current_rules_list) == 0:
            self.rule_widget_list = []
        else:
            self.rule_widget_list = [
                RuleWidget(rule, self.X, self.y, self.is_value_space,
                           self.init_selection_mask, init_rule_mask,
                           self.update_rule)
                for rule in self.current_rules_list.rules.values()
            ]
        self.set_rules_widgets(self.rule_widget_list)

    # ------------- widget macro method ------------- #
    def disable(self, is_disabled: bool = True):
        """
        disables (True) or enables (False) the widget
        Disabled : card in grey with dummy text + no ExpansionPanels
        Enabled : card in light blue / ready to display RuleWidgets
        """

        self.is_disabled = is_disabled

        self.refresh_widget()

    def refresh_widget(self):
        """
        Sets macro widget info and Datatable: scores and rule details int the DataTable
        Returns
        -------

        """
        # We set the title
        self._show_title()

        # We set the scores
        self._show_score()

        # We set the rules
        self._show_rules()

        self._show_rules_wgt()

    def _show_title(self):
        if len(self.current_rules_list) == 0 or self.is_disabled:
            title = f"No rule to display for the {'VS' if self.is_value_space else 'ES'} space"
            css = "ml-3 grey--text" if self.is_disabled else "ml-3"
        else:
            title = f"Rule(s) applied to the {'values' if self.is_value_space else 'explanations'} space"
            css = "ml-3"
        self.set_title(title, css)

    def _show_score(self):
        """
        show rule score
        Returns
        -------

        """
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
        self.set_stats(scores_txt, css)

    def _show_rules(self):
        """
        show rules as text
        Returns
        -------

        """
        if (
            len(self.current_rules_list) == 0
            or self.is_disabled
        ):
            rules_txt = "N/A"
            css = "ml-7 grey--text"
        else:
            rules_txt = repr(self.current_rules_list)
            css = "ml-7 blue--text"
        self.set_rules(rules_txt, css)

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
        self.set_rules(msg, css)

    def _show_rules_wgt(self):
        if self.is_disabled:
            self.set_rules_widgets([])
        else:
            self.set_rules_widgets(self.rule_widget_list)

    # ---------------- update logic -------------------------- #
    @property
    def space(self):
        return 'values' if self.is_value_space else 'explanations'

    def update_X(self, X):
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
        self._create_rule_widgets()
        self.refresh_widget()

    def init_rules(self, rules_set: RuleSet, score_dict: dict, selection_mask: pd.Series):
        """
        initialize the widget with the rule list, the score dict (text displayed) and
        the reference_mask (selection_mask)
        """
        # we start wih a fresh widget
        self.reset_widget()

        self.init_selection_mask = selection_mask
        if len(rules_set) > 0:
            # self.show_msg("No rules found", "red--text")
            # return
            self.disable(False)
            # We populate the db
            self._put_in_db(rules_set, score_dict)

            # We create the RuleWidgets
            self._create_rule_widgets()
        # we display
        self.refresh_widget()

    def reset_widget(self):
        """
        a reseted widget is
        - disabled
        - history is erased
        - reference mask is None
        Returns
        -------

        """
        self.disable()
        self.rules_db = []
        self.rule_widget_list = []
        self.init_selection_mask = None
        self.refresh_widget()

    def update_rule(self, new_rule: Rule):
        """
        callback to synchronize rules in the widget
        called by the edition of a single rule
        """
        # We update the rule in the db
        new_rules_set = self.current_rules_list.copy()
        new_rules_set.add(new_rule)

        new_rules_mask = new_rules_set.get_matching_mask(self.X)
        self.update_from_mask(new_rules_mask, new_rules_set)

    def update_from_mask(self, new_rules_mask: pd.Series, new_rules_set: RuleSet, sync=True):
        """
        updates the widget with the new rule_mask and rule list - the reference_mask is kept for comparison
        Parameters
        ----------
        new_rules_mask : mask to display
        new_rules_list : rules to base widget on
        sync : whether or not to call the sync callback (self.new_rules_defined) to update the other rules widget

        Returns
        -------

        """
        if len(new_rules_set):
            # update rules
            try:
                precision = (new_rules_mask & self.init_selection_mask).sum() / new_rules_mask.sum()
                recall = (new_rules_mask & self.init_selection_mask).sum() / self.init_selection_mask.sum()
                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                precision = recall = f1 = -1
            target_avg = self.y[new_rules_mask].mean()

            new_score_dict = {"precision": precision, "recall": recall, "f1": f1, 'target_avg': target_avg}

            self._put_in_db(new_rules_set, new_score_dict)

        # We refresh the widget macro info
        self.refresh_widget()

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(new_rules_mask)

        # We notify the GUI and tell there are new rules to draw
        if self.new_rules_defined is not None and sync:
            self.new_rules_defined(self, new_rules_mask)

    def undo(self):
        """
        Restore the previous rules
        """
        # We remove last rules item from the db:
        if len(self.rules_db) > 1:
            self.rules_db.pop(-1)

        # We compute again the rules mask
        rules_mask = self.current_rules_list.get_matching_mask(self.X)

        def find_rule(rule_widget: RuleWidget) -> Rule:
            var = rule_widget.rule.variable
            return self.current_rules_list.find_rule(var)

        # We update each of our RuleWidgets
        for rw in self.rule_widget_list:
            rw.update(rules_mask, find_rule(rw))

        # we refresh the widget macro info
        self.refresh_widget()
        # We notify the GUI and tell there are new rules to draw if necessary
        if self.new_rules_defined is not None:
            self.new_rules_defined(self, rules_mask)

    def _put_in_db(self, rule_set: RuleSet, score_dict: dict):
        """
        add the rule list to memory
        Parameters
        ----------
        rule_set
        score_dict

        Returns
        -------

        """
        self.rules_db.append((rule_set, score_dict))

    def _dump_db(self) -> str:
        """
        print db
        Returns
        -------

        """
        txt = ""
        for i in range(len(self.rules_db)):
            rules_list = self.rules_db[i][0]
            scores_dict = self.rules_db[i][1]

            txt += f"({i}) : {len(rules_list)} rules:"
            "\n    ".join(rules_list)
            txt += f"\n   scores = {scores_dict}\n"
        return txt

    @property
    def current_rules_list(self) -> RuleSet:
        """
        get the current rule list
        Returns
        -------

        """
        if len(self.rules_db) == 0:
            return RuleSet()
        else:
            return self.rules_db[-1][0]

    @property
    def current_scores_dict(self) -> dict:
        """
        get the current score dict
        Returns
        -------

        """
        if len(self.rules_db) == 0:
            return {}
        else:
            return self.rules_db[-1][1]

    @property
    def rules_num(self) -> int:
        """
        get the size of the db
        Returns
        -------

        """
        return len(self.rules_db)
