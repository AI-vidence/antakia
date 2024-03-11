import pandas as pd
import ipyvuetify as v
from plotly.graph_objects import FigureWidget, Histogram, Box

from antakia_core.data_handler.rules import Rule, RuleSet

from antakia_core.utils.utils import compute_step, get_mask_comparison_color, boolean_mask
from antakia_core.utils.variable import DataVariables, Variable

from antakia.gui.graphical_elements.rule_slider import RuleSlider
from antakia.utils.stats import log_errors


class RuleWidget:
    """
    A piece of UI handled by a RulesExplorer
    It allows the user to modify one rule

    Attributes :
    """

    def __init__(self, rule: Rule, X: pd.DataFrame, values_space: bool, rule_updated_callback: callable):
        '''

        Parameters
        ----------
        rule : the rule to be dsplayed
        X : training dataset
        y : target variable
        values_space : bool is value space ?
        init_selection_mask : reference selection mask
        init_rules_mask : reference rules mask
        selectable_mask : list of point that could be selected using the current rule
        rule_updated_callback : callable called on update
        '''
        self.rule: Rule = rule
        self.X: pd.DataFrame = X
        self.X_col = X.loc[:, rule.variable.column_name]
        self.values_space: bool = values_space
        self.rule_updated_callback: callable = rule_updated_callback
        self.display_sliders: bool = self.values_space  # enable rule edit
        self.widget = None
        self.init_mask = boolean_mask(X, True)
        self.rule_mask = boolean_mask(X, True)
        self.selectable_mask = boolean_mask(X, True)
        self.type = 'auto'
        self._resolve_type()

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

        self.title = v.ExpansionPanelHeader(
            class_="blue lighten-4",
            children=[self._get_panel_title()]
        )

        # root_widget is an ExpansionPanel
        self.widget: v.ExpansionPanel = v.ExpansionPanel(
            children=[
                self.title,
                v.ExpansionPanelContent(
                    children=[
                        v.Col(
                            children=[
                                self.select_widget,
                                self.figure],
                        )
                    ]
                ),
            ]
        )

        # The variable name bg (ExpansionPanelHeader) is light blue
        # get_widget(self.root_widget, "0").class_ = "blue lighten-4"

    def _resolve_type(self):
        if self.type == 'auto':
            if self.X_col.nunique() > 15:
                self.type = 'swarm'
            else:
                self.type = 'histogram'

    def _build_figure(self):
        """
        draw the histograms
        Returns
        -------

        """
        mask_color, colors_info = get_mask_comparison_color(self.rule_mask, self.init_mask)
        if self.type == 'histogram':
            base_args = {
                'bingroup': 1,
                'nbinsx': 50,
            }
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
        else:
            swarm_plots = []
            for name, color in colors_info.items():
                fig = Box({
                    'alignmentgroup': 'True',
                    'boxpoints': 'all',
                    'fillcolor': 'rgba(255,255,255,0)',
                    'hoveron': 'points',
                    'hovertemplate': f'match={name}<br>{self.X_col.name}' + '=%{x}<extra></extra>',
                    'jitter': 1,
                    'legendgroup': name,
                    'line': {'color': 'rgba(255,255,255,0)'},
                    'marker': {'color': color},
                    'name': name,
                    'offsetgroup': name,
                    'orientation': 'h',
                    'pointpos': 0,
                    'showlegend': True,
                    'x': self.X_col[mask_color == color],
                    'x0': ' ',
                    'xaxis': 'x',
                    'y': self.selectable_mask[mask_color == color],
                    'y0': ' ',
                    'yaxis': 'y'
                })
                # fig.update_yaxes(showticklabels=False)
                swarm_plots.append(fig)
            self.figure = FigureWidget(
                data=swarm_plots
            )
            self.figure.update_layout({
                'boxgap': 0,
                'boxmode': 'overlay',
                'legend': {'title': {'text': None}},
                'margin': {'t': 60},
                'xaxis': {'showticklabels': True, 'title': {'text': name}},
                'yaxis': {'showticklabels': False, 'title': {'text': 'selectable'}}
            })
            # data = pd.DataFrame([self.X_col, mask_color.replace({v: k for k, v in colors_info.items()})],
            #                     index=[self.X_col.name, 'color']).T
            #
            # fig = px.strip(data, x=self.X_col.name, color="color", stripmode='overlay', color_discrete_map=colors_info)
            # fig = fig.update_layout(boxgap=0).update_traces(jitter=1)
            # self.figure = FigureWidget(fig)

    def _get_panel_title(self):
        """
        compute and display accordion title
        Returns
        -------

        """
        return repr(self.rule)

    def _update_panel_title(self):
        self.title.children = [self._get_panel_title()]

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
            range_min = float(self.X[self.rule.variable.column_name].min())
            range_max = float(self.X[self.rule.variable.column_name].max())
            range_min, range_max, step = compute_step(range_min, range_max)
            range_min = range_min - step
            current_min, current_max = self._get_select_widget_values()

            self.slider = RuleSlider(
                range_min,
                range_max,
                step,
                value_min=current_min,
                value_max=current_max,
                change_callback=self._widget_value_changed
            )

            return self.slider.widget
        else:
            return v.Col()

    def _get_select_widget_values(self) -> tuple[float | None, float | None] | list[str]:
        """
        sets the selection values
        Returns
        -------

        """
        if self.rule.is_categorical_rule:
            return self.rule.cat_values
        elif self.rule.rule_type == 1:  # var < max
            return None, self.rule.max
        elif self.rule.rule_type == 2:  # var > min
            return self.rule.min, None
        else:
            return self.rule.min, self.rule.max

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
            min_, max_ = data
        new_rule = Rule(
            self.rule.variable,
            min_,
            self.rule.includes_min,
            max_,
            self.rule.includes_max,
            cat_values
        )
        self.rule = new_rule
        self._update_panel_title()
        self.rule_updated_callback(new_rule)

    def reinit_rule(self, rule: Rule, init_mask: pd.Series):
        """
        edits the rule of the widget and changes reference selection
        warning : does not update the graph, should be used in conjunction with update
        Parameters
        ----------
        rule
        init_mask

        Returns
        -------

        """
        assert rule.variable == self.rule.variable
        self.rule = rule
        self.init_mask = init_mask

    def update(self, new_rules_mask: pd.Series, selectable_mask: pd.Series, rule: Rule = None):
        """
            used to update the display (sliders and histogram) to match the new rule
            (called from outside th object to synchronize it)
        """

        if rule is not None:
            self.rule = rule
        self.rule_mask = new_rules_mask
        self.selectable_mask = selectable_mask
        self.update_figure()

    def update_figure(self):
        if self.display_sliders:
            min_val, max_val = self._get_select_widget_values()
            self.slider.set_value(min_val, max_val)
        mask_color, colors_info = get_mask_comparison_color(self.rule_mask, self.init_mask)
        with self.figure.batch_update():
            for i, color in enumerate(colors_info.values()):
                self.figure.data[i].x = self.X_col[mask_color == color]
                self.figure.data[i].y = self.selectable_mask[mask_color == color]


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
        self.rule_widget_list: dict[Variable, RuleWidget] = {}

        # At startup, we are disabled :
        self.is_disabled = True

        self._build_widget()
        self._create_rule_widgets()

    # ---------------- widget management -------------------- #
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
        # We set new RuleWidget list and put it in our ExpansionPanels children
        if self.X is not None:
            for variable in self.variables.variables.values():
                self.rule_widget_list[variable] = RuleWidget(Rule(variable), self.X, self.is_value_space,
                                                             self.update_rule)
            self.rules_widgets.children = [rw.widget for rw in self.rule_widget_list.values()]
        else:
            self.rules_widgets.children = []

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
            children=[rw.widget for rw in self.rule_widget_list.values()]
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

        # TODO order rules

    def _show_title(self):
        if len(self.current_rules_set) == 0 or self.is_disabled:
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
            len(self.current_rules_set) == 0
            or self.is_disabled
        ):
            rules_txt = "N/A"
            css = "ml-7 grey--text"
        else:
            rules_txt = repr(self.current_rules_set)
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
        self._put_in_db(rules_set, score_dict)

        rule_mask = self.current_rules_set.get_matching_mask(self.X)
        selectable_masks = self._get_selectable_masks()  # for rule, mask_wo_rule in mask_wo_rule:
        self.disable(False)
        # We populate the db

        # We reinit the RuleWidgets
        for var, rule_wgt in self.rule_widget_list.items():
            rule = rules_set.get_rule(var)
            if rule is not None:
                rule_wgt.reinit_rule(rule, self.init_selection_mask)
            else:
                rule_wgt.reinit_rule(Rule(var), self.init_selection_mask)
        # we update the displays
        self.update_rule_widgets()
        # we display
        self.refresh_widget()

    def update_rule_widgets(self):
        rule_mask = self.current_rules_set.get_matching_mask(self.X)
        selectable_masks = self._get_selectable_masks()  # for rule, mask_wo_rule in mask_wo_rule:
        for var, rule_wgt in self.rule_widget_list.items():
            rule_wgt.update(rule_mask, selectable_masks.get(var, rule_mask))

    def _get_selectable_masks(self) -> dict[Variable, pd.Series]:
        """
        computes for each rule the selection mask, ignoring the rule
        Returns
        -------

        """
        res = {}
        for rule in self.current_rules_set.rules.values():
            rules_minus_r = [r for r in self.current_rules_set.rules.values() if r != rule]
            mask = RuleSet(rules_minus_r).get_matching_mask(self.X)
            res[rule.variable] = mask
        return res

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
        for var, rule_wgt in self.rule_widget_list.items():
            rule_wgt.reinit_rule(Rule(var), boolean_mask(self.X, True))
        self.init_selection_mask = None
        self.refresh_widget()

    def update_rule(self, new_rule: Rule):
        """
        callback to synchronize rules in the widget
        called by the edition of a single rule
        """
        # We update the rule in the db
        new_rules_set = self.current_rules_set.copy()
        new_rules_set.replace(new_rule)

        new_rules_mask = new_rules_set.get_matching_mask(self.X)
        self.update_from_mask(new_rules_mask, new_rules_set)

    def update_from_mask(self, new_rules_mask: pd.Series, new_rules_set: RuleSet = None, sync=True):
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
        if new_rules_set is not None:
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
        self.update_rule_widgets()

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
        rules_mask = self.current_rules_set.get_matching_mask(self.X)

        # We update each of our RuleWidgets
        self.update_rule_widgets()

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
    def current_rules_set(self) -> RuleSet:
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
