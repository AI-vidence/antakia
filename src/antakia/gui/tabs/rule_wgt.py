from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import ipyvuetify as v
from antakia_core.data_handler import Rule
from antakia_core.utils import boolean_mask, compute_step, timeit
from plotly.graph_objs import Histogram, FigureWidget, Box

from antakia.config import AppConfig
from antakia.gui.graphical_elements.rule_slider import RuleSlider
from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import Log
from antakia.utils.other_utils import NotInitialized
from antakia.utils.stats import log_errors


class RuleWidget:
    """
    A piece of UI handled by a RulesExplorer
    It allows the user to modify one rule

    Attributes :
    """

    def __init__(self, rule: Rule, data_store: DataStore, X: pd.DataFrame,
                 values_space: bool, rule_updated_callback: Callable,
                 _reset_expanded_callback: Callable):
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
        self.idx: float | None = None
        self.rule: Rule = rule
        self.X: pd.DataFrame = X
        self.data_store: DataStore = data_store
        self.X_col = X.loc[:, rule.variable.column_name]
        self.values_space: bool = values_space
        self.rule_updated_callback: Callable = partial(rule_updated_callback,
                                                       self, 'updated')
        self._reset_expanded_callback = partial(_reset_expanded_callback, self,
                                                'expanded_switch', {})
        self.display_sliders: bool = self.values_space  # enable rule edit
        self.widget = None
        self.init_mask = boolean_mask(X, True)
        self.selectable_mask = boolean_mask(X, True)
        self._display_mask = None
        self.type = 'auto'
        self.expanded = False
        self.edited = True
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
        self.select_widget = None
        # build figure
        self.figure = None
        self.title = v.ExpansionPanelHeader(class_="grey lighten-4",
                                            children=[self._get_panel_title()])

        # root_widget is an ExpansionPanel
        self.widget = v.ExpansionPanel(children=[
            self.title,
            v.ExpansionPanelContent(children=[]),
        ])
        self.widget.on_event('click', self.panel_changed_callback)

        # The variable name bg (ExpansionPanelHeader) is light blue
        # get_widget(self.root_widget, "0").class_ = "blue lighten-4"

    @timeit
    def _resolve_type(self):
        if self.type == 'auto':
            if self.X_col.nunique() > 15:
                self.type = 'swarm'
            else:
                self.type = 'histogram'

    @timeit
    def _build_figure(self):
        """
        draw the histograms
        Returns
        -------

        """
        _, colors_info = self._get_colors()
        if self.type == 'histogram':
            base_args = {
                'bingroup': 1,
                'nbinsx': 50,
            }
            h = []
            for name, color in colors_info.items():
                h.append(
                    Histogram(name=name, x=[], marker_color=color,
                              **base_args))
            self.figure = FigureWidget(data=h)
            self.figure.update_layout(
                barmode="stack",
                bargap=0.1,
                # width=600,
                showlegend=False,
                margin={
                    'l': 0,
                    'r': 0,
                    't': 0,
                    'b': 0
                },
                height=200,
            )
        else:
            swarm_plots = []
            for name, color in colors_info.items():
                fig = self._get_swarm_plot(color, name)
                # fig.update_yaxes(showticklabels=False)
                swarm_plots.append(fig)
            self.figure = FigureWidget(data=swarm_plots)
            self.figure.update_layout({
                'boxgap': 0,
                'boxmode': 'overlay',
                # 'legend': {
                #     'title': {
                #         'text': None
                #     }
                # }
            })
            # data = pd.DataFrame([self.X_col, mask_color.replace({v: k for k, v in colors_info.items()})],
            #                     index=[self.X_col.name, 'color']).T
            #
            # fig = px.strip(data, x=self.X_col.name, color="color", stripmode='overlay', color_discrete_map=colors_info)
            # fig = fig.update_layout(boxgap=0).update_traces(jitter=1)
            # self.figure = FigureWidget(fig)
        self.figure.update_layout({
            'showlegend': False,
            'legend': {
                'title': {
                    'text': None
                }
            },
            'margin': {
                't': 0,
                'b': 0,
                'l': 0,
                'r': 0
            },
            'xaxis': {
                'showticklabels': True,
                'title': {
                    'text': self.rule.variable.column_name
                },
                'categoryorder': 'category ascending'
            },
            'yaxis': {
                'showticklabels': False,
                'title': {
                    'text': 'selectable'
                }
            }
        })

    @timeit
    def _get_swarm_plot(self, color, name):
        box = Box({
            'alignmentgroup': 'True',
            'boxpoints': 'all',
            'fillcolor': 'rgba(255,255,255,0)',
            'hoveron': 'points',
            'hovertemplate':
            f'match={name}<br>{self.X_col.name}' + '=%{x}<extra></extra>',
            'jitter': 1,
            'legendgroup': name,
            'line': {
                'color': 'rgba(255,255,255,0)'
            },
            'marker': {
                'color': color
            },
            'name': name,
            'offsetgroup': name,
            'orientation': 'h',
            'pointpos': 0,
            'showlegend': True,
            'x': [],
            'x0': ' ',
            'xaxis': 'x',
            'y': [],
            'y0': ' ',
            'yaxis': 'y'
        })
        return box

    @timeit
    def _get_panel_title(self):
        """
        compute and display accordion title
        Returns
        -------

        """
        return repr(self.rule)

    @timeit
    def _update_panel_title(self):
        if self.rule.rule_type == -1:
            self.title.class_ = "grey lighten-4"
        else:
            self.title.class_ = "blue lighten-4"
        self.title.children = [self._get_panel_title()]

    @timeit
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
                    items=self.X[
                        self.rule.variable.column_name].unique().tolist(),
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
                change_callback=self._widget_value_changed)

            return self.slider.widget
        else:
            return v.Col()

    @timeit
    def _get_select_widget_values(
            self) -> tuple[float | None, float | None] | list[str]:
        """
        sets the selection values
        Returns
        -------

        """
        if self.rule.is_categorical_rule:
            return self.rule.cat_values

        return self.rule.min, self.rule.max

    # --------------- callbacks ------------------- #
    @log_errors
    @timeit
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
        new_rule = Rule(self.rule.variable, min_, self.rule.includes_min, max_,
                        self.rule.includes_max, cat_values)
        self.rule = new_rule

        if self.values_space:
            self.data_store.rules_mask = self.selectable_mask & self.rule(
                self.X_col)

        self._update_panel_title()
        self._update_data()
        self.rule_updated_callback(new_rule)

    @timeit
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
        if rule.rule_type < 0:
            self.idx = None

        self.rule = rule
        self.init_mask = init_mask
        self.edited = True

    @timeit
    def update(self, selectable_mask: pd.Series, rule: Rule = None):
        """
            used to update the display (sliders and histogram) to match the new rule
            (called from outside th object to synchronize it)
        """

        if rule is not None:
            self.rule = rule
        self.selectable_mask = selectable_mask
        self.edited = True

        self._update_panel_title()
        self._update_expansion_panel()

    @timeit
    def _update_expansion_panel(self):
        if self.expanded:
            if self.select_widget is None:
                self.select_widget = self._get_select_widget()
                self._build_figure()
            self.widget.children[1].children = [
                v.Col(children=[self.select_widget, self.figure])
            ]

            if self.display_sliders:
                min_val, max_val = self._get_select_widget_values()
                self.slider.set_value(min_val, max_val)
            if self.edited:
                self._update_data()
        else:
            self.widget.children[1].children = []

    @timeit
    def _update_data(self):
        mask_color, colors_info = self._get_colors()
        with self.figure.batch_update():
            for i, color in enumerate(colors_info.values()):
                to_display = self.data_store.display_mask & (mask_color
                                                             == color)
                self.figure.data[i].x = self.X_col[to_display]
                self.figure.data[i].y = self.selectable_mask[to_display]
        self.edited = False

    @timeit
    def _get_colors(self):
        mask_color = self.data_store.rule_selection_color
        colors_info = self.data_store.rule_selection_color_legend
        return mask_color, colors_info

    @timeit
    def panel_changed_callback(self, *args):
        with Log('panel_changed_callback', 2):
            self.expanded = not self.expanded
            self._reset_expanded_callback()
            self._update_expansion_panel()
