from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import ipyvuetify as v
from antakia_core.data_handler import Rule
from antakia_core.utils import boolean_mask, get_mask_comparison_color, compute_step
from plotly.graph_objs import Histogram, FigureWidget, Box

from antakia import config
from antakia.gui.graphical_elements.rule_slider import RuleSlider
from antakia.utils.other_utils import NotInitialized
from antakia.utils.stats import log_errors


class RuleWidget:
    """
    A piece of UI handled by a RulesExplorer
    It allows the user to modify one rule

    Attributes :
    """

    def __init__(self, rule: Rule, X: pd.DataFrame, values_space: bool,
                 rule_updated_callback: Callable):
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
        self.X_col = X.loc[:, rule.variable.column_name]
        self.values_space: bool = values_space
        self.rule_updated_callback: Callable = partial(rule_updated_callback,
                                                       self, 'updated')
        self.display_sliders: bool = self.values_space  # enable rule edit
        self.widget = None
        self.init_mask = boolean_mask(X, True)
        self.rule_mask = boolean_mask(X, True)
        self.selectable_mask = boolean_mask(X, True)
        self._display_mask = None
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

        self.title = v.ExpansionPanelHeader(class_="blue lighten-4",
                                            children=[self._get_panel_title()])

        # root_widget is an ExpansionPanel
        self.widget = v.ExpansionPanel(children=[
            self.title,
            v.ExpansionPanelContent(
                children=[v.Col(children=[self.select_widget, self.figure], )
                          ]),
        ])

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
        mask_color, colors_info = self._get_colors()
        if self.type == 'histogram':
            base_args = {
                'bingroup': 1,
                'nbinsx': 50,
            }
            h = []
            for name, color in colors_info.items():
                h.append(
                    Histogram(name=name,
                              x=self.X_col[self.display_mask
                                           & (mask_color == color)],
                              marker_color=color,
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
                fig = self.get_swarm_plot(color, mask_color, name)
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

    def get_swarm_plot(self, color, mask_color, name):
        box = Box({
            'alignmentgroup':
            'True',
            'boxpoints':
            'all',
            'fillcolor':
            'rgba(255,255,255,0)',
            'hoveron':
            'points',
            'hovertemplate':
            f'match={name}<br>{self.X_col.name}' + '=%{x}<extra></extra>',
            'jitter':
            1,
            'legendgroup':
            name,
            'line': {
                'color': 'rgba(255,255,255,0)'
            },
            'marker': {
                'color': color
            },
            'name':
            name,
            'offsetgroup':
            name,
            'orientation':
            'h',
            'pointpos':
            0,
            'showlegend':
            True,
            'x':
            self.X_col[self.display_mask & (mask_color == color)],
            'x0':
            ' ',
            'xaxis':
            'x',
            'y':
            self.selectable_mask[self.display_mask & (mask_color == color)],
            'y0':
            ' ',
            'yaxis':
            'y'
        })
        return box

    def _get_panel_title(self):
        """
        compute and display accordion title
        Returns
        -------

        """
        return repr(self.rule)

    def _update_panel_title(self):
        if self.rule.rule_type == -1:
            self.title.class_ = "grey lighten-4"
        else:
            self.title.class_ = "blue lighten-4"
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
        if rule.rule_type < 0:
            self.idx = None

        self.rule = rule
        self.init_mask = init_mask

    def update(self,
               new_rules_mask: pd.Series,
               selectable_mask: pd.Series,
               rule: Rule = None):
        """
            used to update the display (sliders and histogram) to match the new rule
            (called from outside th object to synchronize it)
        """

        if rule is not None:
            self.rule = rule
        self.rule_mask = new_rules_mask
        self.selectable_mask = selectable_mask

        self._update_panel_title()
        self.update_figure()

    def update_figure(self):
        if self.display_sliders:
            min_val, max_val = self._get_select_widget_values()
            self.slider.set_value(min_val, max_val)
        mask_color, colors_info = self._get_colors()
        with self.figure.batch_update():
            for i, color in enumerate(colors_info.values()):
                self.figure.data[i].x = self.X_col[self.display_mask
                                                   & (mask_color == color)]
                self.figure.data[i].y = self.selectable_mask[
                    self.display_mask & (mask_color == color)]

    def _get_colors(self):
        if self.init_mask.all() or not self.init_mask.any():
            mask_color, colors_info = get_mask_comparison_color(
                self.rule_mask, self.rule_mask)
        else:
            mask_color, colors_info = get_mask_comparison_color(
                self.rule_mask, self.init_mask)
        return mask_color, colors_info

    @property
    def display_mask(self) -> pd.Series:
        """
        mask should be applied on each display (x,y,z,color, selection)
        """
        if self.X is None:
            raise NotInitialized()
        if self._display_mask is None:
            limit = config.ATK_MAX_DOTS
            if len(self.X) > limit:
                self._mask = pd.Series([False] * len(self.X),
                                       index=self.X.index)
                indices = np.random.choice(self.X.index,
                                           size=limit,
                                           replace=False)
                self._mask.loc[indices] = True
            else:
                self._mask = pd.Series([True] * len(self.X),
                                       index=self.X.index)
        return self._mask
