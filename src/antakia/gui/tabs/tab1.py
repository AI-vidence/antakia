from functools import partial
from typing import Callable

import pandas as pd
from antakia_core.compute.skope_rule.skope_rule import skope_rules
import ipyvuetify as v
from antakia_core.data_handler import Region
from antakia_core.data_handler import RuleSet
from antakia_core.utils import format_data
from antakia_core.utils.variable import DataVariables

from antakia.gui.tabs.ruleswidget import RulesWidget
from antakia.utils.stats import log_errors, stats_logger


class Tab1:
    EDIT_RULE = 'edition'
    CREATE_RULE = 'creation'

    def __init__(self, variables: DataVariables, update_callback: Callable,
                 validate_rules_callback: Callable, X: pd.DataFrame,
                 X_exp: pd.DataFrame | None, y: pd.Series):
        self.selection_changed = False
        self.region = Region(X)
        self.reference_mask = self.region.mask
        self.update_callback = partial(update_callback, self, 'rule_updated')
        self.validate_rules_callback = partial(validate_rules_callback, self,
                                               'rule_validated')

        self.X = X
        self.X_exp = X_exp
        self.y = y

        self.variables = variables
        self.vs_rules_wgt = RulesWidget(self.X, self.y, self.variables, True,
                                        self.new_rules_defined)
        self.es_rules_wgt = RulesWidget(self.X_exp, self.y, self.variables,
                                        False)

        self._build_widget()

    def _build_widget(self):
        self.find_rules_btn = v.Btn(  # 43010 Skope button
            v_on='tooltip.on',
            class_="ma-1 primary white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-axis-arrow"],
                ),
                "Find rules",
            ],
        )
        self.cancel_btn = v.Btn(  # 4302
            class_="ma-1",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-close-circle-outline"],
                ),
                "Cancel",
            ],
        )
        self.undo_btn = v.Btn(  # 4302
            class_="ma-1",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-undo"],
                ),
                "Undo",
            ],
        )
        self.validate_btn = v.Btn(  # 43030 validate
            v_on='tooltip.on',
            class_="ma-1 green white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-check"],
                ),
                "Validate rules",
            ],
        )
        self.title_wgt = v.Row(children=[
            v.Html(  # 44000
                tag="h3",
                class_="ml-2",
                children=["Creating new region :"],
            )
        ])
        self.selection_status_str_1 = v.Html(  # 430000
            tag="strong",
            children=["0 points"]
            # 4300000 # selection_status_str_1
        )
        self.selection_status_str_2 = v.Html(  # 43001
            tag="li",
            children=["0% of the dataset"]
            # 430010 # selection_status_str_2
        )
        self.data_table = v.DataTable(  # 432010
            v_model=[],
            show_select=False,
            headers=[{
                "text": column,
                "sortable": True,
                "value": column,
            } for column in self.X.columns],
            items=[],
            hide_default_footer=False,
            disable_sort=False,
        )
        self.widget = [
            self.title_wgt,
            v.Row(  # buttons row # 430
                class_="d-flex flex-row align-top mt-2",
                children=[
                    v.Sheet(  # Selection info # 4300
                        class_="ml-3 mr-3 pa-2 align-top grey lighten-3",
                        style_="width: 20%",
                        elevation=1,
                        children=[
                            v.Html(  # 43000
                                tag="li",
                                children=[self.selection_status_str_1]),
                            self.selection_status_str_2
                        ],
                    ),
                    v.Tooltip(  # 4301
                        bottom=True,
                        v_slots=[{
                            'name': 'activator',
                            'variable': 'tooltip',
                            'children': self.find_rules_btn
                        }],
                        children=['Find a rule to match the selection']),
                    self.cancel_btn,
                    self.undo_btn,
                    v.Tooltip(  # 4303
                        bottom=True,
                        v_slots=[{
                            'name': 'activator',
                            'variable': 'tooltip',
                            'children': self.validate_btn
                        }],
                        children=['Promote current rules as a region']),
                ]),  # End Buttons row
            v.Row(  # tab 1 / row #2 : 2 RulesWidgets # 431
                class_="d-flex flex-row",
                children=[self.vs_rules_wgt.widget,
                          self.es_rules_wgt.widget],  # end Row
            ),
            v.
            ExpansionPanels(  # tab 1 / row #3 : datatable with selected rows # 432
                class_="d-flex flex-row",
                children=[
                    v.
                    ExpansionPanel(  # 4320 # is enabled or disabled when no selection
                        children=[
                            v.ExpansionPanelHeader(  # 43200
                                class_="grey lighten-3",
                                children=["Data selected"]),
                            v.ExpansionPanelContent(  # 43201
                                children=[self.data_table], ),
                        ])
                ],
            ),
        ]
        # get_widget(self.widget[2], "0").disabled = True  # disable datatable
        # We wire the click events
        self.find_rules_btn.on_event("click", self.compute_skope_rules)
        self.undo_btn.on_event("click", self.undo_rules)
        self.cancel_btn.on_event("click", self.cancel_edit)
        self.validate_btn.on_event("click", self.validate_rules)
        self.refresh_buttons()

    @property
    def valid_selection(self):
        return self.reference_mask.any() and not (self.reference_mask.all())

    def refresh_selection_status(self):
        if self.valid_selection:
            selection_status_str_1 = f"{self.reference_mask.sum()} point selected"
            selection_status_str_2 = f"{100 * self.reference_mask.mean():.2f}% of the  dataset"
        else:
            selection_status_str_1 = f"0 point selected"
            selection_status_str_2 = f"0% of the  dataset"
        self.selection_status_str_1.children = [selection_status_str_1]
        self.selection_status_str_2.children = [selection_status_str_2]

    @property
    def edit_type(self):
        return self.CREATE_RULE if self.region.num == -1 else self.EDIT_RULE

    def _update_title_txt(self):
        if self.edit_type == 'creation':
            self.title_wgt.children = [
                v.Html(  # 44000
                    tag="h3",
                    class_="ml-2",
                    children=["Creating new region :"],
                )
            ]
        else:
            region_prefix_wgt = v.Html(class_="mr-2",
                                       tag="h3",
                                       children=["Editing Region"])  # 450000
            region_chip_wgt = v.Chip(
                color=self.region.color,
                children=[str(self.region.num)],
            )
            self.title_wgt.children = [
                v.Sheet(  # 45000
                    class_="ma-1 d-flex flex-row align-center",
                    children=[region_prefix_wgt, region_chip_wgt])
            ]

    def reset(self):
        self.selection_changed = False
        region = Region(self.X)
        self.update_region(region)

    def update_region(self, region: Region):
        self.region = region
        self._update_title_txt()
        self.vs_rules_wgt.change_rules(region.rules, region.mask, True)
        self.es_rules_wgt.change_rules(RuleSet(), region.mask, True)
        self.update_reference_mask(region.mask)
        if self.valid_selection:
            self.new_rules_defined(self, 'region_update', region.mask)

    def update_reference_mask(self, reference_mask):
        self.reference_mask = reference_mask
        self.selection_changed = self.valid_selection
        X_rounded = self.X.loc[reference_mask].copy().apply(format_data)
        self.data_table.items = X_rounded.to_dict("records")
        self.refresh_selection_status()
        self.vs_rules_wgt.update_reference_mask(self.reference_mask)
        self.es_rules_wgt.update_reference_mask(self.reference_mask)
        self.refresh_buttons()

    def update_X_exp(self, X_exp: pd.DataFrame):
        self.X_exp = X_exp
        self.es_rules_wgt.change_underlying_dataframe(X_exp)

    def refresh_buttons(self):
        empty_rule_set = len(self.vs_rules_wgt.current_rules_set) == 0
        empty_history = self.vs_rules_wgt.history_size <= 1

        # data table
        self.data_table.disabled = not self.valid_selection
        # self.widget[2].children[0].disabled = not self.valid_selection
        self.find_rules_btn.disabled = not self.valid_selection or not self.selection_changed
        self.undo_btn.disabled = empty_history
        self.cancel_btn.disabled = empty_rule_set and empty_history

        has_modif = (self.vs_rules_wgt.history_size > 1) or (
            self.es_rules_wgt.history_size == 1
            and self.region.num < 0  # do not validate a empty modif
        )
        self.validate_btn.disabled = not has_modif or empty_rule_set

    @log_errors
    def compute_skope_rules(self, *args):
        self.selection_changed = False
        # compute es rules for info only
        es_skr_rules_set, _ = skope_rules(self.reference_mask, self.X_exp,
                                          self.variables)
        self.es_rules_wgt.change_rules(es_skr_rules_set, self.reference_mask,
                                       False)
        # compute rules on vs space

        skr_rules_set, skr_score_dict = skope_rules(self.reference_mask,
                                                    self.X, self.variables)
        skr_score_dict['target_avg'] = self.y[self.reference_mask].mean()
        # init vs rules widget
        self.vs_rules_wgt.change_rules(skr_rules_set, self.reference_mask,
                                       False)
        # update widgets and hdes
        self.new_rules_defined(self,
                               'skope_rule',
                               rules_mask=skr_rules_set.get_matching_mask(
                                   self.X))
        self.refresh_buttons()
        stats_logger.log('find_rules', skr_score_dict)

    @log_errors
    def undo_rules(self, *args):
        if self.vs_rules_wgt.history_size > 0:
            self.vs_rules_wgt.undo()
        else:
            self.es_rules_wgt.undo()
        self.refresh_buttons()

    @log_errors
    def cancel_edit(self, *args):
        self.update_region(Region(self.X))
        self.update_callback(selection_mask=self.reference_mask,
                             rules_mask=self.reference_mask)
        self.refresh_buttons()

    @log_errors
    def new_rules_defined(self, caller, event: str, rules_mask: pd.Series):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        stats_logger.log('rule_changed')
        # We sent to the proper HDE the rules_indexes to render :
        self.update_callback(selection_mask=self.reference_mask,
                             rules_mask=rules_mask)

        # sync selection between rules_widgets
        if caller != self.vs_rules_wgt:
            self.vs_rules_wgt.update_rule_mask(rules_mask, sync=False)
        if caller != self.es_rules_wgt:
            self.es_rules_wgt.update_rule_mask(rules_mask, sync=False)

        self.refresh_buttons()

    @log_errors
    def validate_rules(self, *args):
        stats_logger.log('validate_rules')
        # get rule set and check validity
        rules_set = self.vs_rules_wgt.current_rules_set
        if len(rules_set) == 0:
            stats_logger.log('validate_rules', info={'error': 'invalid rules'})
            self.vs_rules_wgt.show_msg(
                "No rules found on Value space cannot validate region",
                "red--text")
            return

        # we persist the rule set in the region
        self.region.update_rule_set(rules_set)
        # we ship the region to GUI to synchronize other tab
        self.validate_rules_callback(self.region)
        # we reset the tab
        self.reset()
