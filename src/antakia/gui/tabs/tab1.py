from functools import partial
from typing import Callable

import pandas as pd
from antakia_core.compute.skope_rule.skope_rule import skope_rules
import ipyvuetify as v
from antakia_core.data_handler import Region
from antakia_core.data_handler import RuleSet
from antakia_core.utils import format_data, timeit, boolean_mask

from antakia.gui.helpers.data import DataStore
from antakia.gui.tabs.ruleswidget import RulesWidget
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class Tab1:
    EDIT_RULE = 'edition'
    CREATE_RULE = 'creation'

    def __init__(self, data_store: DataStore, update_callback: Callable,
                 validate_rules_callback: Callable):
        self.data_store = data_store
        self._region = Region(self.data_store.X)
        self.update_callback = partial(update_callback, self, 'rule_updated')
        self.validate_rules_callback = partial(validate_rules_callback, self,
                                               'rule_validated')

        self.X_rounded = None

        self.vs_rules_wgt = RulesWidget(self.data_store, True,
                                        self.refresh_callback)
        self.es_rules_wgt = RulesWidget(self.data_store, False)

        self._build_widget()

    def _build_widget(self):
        self.find_rule_progress = v.ProgressCircular(class_="my-2",
                                                     color="primary",
                                                     width="6",
                                                     indeterminate=False)
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
            } for column in self.data_store.X.columns],
            items=[],
            hide_default_footer=False,
            disable_sort=False,
        )
        self.data_panel = v.ExpansionPanel(  # 4320 # is enabled or disabled when no selection
            children=[
                v.ExpansionPanelHeader(  # 43200
                    class_="grey lighten-3",
                    children=["Data selected"]),
                v.ExpansionPanelContent(  # 43201
                    children=[self.data_table], ),
            ])
        self._data_panel_expanded = False
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
                    self.find_rule_progress,
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
                children=[self.data_panel],
            ),
        ]
        # get_widget(self.widget[2], "0").disabled = True  # disable datatable
        # We wire the click events
        self.find_rules_btn.on_event("click", self.compute_skope_rules)
        self.undo_btn.on_event("click", self.undo_rules)
        self.cancel_btn.on_event("click", self.cancel_edit)
        self.validate_btn.on_event("click", self.validate_rules)
        self.data_panel.on_event('click', self.data_panel_changed)
        self._refresh_buttons()

    @timeit
    def initialize(self):
        self.vs_rules_wgt.initialize()
        self.es_rules_wgt.initialize()

    @property
    def _valid_selection(self) -> bool:
        return not self.data_store.empty_selection

    @property
    def edit_type(self) -> str:
        return self.CREATE_RULE if self._region.num == -1 else self.EDIT_RULE

    def _refresh_selection_stat_card(self):
        if self._valid_selection:
            selection_status_str_1 = f"{self.data_store.selection_mask.sum()} point selected"
            selection_status_str_2 = f"{100 * self.data_store.selection_mask.mean():.2f}% of the  dataset"
        else:
            selection_status_str_1 = f"0 point selected"
            selection_status_str_2 = f"0% of the  dataset"
        self.selection_status_str_1.children = [selection_status_str_1]
        self.selection_status_str_2.children = [selection_status_str_2]

    def _refresh_title_txt(self):
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
                color=self._region.color,
                children=[str(self._region.num)],
            )
            self.title_wgt.children = [
                v.Sheet(  # 45000
                    class_="ma-1 d-flex flex-row align-center",
                    children=[region_prefix_wgt, region_chip_wgt])
            ]

    def _refresh_data_table(self):
        if self.X_rounded is None or not self._data_panel_expanded:
            self.data_table.items = []
        else:
            # TODO : loader
            if self.X_rounded is None:
                self.X_rounded = self.data_store.X.apply(format_data)
            self.data_table.items = self.X_rounded.loc[
                self.data_store.selection_mask].to_dict("records")

    @timeit
    def refresh_X_exp(self):
        self.es_rules_wgt.change_underlying_dataframe(self.data_store.X_exp)
        self.es_rules_wgt.refresh()

    def _refresh_buttons(self):
        empty_rule_set = len(self.vs_rules_wgt.current_rules_set) == 0
        empty_history = self.vs_rules_wgt.history_size <= 1

        # data table
        self.data_table.disabled = not self._valid_selection
        # self.widget[2].children[0].disabled = not self.valid_selection
        self.find_rules_btn.disabled = not self._valid_selection
        self.undo_btn.disabled = empty_history
        self.cancel_btn.disabled = empty_rule_set and empty_history

        has_modif = (self.vs_rules_wgt.history_size > 1) or (
            self.es_rules_wgt.history_size == 1
            and self._region.num < 0  # do not validate a empty modif
        )
        self.validate_btn.disabled = not has_modif or empty_rule_set

    @timeit
    def refresh(self):
        self.vs_rules_wgt.refresh()
        self.es_rules_wgt.refresh()
        self._refresh_data_table()
        self._refresh_selection_stat_card()
        self._refresh_buttons()

    @timeit
    def reset(self):
        self._region = Region(self.data_store.X)
        self.data_store.reset_rules_mask()
        self.vs_rules_wgt.change_rules(RuleSet(), True)
        self.es_rules_wgt.change_rules(RuleSet(), True)
        self.refresh()

    @timeit
    def update_region(self, region: Region):
        self._region = region
        self.data_store.rule_mask = region.mask
        self.data_store.selection_mask = region.mask

        self._refresh_title_txt()
        self.vs_rules_wgt.change_rules(region.rules, True)
        self.es_rules_wgt.change_rules(RuleSet(), True)
        self.refresh()

    # ----------- interactions -----------------#
    @log_errors
    @timeit
    def compute_skope_rules(self, *args):
        with Log('compute_skope_rules', 2):
            self.find_rule_progress.indeterminate = True
            self.find_rules_btn.disabled = True
            # compute es rules for info only
            es_skr_rules_set, _ = skope_rules(self.data_store.selection_mask,
                                              self.data_store.X_exp,
                                              self.data_store.variables)
            self.es_rules_wgt.change_rules(es_skr_rules_set, False)
            # compute rules on vs space

            skr_rules_set, skr_score_dict = skope_rules(
                self.data_store.selection_mask, self.data_store.X,
                self.data_store.variables)
            self.data_store.rules_mask = skr_rules_set.get_matching_mask(
                self.data_store.X)
            skr_score_dict['target_avg'] = self.data_store.y[
                self.data_store.selection_mask].mean()
            # init vs rules widget
            self.vs_rules_wgt.change_rules(skr_rules_set, False)
            # update widgets and hdes
            self.refresh()
            self.update_callback()
            stats_logger.log('find_rules', skr_score_dict)
            self.find_rules_btn.disabled = False
            self.find_rule_progress.indeterminate = False

    @log_errors
    @timeit
    def undo_rules(self, *args):
        with Log('undo_rules', 2):
            if self.vs_rules_wgt.history_size > 0:
                self.vs_rules_wgt.undo()
            self._refresh_buttons()

    @log_errors
    @timeit
    def cancel_edit(self, *args):
        with Log('cancel_edit', 2):
            self.reset()
            self.update_callback()

    @timeit
    @log_errors
    def refresh_callback(self, caller, event: str):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        stats_logger.log('rule_changed')
        self.refresh()
        # We sent to the proper HDE the rules_indexes to render :
        self.update_callback()

    @log_errors
    @timeit
    def validate_rules(self, *args):
        with Log('validate_rules', 2):
            stats_logger.log('validate_rules')
            # get rule set and check validity
            rules_set = self.vs_rules_wgt.current_rules_set
            if len(rules_set) == 0:
                stats_logger.log('validate_rules',
                                 info={'error': 'invalid rules'})
                self.vs_rules_wgt.show_msg(
                    "No rules found on Value space cannot validate region",
                    "red--text")
                return

            # we persist the rule set in the region
            self._region.update_rule_set(rules_set)
            # we ship the region to GUI to synchronize other tab
            self.validate_rules_callback(self._region)
            # we reset the tab
            self.reset()

    @timeit
    def data_panel_changed(self, *args):
        self._data_panel_expanded = not self._data_panel_expanded
        self._refresh_data_table()
