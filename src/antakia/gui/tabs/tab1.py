import pandas as pd
from antakia_core.compute.skope_rule.skope_rule import skope_rules
import ipyvuetify as v
from antakia_core.data_handler.region import Region
from antakia_core.data_handler.rules import RuleSet
from antakia_core.utils.utils import format_data

from antakia.gui.tabs.ruleswidget import RulesWidget
from antakia.gui.widget_utils import change_widget
from antakia.utils.stats import log_errors, stats_logger


class Tab1:
    def __init__(self, variables, update_callback, validate_rules_callback, X, X_exp, y):
        self.selection_changed = False
        self.region = Region(X)
        self.selection_mask = self.region.mask
        self.update_callback = update_callback
        self.validate_rules_callback = validate_rules_callback

        self.X = X
        self.X_exp = X_exp
        self.y = y

        self.variables = variables
        self.vs_rules_wgt = RulesWidget(self.X, self.y, self.variables, True, self.new_rules_defined)
        self.es_rules_wgt = RulesWidget(self.X_exp, self.y, self.variables, False)

        self._build_widget()

    def _build_widget(self):
        self.find_rules_btn = v.Btn(  # 43010 Skope button
            v_on='tooltip.on',
            class_="ma-1 primary white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-axis-arrow"
                    ],
                ),
                "Find rules",
            ],
        )
        self.undo_btn = v.Btn(  # 4302
            class_="ma-1",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-undo"
                    ],
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
                    children=[
                        "mdi-check"
                    ],
                ),
                "Validate rules",
            ],
        )
        self.widget = [
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
                                children=[
                                    v.Html(  # 430000
                                        tag="strong",
                                        children=["0 points"]
                                        # 4300000 # selection_status_str_1
                                    )
                                ]
                            ),
                            v.Html(  # 43001
                                tag="li",
                                children=["0% of the dataset"]
                                # 430010 # selection_status_str_2
                            )
                        ],
                    ),
                    v.Tooltip(  # 4301
                        bottom=True,
                        v_slots=[
                            {
                                'name': 'activator',
                                'variable': 'tooltip',
                                'children':
                                    self.find_rules_btn
                            }
                        ],
                        children=['Find a rule to match the selection']
                    ),
                    self.undo_btn,
                    v.Tooltip(  # 4303
                        bottom=True,
                        v_slots=[
                            {
                                'name': 'activator',
                                'variable': 'tooltip',
                                'children':
                                    self.validate_btn
                            }
                        ],
                        children=['Promote current rules as a region']
                    ),
                ]
            ),  # End Buttons row
            v.Row(  # tab 1 / row #2 : 2 RulesWidgets # 431
                class_="d-flex flex-row",
                children=[
                    self.vs_rules_wgt.widget,
                    self.es_rules_wgt.widget
                ],  # end Row
            ),
            v.ExpansionPanels(  # tab 1 / row #3 : datatable with selected rows # 432
                class_="d-flex flex-row",
                children=[
                    v.ExpansionPanel(  # 4320 # is enabled or disabled when no selection
                        children=[
                            v.ExpansionPanelHeader(  # 43200
                                class_="grey lighten-3",
                                children=["Data selected"]
                            ),
                            v.ExpansionPanelContent(  # 43201
                                children=[
                                    v.DataTable(  # 432010
                                        v_model=[],
                                        show_select=False,
                                        headers=[
                                            {
                                                "text": column,
                                                "sortable": True,
                                                "value": column,
                                            }
                                            for column in self.X.columns
                                        ],
                                        items=[],
                                        hide_default_footer=False,
                                        disable_sort=False,
                                    )
                                ],

                            ),
                        ]
                    )
                ],
            ),
        ]
        # get_widget(self.widget[2], "0").disabled = True  # disable datatable
        # We wire the click events
        self.find_rules_btn.on_event("click", self.compute_skope_rules)
        self.undo_btn.on_event("click", self.undo_rules)
        self.validate_btn.on_event("click", self.validate_rules)

    @property
    def valid_selection(self):
        return self.selection_mask.any() and not (self.selection_mask.all())

    def reset(self):
        self.selection_changed = False
        self.region = Region(self.X)
        self.selection_mask = self.region.mask

        self.vs_rules_wgt.reset_widget()
        self.es_rules_wgt.reset_widget()

    def update_region(self, region: Region):
        # TODO : compute score
        self.region = region
        self.vs_rules_wgt.init_rules(region.rules, {}, region.mask)
        self.update_selection(region.mask)
        self.new_rules_defined(None, region.mask)

    def update_selection(self, selection_mask):
        self.selection_mask = selection_mask
        self.selection_changed = self.valid_selection
        if not self.valid_selection:
            self.reset()
        else:
            # Selection is not empty anymore or changes
            X_rounded = self.X.loc[selection_mask].copy().apply(format_data)
            change_widget(
                self.widget[2],
                "010",
                v.DataTable(
                    v_model=[],
                    show_select=False,
                    headers=[{"text": column, "sortable": True, "value": column} for column in self.X.columns],
                    items=X_rounded.to_dict("records"),
                    hide_default_footer=False,
                    disable_sort=False,
                ),
            )
        self.update_selection_status()
        self.refresh_buttons()

    def update_selection_status(self):
        if self.valid_selection:
            selection_status_str_1 = f"{self.selection_mask.sum()} point selected"
            selection_status_str_2 = f"{100 * self.selection_mask.mean():.2f}% of the  dataset"
        else:
            selection_status_str_1 = f"0 point selected"
            selection_status_str_2 = f"0% of the  dataset"
        change_widget(self.widget[0], "0000", selection_status_str_1)
        change_widget(self.widget[0], "010", selection_status_str_2)

    def update_X_exp(self, X_exp: pd.DataFrame):
        self.X_exp = X_exp
        self.es_rules_wgt.update_X(X_exp)

    def refresh_buttons(self):
        # data table
        self.widget[2].children[0].disabled = not self.valid_selection
        # skope_rule
        self.find_rules_btn.disabled = not self.valid_selection or not self.selection_changed
        # undo
        self.undo_btn.disabled = not (self.vs_rules_wgt.rules_num > 1)
        # validate rule
        self.validate_btn.disabled = not (self.vs_rules_wgt.rules_num > 0)

    @log_errors
    def compute_skope_rules(self, *args):
        self.selection_changed = False
        # compute skope rules
        skr_rules_list, skr_score_dict = skope_rules(self.selection_mask, self.X, self.variables)
        skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
        # init vs rules widget
        self.vs_rules_wgt.init_rules(skr_rules_list, skr_score_dict, self.selection_mask)
        # update VS and ES HDE
        self.update_callback(selection_mask=self.selection_mask, rules_mask=skr_rules_list.get_matching_mask(self.X))

        es_skr_rules_list, es_skr_score_dict = skope_rules(self.selection_mask, self.X_exp, self.variables)
        es_skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
        self.es_rules_wgt.init_rules(es_skr_rules_list, es_skr_score_dict, self.selection_mask)
        self.refresh_buttons()
        stats_logger.log('find_rules', skr_score_dict)

    @log_errors
    def undo_rules(self, *args):
        if self.vs_rules_wgt.rules_num > 0:
            self.vs_rules_wgt.undo()
        else:
            self.es_rules_wgt.undo()
        self.refresh_buttons()

    @log_errors
    def new_rules_defined(self, rules_widget: RulesWidget | None, df_mask: pd.Series):
        stats_logger.log('rule_changed')
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        # We sent to the proper HDE the rules_indexes to render :
        self.update_callback(selection_mask=self.selection_mask, rules_mask=df_mask)

        # sync selection between rules_widgets
        if rules_widget != self.vs_rules_wgt:
            self.vs_rules_wgt.update_from_mask(df_mask, RuleSet(), sync=False)
        if rules_widget != self.es_rules_wgt:
            self.es_rules_wgt.update_from_mask(df_mask, RuleSet(), sync=False)

        self.refresh_buttons()

    @log_errors
    def validate_rules(self, *args):
        stats_logger.log('validate_rules')
        # get rule set and check validity
        rules_set = self.vs_rules_wgt.current_rules_list
        if len(rules_set) == 0:
            stats_logger.log('validate_rules', info={'error': 'invalid rules'})
            self.vs_rules_wgt.show_msg("No rules found on Value space cannot validate region", "red--text")
            return

        # we persist the rule set in the region
        self.region.update_rule_set(rules_set)
        # we ship the region to GUI to synchronize other tab
        self.validate_rules_callback(self.region)
        # we reset the tab
        self.reset()
