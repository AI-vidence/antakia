import ipyvuetify as v
import numpy as np
import pandas as pd
from plotly.graph_objs import FigureWidget
from prometheus_client import Histogram

from antakia.gui.widgets import get_widget


class tab1:
    def __init__(self):

        pass

    def build_widget(self):
        self.widget = v.TabItem(  # Tab 1) Selection # 43
            class_="mt-2",
            children=[
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
                                        v.Btn(  # 43010 Skope button
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
                                        ),
                                }
                            ],
                            children=['Find a rule to match the selection']
                        ),
                        v.Btn(  # 4302
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
                        ),
                        v.Tooltip(  # 4303
                            bottom=True,
                            v_slots=[
                                {
                                    'name': 'activator',
                                    'variable': 'tooltip',
                                    'children':
                                        v.Btn(  # 43030 Skope button
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
                                        ),
                                }
                            ],
                            children=['Promote current rules as a region']
                        ),
                    ]
                ),  # End Buttons row
                v.Row(  # tab 1 / row #2 : 2 RulesWidgets # 431
                    class_="d-flex flex-row",
                    children=[
                        v.Col(  # placeholder for the VS RulesWidget (RsW) # 4310
                            children=[
                                v.Col(  # 43100 / 0
                                    children=[
                                        v.Row(  # 431000 / 00
                                            children=[
                                                v.Icon(children=["mdi-target"]),  #
                                                v.Html(class_="ml-3", tag="h2",
                                                       children=[
                                                           "Rules applied on the values space"]),
                                            ]
                                        ),
                                        v.Html(  # 431001 / 01
                                            class_="ml-7",
                                            tag="li",
                                            children=[
                                                "Precision = n/a, recall = n/a, f1_score = n/a"
                                            ]
                                        ),
                                        v.Html(  # 431002 / 02
                                            class_="ml-7",
                                            tag="li",
                                            children=[
                                                "N/A"
                                            ]
                                        ),
                                    ]
                                ),
                                v.ExpansionPanels(  # Holds VS RuleWidgets  # 43101 / 1
                                    style_="max-width: 95%",
                                    children=[
                                        v.ExpansionPanel(  # PH for VS RuleWidget #431010 10
                                            children=[
                                                v.ExpansionPanelHeader(  # 0 / 100
                                                    class_="blue lighten-4",
                                                    children=[
                                                        "A VS rule variable"  # 1000
                                                    ]
                                                ),
                                                v.ExpansionPanelContent(  # 1
                                                    children=[
                                                        v.Col(
                                                            children=[
                                                                v.Spacer(),
                                                                v.RangeSlider(
                                                                    # class_="ma-3",
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
                                                            data=[  # Dummy histogram
                                                                Histogram(
                                                                    x=pd.Series(
                                                                        np.random.normal(0, 1,
                                                                                         100) * 2,
                                                                        name='x'),
                                                                    bingroup=1,
                                                                    nbinsx=20,
                                                                    marker_color="grey",
                                                                ),
                                                            ],
                                                            layout={
                                                                'height': 300,
                                                                'margin': {'t': 0, 'b': 0,
                                                                           'l': 0,
                                                                           'r': 0},
                                                                'width': 600
                                                            }
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ],
                        ),
                        v.Col(  # placeholder for the ES RulesWidget (RsW) # 4311
                            size_="width=50%",
                            children=[
                                v.Col(  # placeholder for the ES RulesWidget card # 43110
                                    children=[
                                        v.Row(  # 431100
                                            children=[
                                                v.Icon(children=["mdi-target"]),
                                                v.Html(class_="ml-3", tag="h2", children=[
                                                    "Rules applied on the explanations space"]),
                                            ]
                                        ),
                                        v.Html(  # 431101
                                            class_="ml-7",
                                            tag="li",
                                            children=[
                                                "Precision = n/a, Recall = n/a, F1 = n/a"]
                                        ),
                                        v.Html(  # 431102
                                            class_="ml-7",
                                            tag="li",
                                            children=[
                                                "N/A"
                                            ]
                                        )
                                    ]
                                ),
                                v.ExpansionPanels(  # 43111
                                    style_="max-width: 95%",
                                    children=[
                                        v.ExpansionPanel(  # Placeholder for the ES RuleWidgets
                                            children=[
                                                v.ExpansionPanelHeader(  # 0
                                                    class_="blue lighten-4",
                                                    # variant="outlined",
                                                    children=[
                                                        "An ES rule variable"  # 00
                                                    ]
                                                ),
                                                v.ExpansionPanelContent(  # #
                                                    children=[
                                                        v.Col(
                                                            children=[
                                                                v.Spacer(),
                                                                v.RangeSlider(
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
                                                        FigureWidget(  # Dummy histogram
                                                            data=[
                                                                Histogram(
                                                                    x=pd.Series(
                                                                        np.random.normal(0, 1,
                                                                                         100) * 2,
                                                                        name='x'),
                                                                    bingroup=1,
                                                                    nbinsx=20,
                                                                    marker_color="grey",
                                                                ),
                                                            ],
                                                            layout={
                                                                'height': 300,
                                                                'margin': {'t': 0, 'b': 0,
                                                                           'l': 0,
                                                                           'r': 0},
                                                                'width': 600
                                                            }
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ],
                        ),
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
                                                for column in dummy_df.columns
                                            ],
                                            items=dummy_df.to_dict(
                                                "records"
                                            ),
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
        )

    def refresh_buttons_tab_1(self):
        self.disable_hde()
        # data table
        get_widget(self.widget, "20").disabled = bool(self.selection_mask.all())
        # skope_rule
        get_widget(self.widget, "010").disabled = not self.new_selection or bool(self.selection_mask.all())
        # undo
        get_widget(self.widget, "02").disabled = not (self.vs_rules_wgt.rules_num > 1)
        # validate rule
        get_widget(self.widget, "030").disabled = not (self.vs_rules_wgt.rules_num > 0)

    def compute_skope_rules(self, *args):
        self.new_selection = False

        if self.tab != 1:
            self.select_tab(1)
        # compute skope rules
        skr_rules_list, skr_score_dict = skope_rules(self.selection_mask, self.vs_hde.current_X, self.variables)
        skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
        # init vs rules widget
        self.vs_rules_wgt.init_rules(skr_rules_list, skr_score_dict, self.selection_mask)
        # update VS and ES HDE
        self.vs_hde.figure.display_rules(
            selection_mask=self.selection_mask,
            rules_mask=skr_rules_list.get_matching_mask(self.X)
        )
        self.es_hde.figure.display_rules(
            selection_mask=self.selection_mask,
            rules_mask=skr_rules_list.get_matching_mask(self.X)
        )

        es_skr_rules_list, es_skr_score_dict = skope_rules(self.selection_mask, self.es_hde.current_X, self.variables)
        es_skr_score_dict['target_avg'] = self.y[self.selection_mask].mean()
        self.es_rules_wgt.init_rules(es_skr_rules_list, es_skr_score_dict, self.selection_mask)
        self.refresh_buttons_tab_1()
        self.select_tab(1)

    def undo_rules(self, *args):
        if self.tab != 1:
            self.select_tab(1)
        if self.vs_rules_wgt.rules_num > 0:
            self.vs_rules_wgt.undo()
        else:
            # TODO : pourquoi on annule d'abord le VS puis l'ES?
            self.es_rules_wgt.undo()
        self.refresh_buttons_tab_1()

    def validate_rules(self, *args):
        if self.tab != 1:
            self.select_tab(1)

        rules_list = self.vs_rules_wgt.current_rules_list
        # UI rules :
        # We clear selection
        self.selection_changed(None, boolean_mask(self.X, True))
        # We clear the RulesWidget
        self.es_rules_wgt.reset_widget()
        self.vs_rules_wgt.reset_widget()
        if len(rules_list) == 0:
            self.vs_rules_wgt.show_msg("No rules found on Value space cannot validate region", "red--text")
            return

        # We add them to our region_set
        region = self.region_set.add_region(rules=rules_list)
        self.region_num_for_validated_rules = region.num
        # lock rule
        region.validate()

        # And update the rules table (tab 2)
        # we refresh buttons
        self.refresh_buttons_tab_1()
        # We force tab 2
        self.select_tab(2)
