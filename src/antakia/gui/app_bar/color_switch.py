from functools import partial

import ipyvuetify as v

from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class ColorSwitch:

    def __init__(self, data_store: DataStore, update_callback):
        self.update_callback = update_callback
        self.data_store = data_store
        self.btn_list = ["y", "y^", "residual", "all_regions"]
        self.icon_tooltip_dict = {"y": ("mdi-alpha-y-circle-outline", 'Display target values'),
                                  "y^": ("mdi-alpha-y-circle", 'Display predicted values'),
                                  "residual": ("mdi-delta", 'Display residual values'),
                                  "residual_sub": ("mdi-set-right", "Display residual values of the substitute model"),
                                  "all_regions": ("mdi-view-dashboard", 'Display regions')}

        self._build_widget()

    def _build_widget(self):
        self.widget = v.Col(children=[v.BtnToggle(  # 11
            class_="mr-3",
            mandatory=False,
            disabled=False,
            # v_model="Y",
            children=[
                v.Tooltip(  # 110
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1100
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[self.btn_list[0]][0]])],
                                value=self.btn_list[0],
                                v_model=True,
                            ),
                    }],
                    children=['Display target values']),
                v.Tooltip(  # 111
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1110
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[self.btn_list[1]][0]])],
                                value=self.btn_list[1],
                                v_model=True,
                            ),
                    }],
                    children=['Display predicted values']),
                v.Tooltip(  # 112
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1120
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[self.btn_list[2]][0]])],
                                value=self.btn_list[2],
                                v_model=True,
                            ),
                    }],
                    children=['Display residual values']),
                v.Tooltip(  # 112
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1130
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[self.btn_list[3]][0]])],
                                value=self.btn_list[3],
                                v_model=True,
                            ),
                    }],
                    children=['Display regions']),

            ],
        )])

        self.widget.children[0].on_event("change", self.update_callback)

    def update_btn_widget(self, value, btn_list):
        # Updates the button in the switch if the value parameter is one of the buttons,
        # else it will disable all buttons

        if btn_list != self.btn_list:
            self.btn_list = btn_list

            self.widget.children[0].children = [
                v.Tooltip(  # 110
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1100
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[btn_list[0]][0]])],
                                value=btn_list[0],
                                v_model=True,
                            ),
                    }],
                    children=['Display target values']),
                v.Tooltip(  # 111
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1110
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[btn_list[1]][0]])],
                                value=btn_list[1],
                                v_model=True,
                            ),
                    }],
                    children=['Display predicted values']),
                v.Tooltip(  # 112
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1120
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[btn_list[2]][0]])],
                                value=btn_list[2],
                                v_model=True,
                            ),
                    }],
                    children=['Display residual values']),
                v.Tooltip(  # 112
                    bottom=True,
                    v_slots=[{
                        'name':
                            'activator',
                        'variable':
                            'tooltip',
                        'children':
                            v.Btn(  # 1130
                                v_on='tooltip.on',
                                icon=True,
                                children=[v.Icon(children=[self.icon_tooltip_dict[btn_list[3]][0]])],
                                value=btn_list[3],
                                v_model=True,
                            ),
                    }],
                    children=['Display regions']),
            ]

        self.widget.children[0].v_model = value
