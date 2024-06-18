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


    def _build_toggle(self, btn_list, icon_dict) -> v.Col:
        btn_toggle = v.BtnToggle(
            class_="mr-3",
            mandatory=False,
            disabled=False,
            children=[]
        )
        btn_widget_list = []
        for btn in btn_list:
            btn_widget_list.append(v.Tooltip(  # 110
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
                            children=[v.Icon(children=[icon_dict[btn][0]])],
                            value=btn,
                            v_model=True,
                        ),
                }],
                children=[icon_dict[btn][1]]))
        btn_toggle.children = btn_widget_list
        return v.Col(children=[btn_toggle])

    def _build_widget(self):
        self.widget = self._build_toggle(self.btn_list, self.icon_tooltip_dict)
        self.widget.children[0].on_event("change", self.update_callback)

    def update_btn_widget(self, value, btn_list):
        # Updates the button in the switch if the value parameter is one of the buttons,
        # else it will disable all buttons

        if btn_list != self.btn_list:
            self.btn_list = btn_list

            self.widget = self._build_toggle(btn_list, self.icon_tooltip_dict)

        self.widget.children[0].v_model = value
