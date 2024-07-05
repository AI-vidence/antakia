from functools import partial
from typing import List

import ipyvuetify as v
from ipyvuetify import BtnToggle

from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class ColorSwitch:

    def __init__(self, data_store: DataStore, update_callback):
        self.color_update_callback = update_callback
        self.data_store = data_store
        self.btn_list = ["y", "y^", "residual", "all_regions"]

        self.btn_dict = {"y": ("mdi-alpha-y-circle-outline",  # icon
                               'Display target values',  # tootltip
                               ['y' ]),  # list of colors related to the btn
                         "y^": ("mdi-alpha-y-circle",
                                'Display predicted values',
                                ['y^']),
                         "residual": ("mdi-delta",
                                      'Display residual values',
                                      ['residual']),
                         "y^model": ("mdi-alpha-y-box-outline",
                                     "Display predicted values of the substitute model",
                                     ['y^model']),
                         "residual_sub": ("mdi-delta",
                                          "Display residual values of the substitute model",
                                          ['residual_sub']),
                         "all_regions": ("mdi-view-dashboard",
                                         'Display regions',
                                         ['all_regions', "region_selection"])}

        self._build_widget()

    def _build_toggle(self, btn_list, icon_dict) -> list[BtnToggle]:
        """

        Parameters
        ----------
        btn_list : list of buttons to display
        icon_dict : dict containing btns and their matching icon, tooltip

        Returns
        BtnToggle Widget
        -------

        """
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
        return [btn_toggle]

    def _build_widget(self):
        self.widget = v.Col(children=[])
        self.widget.children = self._build_toggle(self.btn_list, self.btn_dict)
        self.widget.children[0].on_event("change", self.color_update_callback)

    def update_btn_widget(self, btn_list):
        # Updates the button in the switch if the value parameter is one of the buttons,
        # else it will disable all buttons

        if btn_list != self.btn_list:  # update the button list
            self.btn_list = btn_list
            self.widget.children = self._build_toggle(btn_list, self.btn_dict)  # updates the toggle with new btns
            self.widget.children[0].on_event("change", self.update_callback)

        for btn_value, colors in self.btn_dict.items():
            if self.data_store.color in colors[2]:
                self.widget.children[0].v_model = btn_value
