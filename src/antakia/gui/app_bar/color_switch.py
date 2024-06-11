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
        # a dictionary which has btn values as keys and icons and tooltip text as value
        self.icon_tooltip_dict = {"y": ("mdi-alpha-y-circle-outline",'Display target values'),
                                  "y^": ("mdi-alpha-y-circle",'Display predicted values'),
                                  "residual": ("mdi-delta",'Display residual values'),
                                  "residual_sub": (),
                                  "all_regions": ("mdi-view-dashboard",'Display regions')}
        self._build_widget(self.btn_list)

    def _build_widget(self, btn_list):
        children_list = []
        for button in btn_list:
            children_list.append(v.Tooltip(  # 110
                bottom=True,
                v_slots=[{
                    'name':
                        'activator',
                    'variable':
                        'tooltip',
                    'children':
                        v.Btn(
                            v_on='tooltip.on',
                            icon=True,
                            children=[
                                v.Icon(children=["mdi-alpha-y-circle-outline"])#TODO remplacer
                            ],
                            value="y",#TODO remplacer
                            v_model=True,
                        ),
                }],
                children=['Display target values']),#TODO remplacer
            )


        self.widget = v.BtnToggle(
            class_="mr-3",
            mandatory=False,
            disabled=False,
            v_model="Y",
            children=children_list,
        )

        self.widget.on_event("change", self.update_callback)

    def update_btn(self, value, event, btn_list):
        # Updates the button in the switch if the value parameter is one of the buttons,
        # else it will disable all buttons
        # TODO faire la table d'équivalence entre les value et les boutons
        # TODO mettre à jour les boutons proposés
        if event == 'substitute':
            btn_list = ["y", "y^", "residual", "residual_sub"]
        self._build_widget(btn_list)
        if btn_list not in btn_list:
            self.widget.v_model = btn_value
