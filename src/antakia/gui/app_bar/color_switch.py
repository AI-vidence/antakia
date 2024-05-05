from functools import partial

import ipyvuetify as v

from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class ColorSwitch:

    def __init__(self, data_store: DataStore, update_callback):
        self.update_callback = partial(update_callback, self)
        self.data_store = data_store
        self.selectable_colors = ["y", "y^", "residual", "regions" ]
        self._build_widget()

    def _build_widget(self):
        self.widget = v.BtnToggle(  # 11
            class_="mr-3",
            mandatory=False,
            disabled=False,
            v_model="Y",
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
                                children=[
                                    v.Icon(children=["mdi-alpha-y-circle-outline"])
                                ],
                                value="y",
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
                                children=[v.Icon(children=["mdi-alpha-y-circle"])],
                                value="y^",
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
                                children=[v.Icon(children=["mdi-delta"])],
                                value="residual",
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
                                children=[
                                    v.Icon(children=["mdi-view-dashboard"])
                                ],
                                value="regions",
                                v_model=True,
                            ),
                    }],
                    children=['Display regions']),

            ],
        )

        self.widget.on_event("change", self.switch_color)

    def update_color(self, tab):


        self.update_callback() #refresh color of ES VS and Rule widget
        self.update_btn(value)
        self.switch_color(value = value)
    @log_errors
    def switch_color(self, widget, event, value):
        """
        Called with the user clicks on the colorChoiceBtnToggle
        Allows change the color of the dots
        """

        # Color : a pd.Series with one color value par row
        with Log('switch_color', 2):
            color = None
            stats_logger.log('color_changed', {'color': value})
            if value == "y":
                self.data_store.colors = self.data_store.y
            elif value == "y^":
                self.data_store.colors = self.data_store.y_pred
            elif value == "residual":
                self.data_store.colors = self.data_store.y - self.data_store.y_pred
            elif value == "regions":
                self.data_store.colors = self.data_store.region_set.get_color_serie()
            elif value == "rules":
                self.data_store.colors = self.data_store.rule_selection_color
            elif value == "region":
                self.data_store.colors = self.data_store.colors



    def update_btn(self, value):
        # Updates the button in the switch if the value parameter is one of the buttons,
        # else it will disable all buttons
        self.widget.v_model = value

