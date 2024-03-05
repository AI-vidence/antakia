from functools import partial

import ipyvuetify as v

from antakia.utils.stats import log_errors, stats_logger


class ColorSwitch:
    def __init__(self, y, y_pred, update_callback):
        self.y = y
        self.y_pred = y_pred
        self.update_callback = partial(update_callback, self)
        self._build_widget()

    def _build_widget(self):
        self.widget = v.BtnToggle(  # 11
            class_="mr-3",
            mandatory=True,
            v_model="Y",
            children=[
                v.Tooltip(  # 110
                    bottom=True,
                    v_slots=[
                        {
                            'name': 'activator',
                            'variable': 'tooltip',
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
                        }
                    ],
                    children=['Display target values']
                ),
                v.Tooltip(  # 111
                    bottom=True,
                    v_slots=[
                        {
                            'name': 'activator',
                            'variable': 'tooltip',
                            'children':
                                v.Btn(  # 1110
                                    v_on='tooltip.on',
                                    icon=True,
                                    children=[v.Icon(children=["mdi-alpha-y-circle"])],
                                    value="y^",
                                    v_model=True,
                                ),
                        }
                    ],
                    children=['Display predicted values']
                ),
                v.Tooltip(  # 112
                    bottom=True,
                    v_slots=[
                        {
                            'name': 'activator',
                            'variable': 'tooltip',
                            'children':
                                v.Btn(  # 1120
                                    v_on='tooltip.on',
                                    icon=True,
                                    children=[v.Icon(children=["mdi-delta"])],
                                    value="residual",
                                    v_model=True,
                                ),
                        }
                    ],
                    children=['Display residual values']
                ),
            ],
        )

        self.widget.on_event("change", self.switch_color)

    @log_errors
    def switch_color(self, widget, event, data):
        """
        Called with the user clicks on the colorChoiceBtnToggle
        Allows change the color of the dots
        """

        # Color : a pd.Series with one color value par row
        color = None
        stats_logger.log('color_changed', {'color': data})
        if data == "y":
            color = self.y
        elif data == "y^":
            color = self.y_pred
        elif data == "residual":
            color = self.y - self.y_pred
        self.update_callback(color)