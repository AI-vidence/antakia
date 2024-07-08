import ipyvuetify as v
from ipyvuetify import BtnToggle

from antakia.gui.helpers.data import DataStore

BTN_DICT = {"y": {'icon':"mdi-alpha-y-circle-outline",  # icon
                  'tooltip':'Display target values',  # tootltip
                  'color_names':['y' ]},  # list of colors related to the btn


             "y^": {'icon':"mdi-alpha-y-circle",  # icon
                  'tooltip':'Display predicted values',  # tootltip
                  'color_names':['residual']},

             "residual": {'icon':"mdi-delta",  # icon
                  'tooltip':'Display residual values',  # tootltip
                  'color_names':['residual']},

             "y^model": {'icon':"mdi-alpha-y-box-outline",  # icon
                  'tooltip':"Display predicted values of the substitute model",  # tootltip
                  'color_names':['y^model']},

             "residual_sub": {'icon':"mdi-delta",  # icon
                  'tooltip':"Display residual values of the substitute model",  # tootltip
                  'color_names':['residual_sub']},

             "all_regions": {'icon':"mdi-view-dashboard",  # icon
                  'tooltip':'Display regions',  # tootltip
                  'color_names':['all_regions', "region_selection"]}}

class ColorSwitch:

    def __init__(self, data_store: DataStore, update_callback):
        self.color_update_callback = update_callback
        self.data_store = data_store
        self.btn_list = ["y", "y^", "residual", "all_regions"]
        self._build_widget()

    def _build_widget(self):
        self.widget = v.Col(children=[])
        self.widget.children = self._build_toggle()
        self.widget.children[0].on_event("change", self.color_update_callback)

    def _build_toggle(self) -> list[BtnToggle]:
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
            mandatory=True,
            disabled=False,
            children=[]
        )
        btn_widget_list = []
        for btn in self.btn_list:
            icon, tooltip = BTN_DICT[btn]['icon'], BTN_DICT[btn]['tooltip']
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
                            children=[v.Icon(children=[icon])],
                            value=btn,
                            v_model=True,
                        ),
                }],
                children=tooltip))
        btn_toggle.children = btn_widget_list
        return [btn_toggle]


    def update_btn_widget(self, btn_list):
        # Updates the button in the switch if the value parameter is one of the buttons,
        # else it will disable all buttons

        if btn_list != self.btn_list:  # update the button list
            self.btn_list = btn_list
            self.widget.children = self._build_toggle()  # updates the toggle with new btns
            self.widget.children[0].on_event("change", self.update_callback)

        for btn_value in BTN_DICT:
            if self.data_store.color in BTN_DICT[btn_value]['color_names']:
                self.widget.children[0].v_model = btn_value
