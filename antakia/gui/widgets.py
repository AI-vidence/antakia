import numpy as np
import pandas as pd

from ipywidgets import Layout, widgets
from ipywidgets.widgets import Widget
import ipyvuetify as v
from plotly.graph_objects import FigureWidget, Histogram, Scattergl
import traitlets

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.gui.colorTable import ColorTable

from importlib.resources import files


def get_widget(root_widget: Widget, address: str) -> Widget:
    """
    Returns a sub widget of root_widget. Address is a sequence of childhood ranks as a string
    Return sub_widget may be modified, it's still the same sub_widget of the root_widget
    get_widget(root_widget, '0') returns root_widgetn first child
    TODO : allow childhood rank > 9
    """
    try:
        int(address)
    except ValueError:
        raise ValueError(address, "must be a string composed of digits")

    if len(address) > 1:
        try:
            return get_widget(root_widget.children[int(address[0])], address[1:])
        except IndexError:
            raise IndexError(f"Nothing found @{address} in this {root_widget.__class__.__name__}")
    else:
        if isinstance(root_widget, v.Tooltip):
            return root_widget.v_slots[0]["children"]
        else:
            return root_widget.children[int(address[0])]


def _get_parent(root_widget: Widget, address: str) -> Widget:
    if len(address) == 1:
        return root_widget
    else:
        return get_widget(root_widget, address[:-1])


def check_address(root_widget: Widget, address: str) -> str:
    """
    For debug purposes : check if the address is reachable and returns the widget class
    """

    widget = root_widget.children[int(address[0])]
    txt = f"[{address[0]}] : {widget.__class__.__name__}"
    if len(address) == 1:
        return txt
    elif widget is not None and len(widget.children) > 0:
        # let's continue further :
        return txt + ", " + check_address(widget, address[1:])
    else:
        # address targets a non existent widget :
        return txt + f", nothing @[{address[0]}]"


def show_tree(parent: Widget, filter: str = "", address: str = ""):
    if len(address) == 0:
        print(f"Root_widget : {parent.__class__.__name__} :")
    for i in range(len(parent.children)):
        child = parent.children[i]
        if filter in child.__class__.__name__:
            print(f"{' ' * 3 * len(address)} {child.__class__.__name__} @{address}{str(i)}")
        if (
                not isinstance(child, widgets.Image) and not isinstance(child, str) and
                not isinstance(child, v.Html) and not isinstance(parent.children[i], widgets.HTML) and
                not isinstance(child, FigureWidget) and not isinstance(child, ColorTable) and
                not isinstance(child, SubModelTable)
        ):
            show_tree(child, filter, address=f"{address}{i}")


def change_widget(root_widget: Widget, address: str, sub_widget: Widget | str):
    """
    Substitutes a sub_widget in a root_widget.
    Address is a sequence of childhood ranks as a string, root_widget first child address is  '0'
    The root_widget is altered but the object remains the same
    """
    try:
        int(address)
    except ValueError:
        raise ValueError(address, "must be a string composed of digits")

    parent_widget = _get_parent(root_widget, address)
    new_children = []
    for i in range(len(parent_widget.children)):
        if i == int(address[-1]):
            if sub_widget is not None:
                new_children.append(sub_widget)
        else:
            new_children.append(parent_widget.children[i])
    parent_widget.children = new_children


# ------------------- Dummy data for UI testing  --------------------------------

dummy_df = pd.DataFrame({'MedInc': [8.3252, 8.3014, 2.0804, 1.3578, 1.7135, 2.4038, 2.4597, 1.9274, 1.7969, 1.375],
                         'HouseAge': [41.0, 21.0, 43.0, 40.0, 43.0, 41.0, 49.0, 49.0, 48.0, 49.0],
                         'AveRooms': [6.984126984126984, 6.238137082601054, 4.294117647058823, 4.524096385543169,
                                      4.478143076502732, 4.495798319327731, 4.728033472803348, 5.068783068783069,
                                      5.737313432835821, 5.030395136778116],
                         'AveBedrms': [1.0238095238095235, 0.9718804920913884, 1.1176470588235294, 1.108433734939759,
                                       1.0027322404371584, 1.0336134453781514, 1.0209205020920502, 1.1825396825396826,
                                       1.2208955223880598, 1.1124620060790271],
                         'Population': [322.0, 2401.0, 1206.0, 409.0, 929.0, 317.0, 607.0, 863.0, 1026.0, 754.0],
                         'AveOccup': [2.555555555555556, 2.109841827768014, 2.026890756302521, 2.463855431686747,
                                      2.5382513661202184, 2.663865546218488, 2.5397489539748954, 2.2830687830687832,
                                      3.062686567164179, 2.291793313069909],
                         'Latitude': [37.88, 37.86, 37.84, 37.85, 37.85, 37.85, 37.85, 37.84, 37.84, 37.83],
                         'Longitude': [-122.23, -122.22, -122.26, -122.27, -122.27, -122.28, -122.28, -122.28, -122.27,
                                       -122.27]})

dummy_sub_models_df = pd.DataFrame(
    {
        "Sub-model": ["LinearRegression", "LassoRegression", "RidgeRegression", "GaM", "EBM", "DecisionTreeRegressor"],
        "MSE": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "MAE": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "R2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
)

dummy_regions_df = pd.DataFrame(
    {
        "Region": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "Rules": ["Income ≥ 2800", "Segment ∈ ⟦D, E, F⟧", "Age ∈ [39, 45⟧", "Income ≥ 2800", "Segment ∈ ⟦D, E, F⟧",
                  "Age ∈ [39, 45⟧", "Income ≥ 2800", "Segment ∈ ⟦D, E, F⟧", "Age ∈ [39, 45⟧", "Income ≥ 2800"],
        "Points": [12, 123, 98, 3, 210, 333, 224, 93, 82, 241],
        "% dataset": ["5.7%", "21%", "13%", "5.7%", "21%", "13%", "5.7%", "21%", "13%", "5.7%"],
        "Sub-model": ["Linear regression", "Random forest", "Gradient boost", "Linear regression", "Random forest",
                      "Gradient boost", "Linear regression", "Random forest", "Gradient boost", "Linear regression"],
        "color": ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
    }
)

headers = [
    {
        "text": column,
        "sortable": False,
        "value": column,
    }
    for column in dummy_regions_df.columns
]
headers2 = headers.copy()
items = dummy_regions_df.to_dict('records')


class SubModelTable(v.VuetifyTemplate):
    headers = traitlets.List([]).tag(sync=True, allow_null=True)
    items = traitlets.List([]).tag(sync=True, allow_null=True)
    selected = traitlets.List([]).tag(sync=True, allow_null=True)
    template = traitlets.Unicode('''
        <template>
            <v-data-table
                v-model="selected"
                :headers="headers"
                :items="items"
                item-key="Sub-model"
                show-select
                single-select
                :hide-default-footer="false"
                @item-selected="tableselect"
            >
            
            <template v-slot:item.delta="{ item }">
              <v-chip :color="item.delta_color" label size="small">
                    {{ item.delta }}
              </v-chip>
            </template>
            </v-data-table>
        </template>
        ''').tag(sync=True)  # type: ignore
    disable_sort = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.callback = None

    # @click:row="tableclick"
    # def vue_tableclick(self, data):
    #     raise ValueError(f"click event data = {data}")

    def set_callback(self, callback: callable):  # type: ignore
        self.callback = callback

    def vue_tableselect(self, data):
        self.callback(data)


# ------------------- Splash screen mega widget --------------------------------
class SplashWidget:
    _widget = None

    @property
    def widget(self):
        if self._widget is None:
            self._widget = self.get_app_widget()
        return self._widget

    def reset(self):
        self._widget = None

    def get_app_widget(self):
        return v.Layout(
            class_="flex-column align-center justify-center",
            children=[
                widgets.Image(  # 0
                    value=widgets.Image._load_file_value(
                        files("antakia").joinpath("assets/logo_antakia.png")
                    ),
                    layout=Layout(width="230px"),
                ),
                v.Row(  # 1
                    style_="width:85%;",
                    children=[
                        v.Col(  # 10
                            children=[
                                v.Html(
                                    tag="h3",
                                    class_="mt-2 text-right",
                                    children=["Computation of explanation values"],
                                )
                            ]
                        ),
                        v.Col(  # 11
                            class_="mt-3",
                            children=[
                                v.ProgressLinear(  # 110
                                    style_="width: 80%",
                                    class_="py-0 mx-5",
                                    v_model=0,
                                    color="primary",
                                    height="15",
                                    striped=True,
                                )
                            ],
                        ),
                        v.Col(  # #12
                            children=[
                                v.TextField(  # 120
                                    variant="plain",
                                    v_model="",
                                    readonly=True,
                                    class_="mt-0 pt-0",
                                )
                            ]
                        ),
                    ],
                ),
                v.Row(  # 2
                    style_="width:85%;",
                    children=[
                        v.Col(  # 20
                            children=[
                                v.Html(
                                    tag="h3",
                                    class_="mt-2 text-right",
                                    children=["Computation of dimension reduction values"],
                                )
                            ]
                        ),
                        v.Col(  # 21
                            class_="mt-3",
                            children=[
                                v.ProgressLinear(  # 210
                                    style_="width: 80%",
                                    class_="py-0 mx-5",
                                    v_model=0,
                                    color="primary",
                                    height="15",
                                    striped=True,
                                )
                            ],
                        ),
                        v.Col(  # 22
                            children=[
                                v.TextField(  # 220
                                    variant="plain",
                                    v_model="",
                                    readonly=True,
                                    class_="mt-0 pt-0",
                                )
                            ]
                        ),
                    ],
                ),
            ],
        )


splash_widget = SplashWidget()


# ------------------- AntakIA app mega widget --------------------------------
class AppWidget:
    _widget = None

    @property
    def widget(self):
        if self._widget is None:
            self._widget = self.get_app_widget()
        return self._widget

    def reset(self):
        self._widget = None

    def get_app_widget(self):
        return v.Col(
            children=[
                v.AppBar(  # Top bar # 0
                    class_="white",
                    children=[
                        v.Layout(  # 00
                            children=[
                                widgets.Image(  # 000
                                    value=open(
                                        files("antakia").joinpath("assets/logo_antakia_horizontal.png"),  # type: ignore
                                        "rb",
                                    ).read(),
                                    height=str(864 / 20) + "px",
                                    width=str(3839 / 20) + "px",
                                )
                            ],
                        ),
                        v.Html(  # 01
                            tag="h2",
                            children=[""],  # 010
                            class_="blue-darken-3--text"
                        ),
                        v.Spacer(),  # 02
                        v.Menu(  # 03 # Menu for the figure width
                            v_slots=[
                                {
                                    "name": "activator",
                                    "variable": "props",
                                    "children":
                                        v.Btn(
                                            v_on="props.on",
                                            icon=True,
                                            size="x-large",
                                            children=[v.Icon(children=["mdi-tune"])],
                                            class_="ma-2 pa-3",
                                            elevation="0",
                                        ),
                                }
                            ],
                            children=[
                                v.Card(  # 030 parameters menu
                                    class_="pa-4",
                                    rounded=True,
                                    children=[
                                    ],
                                    min_width="500",
                                )
                            ],
                            v_model=False,
                            close_on_content_click=False,
                            offset_y=True,
                        ),  # End V.Menu
                    ],  # End AppBar children
                ),  # End AppBar
                v.Row(  # Top buttons bar # 1
                    class_="mt-3 align-center",
                    children=[
                        v.Tooltip(  # 10
                            bottom=True,
                            v_slots=[
                                {
                                    'name': 'activator',
                                    'variable': 'tooltip',
                                    'children':
                                        v.Switch(  # 100 # Dimension switch
                                            v_on='tooltip.on',
                                            class_="ml-6 mr-2",
                                            v_model=False,
                                            label="2D/3D",
                                        ),
                                }  # End v_slots dict
                            ],  # End v_slots list
                            children=['Change dimensions']
                        ),  # End v.Tooltip
                        v.BtnToggle(  # 11
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
                        ),
                        v.Col(  # 12
                            class_="ml-4 mr-4",
                            children=[
                                v.Select(  # Select of explanation method # 120
                                    label="Explanation method",
                                    items=[
                                        {"text": "Imported", "disabled": True},
                                        {"text": "SHAP", "disabled": True},
                                        {"text": "LIME", "disabled": True},
                                    ],
                                    class_="ml-2 mr-2",
                                    style_="width: 15%",
                                    disabled=False,
                                ),
                                v.ProgressCircular(  # 121 # exp menu progress bar
                                    class_="ml-2 mr-2",
                                    indeterminate=False,
                                    color="grey",
                                    width="6",
                                    size="35",
                                )
                            ]
                        ),
                        v.Col(  # 13 VS proj Select
                            class_="ml-6 mr-6",
                            children=[
                                v.Select(  # 130 # VS proj Select
                                    class_="ml-2 mr-2",
                                    label="Projection in the VS :",
                                    items=DimReducMethod.dimreduc_methods_as_str_list(),
                                    style_="width: 15%",
                                ),
                                v.Menu(  # 131 # VS proj settings
                                    class_="ml-2 mr-2",
                                    v_slots=[
                                        {
                                            "name": "activator",
                                            "variable": "props",
                                            "children": v.Btn(
                                                v_on="props.on",
                                                icon=True,
                                                size="x-large",
                                                children=[
                                                    v.Icon(
                                                        children=["mdi-cogs"],
                                                        size="large",
                                                    )
                                                ],
                                                class_="ma-2 pa-3",
                                                elevation="3",
                                            ),
                                        }
                                    ],
                                    children=[
                                        v.Card(  # 1310 VS placeholder for HDE._proj_params_cards
                                            class_="pa-4",
                                            rounded=True,
                                            children=[
                                                widgets.VBox(  # 13100
                                                    [
                                                        v.Slider(  # 131000
                                                            class_="ma-8 pa-2",
                                                            v_model=10,
                                                            min=5,
                                                            max=30,
                                                            step=1,
                                                            label="Number of neighbours",
                                                            thumb_label="always",
                                                            thumb_size=25,
                                                        ),
                                                        v.Slider(  # 131001
                                                            class_="ma-8 pa-2",
                                                            v_model=0.5,
                                                            min=0.1,
                                                            max=0.9,
                                                            step=0.1,
                                                            label="MN ratio",
                                                            thumb_label="always",
                                                            thumb_size=25,
                                                        ),
                                                        v.Slider(  # 131002
                                                            class_="ma-8 pa-2",
                                                            v_model=2,
                                                            min=0.1,
                                                            max=5,
                                                            step=0.1,
                                                            label="FP ratio",
                                                            thumb_label="always",
                                                            thumb_size=25,
                                                        )
                                                    ],
                                                )
                                            ],
                                            min_width="500",
                                        )
                                    ],
                                    v_model=False,
                                    close_on_content_click=False,
                                    offset_y=True,
                                ),
                                v.ProgressCircular(  # 132 # VS side progress bar
                                    class_="ml-2 mr-2",
                                    indeterminate=True,
                                    color="blue",
                                    width="6",
                                    size="35",
                                )
                            ]
                        ),
                        v.Col(  # 14 ES proj Select
                            class_="ml-6 mr-6",
                            children=[
                                v.Select(  # 140 # Selection of ES proj method
                                    label="Projection in the ES :",
                                    items=DimReducMethod.dimreduc_methods_as_str_list(),
                                    style_="width: 15%",
                                    class_="ml-2 mr-2",
                                ),
                                v.Menu(  # 141 # ES proj settings
                                    class_="ml-2 mr-2",
                                    v_slots=[
                                        {
                                            "name": "activator",
                                            "variable": "props",
                                            "children": v.Btn(
                                                v_on="props.on",
                                                icon=True,
                                                size="x-large",
                                                children=[
                                                    v.Icon(
                                                        children=["mdi-cogs"],
                                                        size="large",
                                                    )
                                                ],
                                                class_="ma-2 pa-3",
                                                elevation="3",
                                            ),
                                        }
                                    ],
                                    children=[
                                        v.Card(  # 1410
                                            class_="pa-4",
                                            rounded=True,
                                            children=[
                                                widgets.VBox(  # 14100
                                                    [
                                                        v.Slider(  # 141000
                                                            class_="ma-8 pa-2",
                                                            v_model=10,
                                                            min=5,
                                                            max=30,
                                                            step=1,
                                                            label="Number of neighbours",
                                                            thumb_label="always",
                                                            thumb_size=25,
                                                        ),
                                                        v.Slider(  # 141001
                                                            class_="ma-8 pa-2",
                                                            v_model=0.5,
                                                            min=0.1,
                                                            max=0.9,
                                                            step=0.1,
                                                            label="MN ratio",
                                                            thumb_label="always",
                                                            thumb_size=25,
                                                        ),
                                                        v.Slider(  # 141002
                                                            class_="ma-8 pa-2",
                                                            v_model=2,
                                                            min=0.1,
                                                            max=5,
                                                            step=0.1,
                                                            label="FP ratio",
                                                            thumb_label="always",
                                                            thumb_size=25,
                                                        )
                                                    ],
                                                )
                                            ],
                                            min_width="500",
                                        )
                                    ],
                                    v_model=False,
                                    close_on_content_click=False,
                                    offset_y=True,
                                ),
                                v.ProgressCircular(  # 142 # ES side progress bar
                                    indeterminate=True,
                                    color="blue",
                                    width="6",
                                    size="35",
                                    class_="ml-2 mr-6",
                                )
                            ]
                        ),

                    ]
                ),
                v.Row(  # The two HighDimExplorer # 2
                    class_="d-flex",
                    children=[
                        v.Col(  # VS HDE # 20
                            style_="width: 50%",
                            class_="d-flex flex-column justify-center",
                            children=[
                                v.Html(  # 200
                                    tag="h3",
                                    style_="align-self: center",
                                    class_="mb-3",
                                    children=["Values space"]
                                ),
                                v.Container(  # VS HDE placeholder # 201
                                    class_="flex-fill yellow",
                                    children=[
                                        FigureWidget(
                                            data=[
                                                Scattergl(
                                                    x=pd.DataFrame({'X': np.random.normal(0, 1, 500) * 2})['X'],
                                                    y=pd.DataFrame({'Y': np.random.normal(0, 1, 500) * 2})['Y'],
                                                    mode="markers",
                                                    marker=dict(
                                                        color=pd.Series(np.random.randint(0, 5, size=500)),
                                                    ),
                                                ),
                                            ],
                                            layout={
                                                'height': 300,
                                                'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
                                                'width': 600
                                            }
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        v.Col(  # ES HDE placeholder # 21
                            style_="width: 50%",
                            class_="d-flex flex-column justify-center",
                            children=[
                                v.Html(  # 210
                                    tag="h3",
                                    style_="align-self: center",
                                    class_="mb-3",
                                    children=["Explanations space"]
                                ),
                                v.Container(  # 211
                                    class_="flex-fill ",
                                    children=[
                                        FigureWidget(
                                            data=[
                                                Scattergl(
                                                    x=pd.DataFrame({'X': np.random.normal(0, 1, 500) * 2})['X'],
                                                    y=pd.DataFrame({'Y': np.random.normal(0, 1, 500) * 2})['Y'],
                                                    mode="markers",
                                                    marker=dict(
                                                        color=pd.Series(np.random.randint(0, 5, size=500)),
                                                    ),
                                                ),
                                            ],
                                            layout={
                                                'height': 300,
                                                'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
                                                'width': 600
                                            }
                                        ),
                                    ]
                                )
                            ],
                        ),
                    ],
                ),
                v.Divider(),  # 3
                v.Tabs(  # 4
                    v_model=0,  # default active tab
                    children=[
                                 v.Tab(children=["Selection"]),  # 40
                                 v.Tab(children=["Regions"]),  # 41
                                 v.Tab(children=["Substitution"]),  # 42
                             ]
                             +
                             [
                                 v.TabItem(  # Tab 1) Selection # 43
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
                                 ),
                                 v.TabItem(  # Tab 2) Regions #44
                                     children=[
                                         v.Row(  # 440
                                             class_="d-flex flex-row",
                                             children=[
                                                 v.Col(  # v.Sheet Col 1 # 4400
                                                     class_="col-8",
                                                     children=[
                                                         v.Html(  # 44000
                                                             tag="h3",
                                                             class_="ml-2",
                                                             children=["Regions :"],
                                                         ),
                                                         ColorTable(  # 44001
                                                             headers=headers2[:-1],
                                                             items=dummy_regions_df.to_dict(
                                                                 "records"
                                                             ),
                                                         ),
                                                         v.Html(  # 44002
                                                             tag="p",
                                                             class_="ml-2 mb-2",
                                                             children=["0 region, 0% of the dataset"],
                                                         ),
                                                     ]
                                                 ),  # End Col 1
                                                 v.Col(  # v.Sheet Col 2 = buttons #4401
                                                     class_="col-2",
                                                     style_="size: 50%",
                                                     children=[
                                                         v.Row(  # 44010
                                                             class_="flex-column",
                                                             children=[
                                                                 v.Tooltip(  # 440100
                                                                     bottom=True,
                                                                     v_slots=[
                                                                         {
                                                                             'name': 'activator',
                                                                             'variable': 'tooltip',
                                                                             'children':
                                                                                 v.Btn(  # 4401000
                                                                                     v_on='tooltip.on',
                                                                                     class_="ml-3 mt-8 green white--text",
                                                                                     children=[
                                                                                         v.Icon(
                                                                                             class_="mr-2",
                                                                                             children=[
                                                                                                 "mdi-swap-horizontal-circle-outline"
                                                                                             ],
                                                                                         ),
                                                                                         "Substitute",
                                                                                     ],
                                                                                 )
                                                                         }
                                                                     ],
                                                                     children=[
                                                                         'Find an explicable surrogale model on this region']
                                                                 )
                                                             ]
                                                         ),
                                                         v.Row(  # 44011
                                                             class_="flex-column",
                                                             children=[
                                                                 v.Tooltip(  # 440110
                                                                     bottom=True,
                                                                     v_slots=[
                                                                         {
                                                                             'name': 'activator',
                                                                             'variable': 'tooltip',
                                                                             'children':
                                                                                 v.Btn(  # 4401100
                                                                                     v_on='tooltip.on',
                                                                                     class_="ml-3 mt-8",
                                                                                     children=[
                                                                                         v.Icon(
                                                                                             class_="mr-2",
                                                                                             children=[
                                                                                                 "mdi-content-cut"
                                                                                             ],
                                                                                         ),
                                                                                         "Divide",
                                                                                     ],
                                                                                 )
                                                                         }
                                                                     ],
                                                                     children=[
                                                                         'Divide a region into sub-regions']
                                                                 )
                                                             ]
                                                         ),
                                                         v.Row(  # 44012
                                                             class_="flex-column",
                                                             children=[
                                                                 v.Tooltip(  # 440120
                                                                     bottom=True,
                                                                     v_slots=[
                                                                         {
                                                                             'name': 'activator',
                                                                             'variable': 'tooltip',
                                                                             'children':
                                                                                 v.Btn(  # 4401200
                                                                                     v_on='tooltip.on',
                                                                                     class_="ml-3 mt-3",
                                                                                     children=[
                                                                                         v.Icon(
                                                                                             class_="mr-2",
                                                                                             children=[
                                                                                                 "mdi-table-merge-cells"
                                                                                             ],
                                                                                         ),
                                                                                         "Merge",
                                                                                     ],
                                                                                 )
                                                                         }
                                                                     ],
                                                                     children=[
                                                                         'Merge regions']
                                                                 )
                                                             ]
                                                         ),
                                                         v.Row(  # 44013
                                                             class_="flex-column",
                                                             children=[
                                                                 v.Tooltip(  # 440130
                                                                     bottom=True,
                                                                     v_slots=[
                                                                         {
                                                                             'name': 'activator',
                                                                             'variable': 'tooltip',
                                                                             'children':
                                                                                 v.Btn(  # 4401300
                                                                                     v_on='tooltip.on',
                                                                                     class_="ml-3 mt-3",
                                                                                     children=[
                                                                                         v.Icon(
                                                                                             class_="mr-2",
                                                                                             children=[
                                                                                                 "mdi-delete"
                                                                                             ],
                                                                                         ),
                                                                                         "Delete",
                                                                                     ],
                                                                                 )
                                                                         }
                                                                     ],
                                                                     children=[
                                                                         'Delete region']
                                                                 )
                                                             ]
                                                         ),
                                                     ]  # End v.Sheet Col 2 children
                                                 ),  # End v.Sheet Col 2 = buttons
                                                 v.Col(  # v.Sheet Col 3 # 4402
                                                     class_="col-2 px-6",
                                                     style_="size: 50%",
                                                     children=[
                                                         v.Row(  # 44020
                                                             class_="flex-column",
                                                             children=[
                                                                 v.Tooltip(  # 440200
                                                                     bottom=True,
                                                                     v_slots=[
                                                                         {
                                                                             'name': 'activator',
                                                                             'variable': 'tooltip',
                                                                             'children':
                                                                                 v.Btn(  # 4402000
                                                                                     v_on='tooltip.on',
                                                                                     class_="mt-8 primary",
                                                                                     children=[
                                                                                         v.Icon(  # 44020000
                                                                                             class_="mr-2",
                                                                                             children=[
                                                                                                 "mdi-auto-fix"
                                                                                             ],
                                                                                         ),
                                                                                         "Auto-clustering",
                                                                                     ],
                                                                                 )
                                                                         }
                                                                     ],
                                                                     children=[
                                                                         'Find homogeneous regions in both spaces']
                                                                 )
                                                             ]
                                                         ),
                                                         v.Row(  # 44021
                                                             class_="flex-column",
                                                             children=[
                                                                 v.Tooltip(  # 440210
                                                                     bottom=True,
                                                                     v_slots=[
                                                                         {
                                                                             'name': 'activator',
                                                                             'variable': 'tooltip',
                                                                             'children':
                                                                                 v.Slider(  # 4402100
                                                                                     v_on='tooltip.on',
                                                                                     class_="mt-10 mx-2",
                                                                                     v_model=6,
                                                                                     min=2,
                                                                                     max=12,
                                                                                     thumb_color='blue',  # marker color
                                                                                     step=1,
                                                                                     thumb_label="always"
                                                                                 ),
                                                                         }
                                                                     ],
                                                                     children=['Numner of clusters you expect to find']
                                                                 ),
                                                                 v.Checkbox(  # 440211
                                                                     class_="px-3",
                                                                     v_model=True,
                                                                     label="Automatic number of clusters"
                                                                 ),
                                                                 v.ProgressLinear(  # 440212
                                                                     style_="width: 100%",
                                                                     class_="px-3",
                                                                     v_model=0,
                                                                     color="primary",
                                                                     height="15",
                                                                 )
                                                             ]
                                                         ),
                                                     ]  # End v.Sheet Col 3 children
                                                 )  # End v.Sheet Col 3
                                             ]  # End v.Sheet children
                                         ),  # End v.Sheet
                                     ]  # End of v.TabItem #2 children
                                 ),  # End of v.TabItem #2
                                 v.TabItem(  # TabItem #3 Substitution #45
                                     children=[
                                         v.Row(  # 450
                                             class_="d-flex",
                                             children=[
                                                 v.Col(  # Col1 - sub model table #4500
                                                     class_="col-5",
                                                     children=[
                                                         v.Sheet(  # 45000
                                                             class_="ma-1 d-flex flex-row align-center",
                                                             children=[
                                                                 v.Html(class_="mr-2", tag="h3", children=["Region"]),
                                                                 # 450000
                                                                 v.Chip(  # 450001
                                                                     color="red",
                                                                     children=["1"],
                                                                 ),
                                                                 v.Html(class_="ml-2", tag="h3",
                                                                        children=["3 rules, 240 points, 23% dataset"]),
                                                                 # 450002
                                                             ]
                                                         ),
                                                         SubModelTable(  # 45001
                                                             headers=[
                                                                 {
                                                                     "text": column,
                                                                     "sortable": True,
                                                                     "value": column,
                                                                     # "class": "primary white--text",\
                                                                 }
                                                                 for column in dummy_sub_models_df.columns
                                                             ],
                                                             items=dummy_sub_models_df.to_dict("records"),
                                                         )
                                                     ]
                                                 ),
                                                 v.Col(  # Col2 - buttons #4501
                                                     class_="col-2",
                                                     children=[
                                                         v.Row(
                                                             class_="flex-column",
                                                             children=[
                                                                 v.Tooltip(  # 45010
                                                                     bottom=True,
                                                                     v_slots=[
                                                                         {
                                                                             'name': 'activator',
                                                                             'variable': 'tooltip',
                                                                             'children':
                                                                                 v.Btn(  # 4501000
                                                                                     v_on='tooltip.on',
                                                                                     class_="ma-1 mt-12 green white--text",
                                                                                     children=[
                                                                                         v.Icon(
                                                                                             class_="mr-2",
                                                                                             children=[
                                                                                                 "mdi-check"
                                                                                             ],
                                                                                         ),
                                                                                         "Validate sub-model",
                                                                                     ],
                                                                                 ),
                                                                         }
                                                                     ],
                                                                     children=['Chose this submodel']
                                                                 )
                                                             ]
                                                         ),
                                                         v.Row(
                                                             class_="flex-column",
                                                             children=[
                                                                 v.ProgressLinear(  # 450110
                                                                     style_="width: 100%",
                                                                     class_="mt-4",
                                                                     v_model=0,
                                                                     height="15",
                                                                     indeterminate=True,
                                                                     color="blue",
                                                                 )
                                                             ]
                                                         )
                                                     ]
                                                 ),
                                                 v.Col(  # Col3 - model explorer #4502
                                                     class_="col-5",
                                                     children=[

                                                     ]
                                                 ),
                                             ]
                                         )
                                     ]
                                 )
                             ]
                )  # End of v.Tabs
            ]  # End v.Col children
        )  # End of v.Col


app_widget = AppWidget()
