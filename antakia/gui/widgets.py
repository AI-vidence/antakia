import numpy as np
import pandas as pd

from ipywidgets import Layout, widgets
from ipywidgets.widgets import Widget
from IPython.display import display
import ipyvuetify as v
from traitlets import HasTraits, TraitError
from plotly.graph_objects import FigureWidget, Histogram, Scatter, Scatter3d
import seaborn as sns

from antakia.data import ExplanationMethod, DimReducMethod, Variable, ProjectedValues
import antakia.compute as compute
from antakia.utils import confLogger
from antakia.rules import Rule
import antakia.config as config

from copy import deepcopy, copy
from importlib.resources import files

import logging

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()


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
        return root_widget.children[int(address[0])]


def _get_parent(root_widget: Widget, address: str) -> Widget:
    if len(address) == 1:
        return root_widget
    else:
        return get_widget(root_widget, address[:-1])


def check_address(rootwidget: Widget, address: str) -> str:
    """
    For debug purposes
    """

    widget = rootwidget.children[int(address[0])]
    txt = f"[{address[0]}] : {widget.__class__.__name__}"
    if len(address) == 1:
        return txt
    elif widget is not None and len(widget.children) > 0:
        # let's continue further :
        return txt + ", " + check_address(widget, address[1:])
    else:
        # address targets a non existent widget :
        return txt + f", nothing @[{address[0]}]"
    

def change_widget(root_widget: Widget, address: str, sub_widget: Widget):
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
            new_children.append(sub_widget)
        else:
            new_children.append(parent_widget.children[i])
    parent_widget.children = new_children


def create_rule_card(object) -> list:
    return None


def datatable_from_Selection(sel: list, length: int) -> v.Row:
    """Returns a DataTable from a list of Selections"""
    new_df = []

    for i in range(len(sel)):
        new_df.append(
            [
                i + 1,
                sel[i].size(),
                str(
                    round(
                        sel[i].size() / length * 100,
                        2,
                    )
                )
                + "%",
            ]
        )
    new_df = pd.DataFrame(
        new_df,
        columns=["Region #", "Number of points", "Percentage of the dataset"],
    )
    data = new_df.to_dict("records")
    columns = [{"text": c, "sortable": False, "value": c} for c in new_df.columns]
    datatable = v.DataTable(
        class_="w-100",
        style_="width : 100%",
        show_select=False,
        single_select=True,
        v_model=[],
        headers=columns,
        explanationsMenuDict=data,
        item_value="Region #",
        item_key="Region #",
        hide_default_footer=True,
    )
    all_chips = []
    all_radio = []
    size = len(sel)
    coeff = 100
    start = 0
    end = (size * coeff - 1) * (1 + 1 / (size - 1))
    step = (size * coeff - 1) / (size - 1)
    scale_colors = np.arange(start, end, step)
    a = 0
    for i in scale_colors:
        color = sns.color_palette("viridis", size * coeff).as_hex()[round(i)]
        all_chips.append(v.Chip(class_="rounded-circle", color=color))
        all_radio.append(v.Radio(class_="mt-4", value=str(a)))
        a += 1
    all_radio[-1].class_ = "mt-4 mb-0 pb-0"
    radio_group = v.RadioGroup(
        v_model=None,
        class_="mt-10 ml-7",
        style_="width : 10%",
        children=all_radio,
    )
    chips_col = v.Col(
        class_="mt-10 mb-2 ml-0 d-flex flex-column justify-space-between",
        style_="width : 10%",
        children=all_chips,
    )
    return v.Row(
        children=[
            v.Layout(class_="flex-grow-0 flex-shrink-0", children=[radio_group]),
            v.Layout(class_="flex-grow-0 flex-shrink-0", children=[chips_col]),
            v.Layout(
                class_="flex-grow-1 flex-shrink-0",
                children=[datatable],
            ),
        ],
    )


dummy_df = pd.DataFrame(
    {
        "Variable": ["Population", "MedInc", "Latitude", "Longitude"],
        "Unit": ["people", "k€", "° N", "° W"],
        "Desc": ["People living in the block", "Median income", "-", "-"],
        "Critical ?": [False, True, False, False],
        "Rule": [
            "Population ≤ 2 309",
            "MedInc ∈ [3.172, 5.031⟧",
            "Latitude ≥ 37.935",
            "Longitude > 0.559",
        ],
    }
)

splash_widget = v.Layout(
    class_="d-flex flex-column align-center justify-center",
    children=[
        widgets.Image(  # 0
            value=widgets.Image._load_file_value(
                files("antakia.assets").joinpath("logo_antakia.png")
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
                        v.ProgressLinear(
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



app_widget = widgets.VBox(
    [
        v.AppBar(  # 0
            elevation="4",
            class_="ma-4",
            rounded=True,
            children=[
                v.Layout( # 00
                    children=[
                        widgets.Image(  # 010
                            value=open(
                                files("antakia.assets").joinpath("logo_ai-vidence.png"),
                                "rb",
                            ).read(),
                            height=str(864 / 20) + "px",
                            width=str(3839 / 20) + "px",
                        )
                    ],
                    class_="mt-1",
                ),
                v.Html( # 01
                    tag="h2", 
                    children=["AntakIA"], # 010
                    class_="ml-3"),  
                v.Spacer(),  # 02
                v.Btn(  # backupBtn # 03
                    icon=True,
                    children=[v.Icon(children=["mdi-content-save"])],
                    elevation=0,
                    disabled=True,
                ),
                v.Menu(  # 04
                    v_slots=[
                        {
                            "name": "activator",
                            "variable": "props",
                            "children": v.Btn(
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
                        v.Card(  # 040
                            class_="pa-4",
                            rounded=True,
                            children=[
                                widgets.VBox(  # 040 0
                                    [
                                        v.Slider(  # 040 0O
                                            v_model=400,
                                            min=100,
                                            max=3000,
                                            step=10,
                                            label="Figure width",
                                        )
                                    ]
                                )
                            ],
                            min_width="500",
                        )
                    ],
                    v_model=False,
                    close_on_content_click=False,
                    offset_y=True,
                ),
                v.Btn(  # gotoWebBtn # 06
                    icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
                ),
            ],
        ),
        widgets.HBox(  # 1
            [
                v.Row(  # 10
                    class_="ma-3",
                    children=[
                        v.Icon(children=["mdi-numeric-2-box"]),
                        v.Switch( # 102
                            class_="ml-3 mr-2",
                            v_model=False,
                            label="",
                        ),
                        v.Icon(children=["mdi-numeric-3-box"]),
                    ],
                ),
                v.Layout(  # 11
                    class_="pa-2 ma-2",
                    elevation="3",
                    children=[
                        v.BtnToggle(  # colorChoiceBtnToggle # 110
                            mandatory=True,
                            v_model="Y",
                            children=[
                                v.Btn(  # 1100
                                    icon=True,
                                    children=[
                                        v.Icon(children=["mdi-alpha-y-circle-outline"])
                                    ],
                                    value="y",
                                    v_model=True,
                                ),
                                v.Btn(  # 1101
                                    icon=True,
                                    children=[v.Icon(children=["mdi-alpha-y-circle"])],
                                    value="y^",
                                    v_model=True,
                                ),
                                v.Btn(  # 1102
                                    icon=True,
                                    children=[v.Icon(children=["mdi-delta"])],
                                    value="residual",
                                ),
                            ],
                        ),
                        v.Select(  # explanationSelect # 111
                            label="Explanation method",
                            items=[
                                {"text": "SHAP (imported)", "disabled": True},
                                {"text": "SHAP (computed)", "disabled": True},
                                {"text": "LIME (imported)", "disabled": True},
                                {"text": "LIME (computed)", "disabled": True},
                            ],
                            class_="ma-2 mt-1 ml-6",
                            style_="width: 150px",
                            disabled=False,
                        ),
                        v.Btn(  # computeMenuBtnBtn # 112
                            icon=True,
                            children=[
                                v.Icon(  # 112 0
                                    children=[
                                        "mdi-opacity"  # 112 00
                                        ])
                                ],
                            class_="ma-2 ml-6 pa-3",
                            elevation="3",
                        ),
                    ],
                ),
                v.Layout(  # 12
                    class_="mt-3",
                    children=[
                        widgets.HBox(  # 120
                            [
                                v.Select(   # 1200
                                    label="Projection in the VS :",
                                    items=DimReducMethod.dimreduc_methods_as_str_list(),
                                    style_="width: 150px",
                                ),
                                v.Layout(  # 120 1
                                    children=[
                                        v.Menu(   # 120 10
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
                                                v.Card(  # 120 100
                                                    class_="pa-4",
                                                    rounded=True,
                                                    children=[
                                                        widgets.VBox(
                                                            [  # 120 100 0
                                                                v.Slider(
                                                                    v_model=10,
                                                                    min=5,
                                                                    max=30,
                                                                    step=1,
                                                                    label="Number of neighbours",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=0.5,
                                                                    min=0.1,
                                                                    max=0.9,
                                                                    step=0.1,
                                                                    label="MN ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=2,
                                                                    min=0.1,
                                                                    max=5,
                                                                    step=0.1,
                                                                    label="FP ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                            ],
                                                        )
                                                    ],
                                                    min_width="500",
                                                )
                                            ],
                                            v_model=False,
                                            close_on_content_click=False,
                                            offset_y=True,
                                        )
                                    ]
                                ),
                                widgets.HBox(  # 120 2
                                    [
                                        v.ProgressCircular(  # 120 20
                                            indeterminate=True,
                                            color="blue",
                                            width="6",
                                            size="35",
                                            class_="mx-4 my-3",
                                        )
                                    ]
                                ),
                            ]
                        ),
                        widgets.HBox(  # 121
                            [
                                v.Select(  # 121 0
                                    label="Projection in the ES :",
                                    items=DimReducMethod.dimreduc_methods_as_str_list(),
                                    style_="width: 150px",
                                ),
                                v.Layout(  # 121 1
                                    children=[
                                        v.Menu(  # 121 10
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
                                                v.Card(  # 121 100
                                                    class_="pa-4",
                                                    rounded=True,
                                                    children=[
                                                        widgets.VBox(
                                                            [  # 121 100 0
                                                                v.Slider(
                                                                    v_model=10,
                                                                    min=5,
                                                                    max=30,
                                                                    step=1,
                                                                    label="Number of neighbours",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=0.5,
                                                                    min=0.1,
                                                                    max=0.9,
                                                                    step=0.1,
                                                                    label="MN ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=2,
                                                                    min=0.1,
                                                                    max=5,
                                                                    step=0.1,
                                                                    label="FP ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                            ],
                                                        )
                                                    ],
                                                    min_width="500",
                                                )
                                            ],
                                            v_model=False,
                                            close_on_content_click=False,
                                            offset_y=True,
                                        )
                                    ]
                                ),
                                widgets.HBox(  # 121 2
                                    [
                                        v.ProgressCircular(  # 121 20
                                            indeterminate=True,
                                            color="blue",
                                            width="6",
                                            size="35",
                                            class_="mx-4 my-3",
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            layout=Layout(
                width="100%",
                display="flex",
                flex_flow="row",
                justify_content="space-around",
            ),
        ),
        widgets.VBox(  # 2
            [
                widgets.HBox(  # 20
                    [
                        widgets.VBox(  # 200
                            [
                                widgets.HTML("<h3>Values Space<h3>"),  # 2000
                                v.Container(  # 2001 # placeholder for vs_hde.figure_widget
                                    children=[
                                        FigureWidget(
                                            [Scatter(mode="markers")]
                                        ),  # 20010
                                    ]
                                ),
                            ],
                            layout=Layout(
                                display="flex",
                                align_items="center",
                                margin="0px 0px 0px 0px",
                            ),
                        ),
                        widgets.VBox(  #  #201
                            [
                                widgets.HTML("<h3>Explanations Space<h3>"),  # 201 0
                                v.Container(  # 201 1
                                    children=[
                                        FigureWidget(
                                            [Scatter(mode="markers")]
                                        ),  # 201 10
                                    ]
                                ),
                            ],
                            layout=Layout(
                                display="flex",
                                align_items="center",
                                margin="0px 0px 0px 0px",
                            ),
                        ),
                    ],
                    layout=Layout(width="100%"),
                )
            ]
        ),
        v.Col( # 3
            # [  
            # fluid=True,
            children=[
                v.Tabs(  # 30
                    v_model=0,  # default active tab
                    children=[
                        v.Tab(children=["1. Selection"]),  # 300
                        v.Tab(children=["2. Refinement"]),  # 301
                        v.Tab(children=["3. Sub-model"]),  # 302
                        v.Tab(children=["4. Regions"]),  # 303
                    ]
                    + [
                        v.TabItem(  # Tab 1) Selection # 304
                            class_="d-flex flex-column align-left",
                            children=[
                                # v.Card(  # 304 0
                                #     class_="ma-2",
                                #     elevation=0,
                                #     children=[
                                v.Row(  # 304 00
                                    children=[
                                        v.Icon( # 304 000
                                            children=["mdi-lasso"]
                                        ),  # 304 000
                                        v.Html(  # 304 001
                                            class_="mt-2 ml-4",
                                            tag="h4",
                                            children=[
                                                "0 point selected : use the lasso tool on the figures above or use the auto-selection tool below"  # 304 001 0
                                            ],
                                        ),
                                    ]
                                ),
                                #     ],
                                # ),
                                v.ExpansionPanels(  # 304 1
                                    class_="ma-2",
                                    children=[
                                        v.ExpansionPanel(  # 304 10
                                            style_="max-width: 90%",
                                            class_="flex",
                                            children=[
                                                v.ExpansionPanelHeader(  # 304 100
                                                    children=["Data selected"]
                                                ),  # 304 100 0
                                                v.ExpansionPanelContent(  # 304 101
                                                    # children=[
                                                    #     v.Alert(  # 304 101 0
                                                            children=[
                                                                v.Row(  # 304 101 00
                                                                    children=[
                                                                        v.Layout(  # 304 101 000
                                                                            children=[
                                                                                v.DataTable(  # 304 410 000 1
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
                                                                        )
                                                                    ],
                                                                ),
                                                            ],
                                                    #     ),
                                                    # ]
                                                ),
                                            ]
                                        )
                                    ],
                                )
                            ]
                        ),
                        v.TabItem(  # Tab 2) Refinement # 305
                            class_="mt-5",
                            children=[
                                v.Col(  # 305 0
                                    children=[
                                        widgets.VBox( # 305 00
                                            [
                                                v.Layout( # 305 000
                                                    class_="d-flex flex-row",
                                                    children=[
                                                        v.Btn( # 305 000 0
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(
                                                                    class_="mr-2",
                                                                    children=[
                                                                        "mdi-lasso"
                                                                    ],
                                                                ),
                                                                "Back to selection",
                                                            ],
                                                        ),
                                                        v.Btn( # 305 000 1
                                                            class_="ma-1 light-blue white--text",
                                                            children=[
                                                                v.Icon(
                                                                    class_="mr-2",
                                                                    children=[
                                                                        "mdi-axis-arrow"
                                                                    ],
                                                                ),
                                                                "Skope rules",
                                                            ],
                                                        ),
                                                        v.Btn( # 305 000 2
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(
                                                                    class_="mr-2",
                                                                    children=[
                                                                        "mdi-arrow-up-bold-outline"
                                                                    ],
                                                                ),
                                                                "Update graphs",
                                                            ],
                                                        ),
                                                        v.Btn( # 305 000 3
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
                                                        v.Btn( # 305 000 4
                                                            class_="ma-1 light-green white--text",
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
                                                    ],
                                                ),
                                            ]
                                        ),
                                        widgets.HBox(  # 305 01
                                            [
                                                widgets.VBox(  # placeholder for the VS RulesWidget (RsW) # 305 010
                                                    [
                                                        v.Col( # # 305 010 0
                                                            children=[
                                                                v.Row( # 305 010 00
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), # 305 010 000
                                                                        v.Html(class_="ml-3", tag="h2", children=["Rules applied on the value space"]), # 305 010 001
                                                                    ]
                                                                ),
                                                                v.Html( # 305 010 01
                                                                    class_="ml-7", 
                                                                    tag="p", 
                                                                    children=[
                                                                        "Precision = 0.3, Recall = 0.8, F1 = 22" # 305 010 010 0
                                                                    ]
                                                                ), 
                                                                ]
                                                        ),
                                                        v.ExpansionPanels( # 305 010 1
                                                            style_="max-width: 90%",
                                                            children=[
                                                                v.ExpansionPanel( # 305 010 10 Placeholder for a VS RuleWidget
                                                                    children=[
                                                                        v.ExpansionPanelHeader( # 305 010 100
                                                                            class_="font-weight-bold blue lighten-4",
                                                                            children=[
                                                                                "Variable"
                                                                            ]
                                                                        ),
                                                                        v.ExpansionPanelContent( # 305 010 101
                                                                            children=[
                                                                                v.Col( # 305 010 101 0
                                                                                    # class_="ma-3 pa-3",
                                                                                    children=[
                                                                                        v.Spacer(), # 305 010 101 00
                                                                                        v.RangeSlider(  # 305 010 101 01
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
                                                                                FigureWidget( # 305 010 101 1
                                                                                    data=[
                                                                                        Histogram(
                                                                                            x=pd.DataFrame(
                                                                                                np.random.randint(
                                                                                                    0,
                                                                                                    100,
                                                                                                    size=(
                                                                                                        100,
                                                                                                        # 4,
                                                                                                    ),
                                                                                                ),
                                                                                                # columns=list(
                                                                                                #     "ABCD"
                                                                                                # ),
                                                                                            ),
                                                                                            bingroup=1,
                                                                                            nbinsx=50,
                                                                                            marker_color="grey",
                                                                                        )
                                                                                    ]
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    layout=Layout(width='50%'),
                                                ),
                                                widgets.VBox(  # placeholder for the ES RulesWidget (RsW) # 305 011
                                                    [
                                                        v.Col( # placeholder for the ES RulesWidget card # 305 011 0
                                                            children=[
                                                                v.Row( 
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), 
                                                                        v.Html(class_="ml-3", tag="h2", children=["Rules applied on the explanations space"]),
                                                                    ]
                                                                    ),
                                                                v.Html(class_="ml-7", tag="p", children=["Precision = 0.3, Recall = 0.8, F1 = 22"]),
                                                                ]
                                                        ),
                                                        v.ExpansionPanels( # 305 011 1
                                                            style_="max-width: 90%",
                                                            children=[
                                                                v.ExpansionPanel( # 305 011 10 Placeholder for the ES RuleWidgets 
                                                                    children=[
                                                                        v.ExpansionPanelHeader( # 0 
                                                                            class_="font-weight-bold blue lighten-4",
                                                                            # variant="outlined",
                                                                            children=[
                                                                                "Another variable"
                                                                            ]
                                                                        ),
                                                                        v.ExpansionPanelContent( # 1
                                                                            children=[
                                                                                v.Col( # 10
                                                                                    # class_="ma-3 pa-3",
                                                                                    children=[
                                                                                        v.Spacer(), # 100
                                                                                        v.RangeSlider(  # 101 
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
                                                                                    data=[
                                                                                        Histogram(
                                                                                            x=pd.DataFrame(
                                                                                                np.random.randint(
                                                                                                    0,
                                                                                                    100,
                                                                                                    size=(
                                                                                                        100,
                                                                                                        # 4,
                                                                                                    ),
                                                                                                ),
                                                                                                # columns=list(
                                                                                                #     "ABCD"
                                                                                                # ),
                                                                                            ),
                                                                                            bingroup=1,
                                                                                            nbinsx=50,
                                                                                            marker_color="grey",
                                                                                        )
                                                                                    ]
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ]
                                                                )
                                                            ]
                                                        )
                                                    ],
                                                    layout=Layout(width='50%'),
                                                ),
                                            ],
                                        )
                                    ]
                                )
                            ]
                        ),
                        v.TabItem(  # Tab 3) = tabThreeSubstitutionVBox ? # 306
                            children=[
                                widgets.VBox(  # 306 0
                                    [
                                        v.ProgressLinear(  # loadingModelsProgLinear # 306 00
                                            indeterminate=True,
                                            class_="my-0 mx-15",
                                            style_="width: 100%;",
                                            color="primary",
                                            height="5",
                                        ),
                                        v.SlideGroup(  # subModelslides # 306 01
                                            v_model=None,
                                            class_="ma-3 pa-3",
                                            elevation=4,
                                            center_active=True,
                                            show_arrows=True,
                                            children=[
                                                v.SlideItem(  # 306 010 # dummy SlideItem. Will be replaced by the app
                                                    # style_="width: 30%",
                                                    children=[
                                                        v.Card(
                                                            class_="grow ma-2",
                                                            children=[
                                                                v.Row(
                                                                    class_="ml-5 mr-4",
                                                                    children=[
                                                                        v.Icon(
                                                                            children=[
                                                                                "a name"
                                                                            ]
                                                                        ),
                                                                        v.CardTitle(
                                                                            children=[
                                                                                "model foo"
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                                v.CardText(
                                                                    class_="mt-0 pt-0",
                                                                    children=[
                                                                        "Model's score"
                                                                    ],
                                                                ),
                                                            ],
                                                        )
                                                    ],
                                                )
                                            ],
                                        ),
                                    ]
                                )
                            ]
                        ),
                        v.TabItem(  # Tab 4) = tabFourRegionListVBox # 307
                            children=[
                                v.Col(  # 307 0
                                    children=[
                                        widgets.VBox(  # 307 00
                                            [
                                                v.Btn(  # 307 000
                                                    class_="ma-4 pa-2 mb-1",
                                                    children=[
                                                        v.Icon(
                                                            class_="mr-4",
                                                            children=["mdi-map"],
                                                        ),
                                                        "Validate the region",
                                                    ],
                                                    color="white",
                                                    disabled=True,
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ],
                    layout=Layout(width="100%"),
                ) # end v.Tabs
            ],
            class_="mt-00",
        ), # End v.Col
    ]
)
