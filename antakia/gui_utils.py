"""
GUI Factory for AntakIA components
"""


import json
import logging
import os
import time
from copy import deepcopy
from importlib.resources import files

import ipyvuetify as v
import numpy as np
import pandas as pd
from ipywidgets import Layout, widgets
from ipywidgets.widgets import Widget
from IPython.display import display
from plotly.graph_objects import FigureWidget, Histogram, Scatter
import seaborn as sns

from antakia.antakia import AntakIA
from antakia.data import ProjectedValues, ExplanationMethod, DimReducMethod, Model, Variable
from antakia.utils import confLogger
from antakia.selection import Selection
import antakia.config as config

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()

def update_skr_infocards(selection: Selection, side: int, graph: widgets.VBox):
    """ Sets a message + indicates the scores of the sub_models
        Do not set the rules themselves
    """

    # We read 

    if selection.is_empty():
            temp_card_children = [widgets.HTML("Please select points")]
    else :
        if 0 not in selection.getYMaskList() or 1 not in selection.getYMaskList() :
                temp_card_children = [widgets.HTML("You can't choose everything/nothing !")]
        else:
            # If no rule for one of the two, nothing is displayed
            if not selection.has_rules_defined():
                    temp_card_children = [widgets.HTML("No rule found")]
            else:
                if side == config.VS :
                    scores = selection.getVSScore()
                else :
                    scores = selection.getESScore()
                temp_text_children= \
                    "p = " + str(scores[0]) + "% " \
                    + "r = " + str(scores[1]) + "% " \
                    + " ext. of the tree = " + str(scores[2])

    widget_at_address(graph, "30500101").children = temp_card_children
    widget_at_address(graph, "305001103").children = temp_text_children

    

def datatable_from_Selectiones(Selectiones: list, length: int) -> v.Row:
    """ Returns a DataTable from a list of Selectiones
    """
    new_df = []
    
    for i in range(len(Selectiones)):
        new_df.append(
            [
                i + 1,
                Selectiones[i].size(),
                str(
                    round(
                        Selectiones[i].size()
                        / length
                        * 100,
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
    columns = [
        {"text": c, "sortable": False, "value": c} for c in new_df.columns
    ]
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
    size = len(Selectiones)
    coeff = 100
    start = 0
    end = (size * coeff - 1) * (1 + 1 / (size - 1))
    step = (size * coeff - 1) / (size - 1)
    scale_colors = np.arange(start, end, step)
    a = 0
    for i in scale_colors:
        color = sns.color_palette(
            "viridis", size * coeff
        ).as_hex()[round(i)]
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
            v.Layout(
                class_="flex-grow-0 flex-shrink-0", children=[radio_group]
            ),
            v.Layout(
                class_="flex-grow-0 flex-shrink-0", children=[chips_col]
            ),
            v.Layout(
                class_="flex-grow-1 flex-shrink-0",
                children=[datatable],
            ),
        ],
    )


def widget_at_address(graph : Widget, address : str) :
    """ Returns the widget at the given address in the graph.
        Ex: widget_at_address(graph, "012") returns graph.children[0].children[1].children[2]
        Implemented recursively.
    """
    if len(address) > 1 :
        return widget_at_address(
            graph.children[int(address[0])],
            address[1:]
        )
    else :
        widget = graph.children[int(address[0])]
        # logger.debug(f"widget_at_address {address} : {type(widget)}")
        return widget


def add_model_slideItem(group : v.SlideGroup, model: Model):
    """ Adds a SlideItem to a SlideGroup with details about a model
    """
    item = v.SlideItem()
    item.children = [v.Card(
        class_="grow ma-2",
        children=[
            v.Row(
                class_="ml-5 mr-4",
                children=[
                    v.Icon(children=["mdi-numeric-1-box"]),
                    v.CardTitle(
                        children=[model.__class__.__name__]
                    ),
                ],
            ),
            v.CardText(
                class_="mt-0 pt-0",
                children=["Model's score"],
            ),
        ],
    )
    ]
    group.children.append(item)

def wrap_in_a_tooltip(widget, text):
    """ Allows to add a tooltip to a widget  # noqa: E501
    """
    pass
    wrapped_widget = v.Tooltip(
        bottom=True,
        v_slots=[
            {
                'name': 'activator',
                'variable': 'tooltip',
                'children': [widget]
            }
        ],
        children=[text]
    )
    widget.v_on = 'tooltip.on'
    widget = wrapped_widget


def createProgLinear() -> v.VuetifyWidget :
    widget = v.ProgressLinear(
            style_="width: 80%",
            class_="py-0 mx-5",
            v_model=0,
            color="primary",
            height="15",
            striped=True,
        )
    return widget

def createRow(text, element) -> v.VuetifyWidget :
    widget = v.Row(
            style_="width:85%;",
            children=[
                v.Col(
                    children=[
                        v.Html(
                            tag="h3",
                            class_="mt-2 text-right",
                            children=[text],
                        )
                    ]
                ),
                v.Col(class_="mt-3", children=[element]),
                v.Col(
                    children=[
                        v.TextField(
                            variant="plain",
                            v_model="0.00% [0/?] - 0m0s (estimated time : /min /s)",
                            readonly=True,
                            class_="mt-0 pt-0",
                            )
                    ]
                ),
            ],
        )
    return widget

def createProgressLinearColumn(methodName) -> v.VuetifyWidget :
    pb = v.ProgressLinear(
        style_="width: 80%",
        v_model=0,
        color="primary",
        height="15",
        striped=True,
    )
    widget = v.Col(
        class_="d-flex flex-column align-center",
        children=[
                v.Html(
                    tag="h3",
                    class_="mb-3",
                    children=["Compute " + methodName + " values"],
            ),
            pb,
            v.TextField(
                class_="w-100",
                style_="width: 100%",
                v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
                readonly=True,
            ),
            v.Btn(
                children=[v.Icon(class_="mr-2", children=["mdi-calculator-variant"]), "Compute values"],
                class_="ma-2 ml-6 pa-3",
                elevation="3",
                v_model=methodName,
                color="primary",
            ),
        ],
    )
    return widget

def SliderParam(v_model, min, max, step, label):
    widget = v.Layout(
            class_="mt-3",
            children=[
                v.Slider(
                    v_model=v_model, min=min, max=max, step=step, label=label
                ),
                v.Html(class_="ml-3", tag="h3", children=[str(v_model)]),
            ],
        )
    return widget

def createColorChoiceBtnToggle():
    colorChoiceBtnToggle = v.BtnToggle(
        color="blue",
        mandatory=True,
        v_model="Y",
        children=[
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-alpha-y-circle-outline"])],
                value="y",
                v_model=True,
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-alpha-y-circle"])],
                value="y^",
                v_model=True,
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-minus-box-multiple"])],
                value="Résidus",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children="mdi-lasso")],
                value="Selec actuelle",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-ungroup"])],
                value="Régions",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-select-off"])],
                value="Non selec",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-star"])],
                value="Clustering auto",
            ),
        ],
    )

    # added all tooltips!
    colorChoiceBtnToggle.children[0].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[0].children[0], "Real values")
    ]
    colorChoiceBtnToggle.children[1].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[1].children[0], "Predicted values")
    ]
    colorChoiceBtnToggle.children[2].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[2].children[0], "Residuals")
    ]
    colorChoiceBtnToggle.children[3].children = [
        wrap_in_a_tooltip(
            colorChoiceBtnToggle.children[3].children[0],
            "Selected points",
        )
    ]
    colorChoiceBtnToggle.children[4].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[4].children[0], "Region created")
    ]
    colorChoiceBtnToggle.children[5].children = [
        wrap_in_a_tooltip(
            colorChoiceBtnToggle.children[5].children[0],
            "Points that belong to no region",
        )
    ]
    colorChoiceBtnToggle.children[6].children = [
        wrap_in_a_tooltip(
            colorChoiceBtnToggle.children[6].children[0],
            "Automatic dyadic-clustering result",
        )
    ]
    return colorChoiceBtnToggle

def createMenuBar():
    """
    Create the menu bar"""

    figureSizeSlider = v.Slider(
        style_="width:20%",
        v_model=700,
        min=200,
        max=1200,
        label="With of both scattered plots (in pixels)",
    )

    figureSizeSliderIntText = widgets.IntText(
        value="700", disabled=True, layout=Layout(width="40%")
    )

    widgets.jslink((figureSizeSlider, "v_model"), (figureSizeSliderIntText, "value"))

    figureSizeSliderRow = v.Row(children=[figureSizeSlider, figureSizeSliderIntText])

    backupBtn = v.Btn(
        icon=True, children=[v.Icon(children=["mdi-content-save"])], elevation=0
    )
    backupBtn.children = [
        wrap_in_a_tooltip(backupBtn.children[0], "Backup management")
    ]

    settingsBtn = v.Btn(
        icon=True, children=[v.Icon(children=["mdi-tune"])], elevation=0
    )
    settingsBtn.children = [
        wrap_in_a_tooltip(settingsBtn.children[0], "Settings of the GUI")
    ]
    goToWebsiteBtn = v.Btn(
        icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
    )
    goToWebsiteBtn.children = [
        wrap_in_a_tooltip(goToWebsiteBtn.children[0], "AI-Vidence's website")
    ]

    goToWebsiteBtn.on_event(
        "click", lambda *args: webbrowser.open_new_tab("https://ai-vidence.com/")
    )

    data_path = files("antakia.assets").joinpath("logo_ai-vidence.png")
    with open(data_path, "rb") as f:
        logo = f.read()
    logoImage = widgets.Image(
        value=logo, height=str(864 / 20) + "px", width=str(3839 / 20) + "px"
    )
    logoImage.layout.object_fit = "contain"
    logoLayout = v.Layout(
        children=[logoImage],
        class_="mt-1",
    )

    # Dialog for the size of the figures  (before finding better)
    figureSizeDialog = v.Dialog(
        children=[
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            v.Icon(class_="mr-5", children=["mdi-cogs"]),
                            "Settings",
                        ]
                    ),
                    v.CardText(children=[figureSizeSliderRow]),
                        ]
                    ),
                ]
    )
    figureSizeDialog.v_model = False

    def openFigureSizeDialog(*args):
        figureSizeDialog.v_model = True

    settingsBtn.on_event("click", openFigureSizeDialog)

    # Menu bar (logo, links etc...)
    menuAppBar = v.AppBar(
        elevation="4",
        class_="ma-4",
        rounded=True,
        children=[
            logoLayout, # TODO : decide if we display
            v.Html(tag="h2", children=["AntakIA"], class_="ml-3"),
            v.Spacer(),
            backupBtn,
            settingsBtn,
            figureSizeDialog,
            goToWebsiteBtn,
        ],
    )
    return menuAppBar, figureSizeSlider, backupBtn


def createBackupsGUI(backupBtn : v.Btn ,ourbackups : list, initialNumBackups : int):

    # We create the backupDataTable
    text = "There is no backup"
    if len(ourbackups) > 0:
        text = str(len(ourbackups)) + " backup(s) found"
        dataTableRowList = []
        for i in range(len(bu)):
            if i > initialNumBackups:
                pass
            dataTableRowList.append(
                [
                    i + 1,
                    bu[i]["name"],
                    new_or_not,
                    len(bu[i]["Regions"]),
                ]
            )
        pd.DataFrame(
            dataTableRowList,
            columns=[
                "Backup #",
                "Name",
                "Origin",
                "Number of Regions",
            ],
        )

    dataTableColumnsDict = [ 
        # {"text": c, "sortable": True, "value": c} for c in dataTableColumnsDataFrame.columns
    ]
    v.DataTable(
        v_model=[],
        show_select=True,
        single_select=True,
        headers=dataTableColumnsDict,
        # explanationsMenuDict=dataTableRowList.to_dict("records"),
        item_value="Backup #",
        item_key="Backup #",
    )


    showBackupBtn = v.Btn(
        class_="ma-4",
        children=[v.Icon(children=["mdi-eye"])],
    )
    showBackupBtn.children = [
        wrap_in_a_tooltip(showBackupBtn.children[0], "Visualize the selected backup")
    ]

    def showBackup(*args): 
        pass
        raise NotImplementedError("showBackup() not implemented yet)") # TODO implement
        # self._backupTable = backupsCard.children[1]
        # if len(self._backupTable.v_model) == 0:
        #     return
        # index = self._backupTable.v_model[0]["Save #"] - 1
        # self.atk.Regions = [element for element in self.atk.saves[index]["Regions"]]
        # color = deepcopy(self.atk.saves[index]["labels"])
        # self._autoClusterRegionColors = deepcopy(color)
        # with self._leftVSFigure.batch_update():
        #     self._leftVSFigure.data[0].marker.color = color
        #     self._leftVSFigure.data[0].marker.opacity = 1
        # with self._rightESFigure.batch_update():
        #     self._rightESFigure.data[0].marker.color = color
        #     self._rightESFigure.data[0].marker.opacity = 1
        # with self._leftVSFigure3D.batch_update():
        #     self._leftVSFigure3D.data[0].marker.color = color
        # with self._rightESFigure3D.batch_update():
        #     self._rightESFigure3D.data[0].marker.color = color
        # colorChoiceBtnToggle.v_model = "Regions"
        # self._leftVSFigure.update_traces(marker=dict(showscale=False))
        # self._rightESFigure.update_traces(marker=dict(showscale=False))
        # newRegionValidated()

    showBackupBtn.on_event("click", showBackup)



    deleteBackupBtn = v.Btn(
        class_="ma-4",
        children=[
            v.Icon(children=["mdi-trash-can"]),
        ],
    )
    deleteBackupBtn.children = [
        wrap_in_a_tooltip(deleteBackupBtn.children[0], "Delete the selected backup")
    ]
    def deleteBackup(*args):
        ourbackups.remove(args[0]) #TODO check if it works
        # Update backupDataTable # TODO implement
    deleteBackupBtn.on_event("click", deleteBackup)

    newBackupBtn = v.Btn(
        class_="ma-4",
        children=[
            v.Icon(children=["mdi-plus"]),
        ],
    )
    newBackupBtn.children = [
        wrap_in_a_tooltip(newBackupBtn.children[0], "Create a new backup")
    ]
    def newBackup(*args): 

        if len(backupNameTextField.v_model) == 0 or len(backupNameTextField.v_model) > 25:
            raise Exception("The name of the backup must be between 1 and 25 characters !")
        # TODO : use a getter instead of __atk.__Regions
        backup1 = {"Regions": self.__atk.__Regions, "labels": self._regionColor,"name": backupNameTextField.v_model}
        backup = {key: value[:] for key, value in backup1.explanationsMenuDict()}
        self.__atk.__saves.append(backup)
        self._backupTable, text_region = initBackup(self.atk.saves)
        backupsCard.children = [text_region, self._backupTable] + backupsCard.children[2:]
    newBackupBtn.on_event("click", newBackup) #622
    
    backupNameTextField = v.TextField(
        label="Name of the backup",
        v_model="Default name",
        class_="ma-4",
    )# save backups locally
    saveAllBtn = v.Btn(
        class_="ma-0",
        children=[
            v.Icon(children=["mdi-content-save"], class_="mr-2"),
            "Save",
        ],
    )

    out_save = v.Alert(
        class_="ma-4 white--text",
        children=[
            "Save successful!",
        ],
        v_model=False,
    )

    # TODO Understand and refactor
    # part to save locally: choice of name, location, etc...
    saveLocal = v.Col(
        class_="text-center d-flex flex-column align-center justify-center",
        children=[
            v.Html(tag="h3", children=["Save in local_path"]),
            v.Row(
                children=[
                    v.Spacer(),
                    v.TextField(
                        prepend_icon="mdi-folder-search",
                        class_="w-40 ma-3 mr-0",
                        elevation=3,
                        variant="outlined",
                        style_="width: 30%;",
                        v_model="data_save",
                        label="Location",
                    ),
                    v.TextField(
                        prepend_icon="mdi-slash-forward",
                        class_="w-50 ma-3 mr-0 ml-0",
                        elevation=3,
                        variant="outlined",
                        style_="width: 50%;",
                        v_model="my_save",
                        label="Name of the file",
                    ),
                    v.Html(class_="mt-7 ml-0 pl-0", tag="p", children=[".json"]),
                    v.Spacer(),
                ]
            ),
            saveAllBtn,
            out_save,
        ],
    )

    # card to manage backups, which opens
    backupsCard = v.Card(
        elevation=0,
        children=[
            v.Html(tag="h3", children=[text]),
            # backupTable,
            v.Row(
                children=[
                    v.Spacer(),
                    showBackupBtn,
                    deleteBackupBtn,
                    v.Spacer(),
                    newBackupBtn,
                    backupNameTextField,
                    v.Spacer(),
                ]
            ),
            v.Divider(class_="mt mb-5"),
            saveLocal,
        ],
    )

    backupsDialog = v.Dialog(
        children=[
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            v.Icon(class_="mr-5", children=["mdi-content-save"]),
                            "Backup management",
                        ]
                    ),
                    v.CardText(children=[backupsCard]),
                ],
                width="100%",
            )
        ],
        width="50%",
    )
    backupsDialog.v_model = False

    def openBackup(*args):
        backupsDialog.v_model = True

    backupBtn.on_event("click", openBackup)

    def saveAllBackups(*args):
        save_regions = atk.saves
        orign = saveLocal.children[1].children[1].v_model
        fichier = saveLocal.children[1].children[2].v_model
        if len(orign) == 0 or len(fichier) == 0:
            out_save.color = "error"
            out_save.children = ["Please fill in all fields!"]
            out_save.v_model = True
            time.sleep(3)
            out_save.v_model = False
            return
        destination = orign + "/" + fichier + ".json"
        destination = destination.replace("//", "/")
        destination = destination.replace(" ", "_")

        cwd = os.getcwd()

        destination = cwd + "/" + destination

        isFile = os.path.isfile(destination)

        if (not isFile):
            with open(destination, 'w') as fp:
                pass

        for save in save_regions:
            for i in range(len(save["regions"])):
                save["regions"][i] = save["regions"][i].toJson()

        with open(destination, "w") as fp:
            json.dump(save_regions, fp)
        out_save.color = "success"
        out_save.children = [
            "Save successful in the following location:",
            destination,
        ]
        out_save.v_model = True
        time.sleep(3)
        out_save.v_model = False
        return

    saveAllBtn.on_event("click", saveAllBackups)

    # return backupsDialog, backupsCard
    return backupsDialog, backupsCard, deleteBackupBtn, backupNameTextField, showBackup, newBackupBtn



# ------

def createSubModelsSlides(models: list):
    sModelsSlide = []
    for i in range(len(models)):
        name = "mdi-numeric-" + str(i + 1) + "-box"
        modelSlideItem = v.SlideItem(
            # style_="width: 30%",
            children=[
                v.Card(
                    class_="grow ma-2",
                    children=[
                        v.Row(
                            class_="ml-5 mr-4",
                            children=[
                                v.Icon(children=[name]),
                                v.CardTitle(
                                    children=[models[i].__class__.__name__]
                                ),
                            ],
                        ),
                        v.CardText(
                            class_="mt-0 pt-0",
                            children=["Model's score"],
                        ),
                    ],
                )
            ],
        )
        sModelsSlide.append(modelSlideItem)

    slides = v.SlideGroup(
        v_model=None,
        class_="ma-3 pa-3",
        elevation=4,
        center_active=True,
        show_arrows=True,
        children=sModelsSlide,
    )

    return slides

def createSkopeSlider():
    skopeSlider = v.RangeSlider(
        class_="ma-3",
        v_model=[-1, 1],
        min=-10e10,
        max=10e10,
        step=0.01,
    )

    realTimeUpdateCheck = v.Checkbox(
        v_model=False, label="Real-time updates on the figures", class_="ma-3"
    )

    skopeSliderGroup = v.Layout(
        children=[
            v.TextField(
                style_="max-width:100px",
                v_model=skopeSlider.v_model[0], # min value of the slider
                hide_details=True,
                type="number",
                density="compact",
            ),
            skopeSlider,
            v.TextField(
                style_="max-width:100px",
                v_model=skopeSlider.v_model[1], # max value of the slider
                hide_details=True,
                type="number",
                density="compact",
                step="0.1",
            ),
        ],
    )
    return skopeSlider, realTimeUpdateCheck, skopeSliderGroup

def createHistograms(nombre_bins, fig_size):
    # TODO : i understand we create empty histograms to be further used
    x = np.linspace(0, 20, 20)
    histogram1 = FigureWidget(
        data=[
            Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="grey")
        ]
    )
    histogram1.update_layout(
        barmode="overlay",
        bargap=0.1,
        width=0.9 * int(fig_size),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
    )
    histogram1.add_trace(
        Histogram(
            x=x,
            bingroup=1,
            nbinsx=nombre_bins,
            marker_color="LightSkyBlue",
            opacity=0.6,
        )
    )
    histogram1.add_trace(
        Histogram(x=x, bingroup=1, nbinsx=nombre_bins, marker_color="blue")
    )
    histogram2 = deepcopy(histogram1)
    histogram3 = deepcopy(histogram2)
    return [histogram1, histogram2, histogram3]

def createBeeswarms(expDS : ExplanationDataset, explainMethod : int, figureSize : int):
    # TODO : understand this "color choice"
    # TODO : I guess this method creates 3 identical beeswarms with different colors
    bs1ColorChoice = v.Row(
        class_="pt-3 mt-0 ml-4",
        children=[
            "Value of Xi",
            v.Switch(
                class_="ml-3 mr-2 mt-0 pt-0",
                v_model=False,
                label="",
            ),
            "Current selection",
        ],
    )

    # TODO : we should use the Explanation method stored in GUI.__explanationES
    expValues = expDS.get_full_values(explainMethod)
    expValuesHistogram = [0] * len(expValues)

    beeswarm1 = FigureWidget(
        data=[Scatter(x=expValues, y=expValuesHistogram, mode="markers")]
    )
    beeswarm1.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        width=0.9 * int(figureSize),
    )
    beeswarm1.update_yaxes(visible=False, showticklabels=False)

    beeswarmGrp1 = widgets.VBox([bs1ColorChoice, beeswarm1])
    beeswarmGrp1.layout.margin = "0px 0px 0px 20px"

    bs2ColorChoice = v.Row(
        class_="pt-3 mt-0 ml-4",
        children=[
            "Value of Xi",
            v.Switch(
                class_="ml-3 mr-2 mt-0 pt-0",
                v_model=False,
                label="",
            ),
            "Current selection",
        ],
    )

    beeswarm2 = FigureWidget(
        data=[Scatter(x=expValues, y=expValuesHistogram, mode="markers")]
    )
    beeswarm2.update_layout(
        margin=dict(l=20, r=0, t=0, b=0),
        height=200,
        width=0.9 * int(figureSize),
    )
    beeswarm2.update_yaxes(visible=False, showticklabels=False)

    beeswarmGrp2 = widgets.VBox([bs2ColorChoice, beeswarm2])
    beeswarmGrp2.layout.margin = "0px 0px 0px 20px"

    beeswarm3 = FigureWidget(
        data=[Scatter(x=expValues, y=expValuesHistogram, mode="markers")]
    )
    beeswarm3.update_layout(
        margin=dict(l=20, r=0, t=0, b=0),
        height=200,
        width=0.9 * int(figureSize),
    )
    beeswarm3.update_yaxes(visible=False, showticklabels=False)

    bs3ColorChoice = v.Row(
        class_="pt-3 mt-0 ml-4",
        children=[
            "Value of Xi",
            v.Switch(
                class_="ml-3 mr-2 mt-0 pt-0",
                v_model=False,
                label="",
            ),
            "Current selection",
        ],
    )

    beeswarmGrp3 = widgets.VBox([bs3ColorChoice, beeswarm3])
    beeswarmGrp3.layout.margin = "0px 0px 0px 20px"

    return [beeswarmGrp1, beeswarmGrp2, beeswarmGrp3]

def colorChoiceBtnToggle() -> v.BtnToggle:
    colorChoiceBtnToggle = v.BtnToggle(
        color="blue",
        mandatory=True,
        v_model="Y",
        children=[
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-alpha-y-circle-outline"])],
                value="y",
                v_model=True,
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-alpha-y-circle"])],
                value="y^",
                v_model=True,
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-minus-box-multiple"])],
                value="Résidus",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children="mdi-lasso")],
                value="Selec actuelle",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-ungroup"])],
                value="Régions",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-select-off"])],
                value="Non selec",
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-star"])],
                value="Clustering auto",
            ),
        ],
    )

    colorChoiceBtnToggle.children[0].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[0].children[0], "Real values")
    ]
    colorChoiceBtnToggle.children[1].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[1].children[0], "Predicted values")
    ]
    colorChoiceBtnToggle.children[2].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[2].children[0], "Residuals")
    ]
    colorChoiceBtnToggle.children[3].children = [
        wrap_in_a_tooltip(
            colorChoiceBtnToggle.children[3].children[0],
            "Selected points",
        )
    ]
    colorChoiceBtnToggle.children[4].children = [
        wrap_in_a_tooltip(colorChoiceBtnToggle.children[4].children[0], "Region created")
    ]
    colorChoiceBtnToggle.children[5].children = [
        wrap_in_a_tooltip(
            colorChoiceBtnToggle.children[5].children[0],
            "Points that belong to no region",
        )
    ]
    colorChoiceBtnToggle.children[6].children = [
        wrap_in_a_tooltip(
            colorChoiceBtnToggle.children[6].children[0],
            "Automatic dyadic-clustering result",
        )
    ]
    return colorChoiceBtnToggle



def create_rule_card(string : str, is_class=False) -> list:
    """ Returns a list of CardText with string
    """
    chars = str(string).split()
    size = int(len(chars) / 5)
    l = []
    for i in range(size):
        l.append(
            v.CardText(
                children=[
                    v.Row(
                        class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                        children=[
                            chars[5 * i],
                            v.Icon(children=["mdi-less-than-or-equal"]),
                            chars[5 * i + 2],
                            v.Icon(children=["mdi-less-than-or-equal"]),
                            chars[5 * i + 4],
                        ],
                    )
                ]
            )
        )
        if i != size - 1:
            if chars[5 * i + 2] == chars[5 * (i+1) + 2]:
                l.append(v.Layout(class_="ma-n2 pa-0 d-flex flex-row justify-center align-center", children=[v.Html(class_="ma0 pa-0", tag="i", children=["or"])]))
            else:
                l.append(v.Divider())
    return l

# ----

def get_beeswarm_values(ds : Dataset, xds : ExplanationDataset, explainationMth : int, var_name : str) -> tuple:
    X = ds.getXValues(Dataset.CURRENT)
    y = ds.get_y_values(Dataset.PREDICTED)
    XP = xds.getValues(explainationMth)

    def order_ascending(lst : list):
        positions = list(range(len(lst)))  # Create a list of initial positions
        positions.sort(key=lambda x: lst[x])
        l = []
        for i in range(len(positions)):
            l.append(positions.index(i))  # Sort positions by list items
        return l
    
    explain_values_list = [0] * len(XP)
    es_bin_num = 60
    keep_index = []
    keep_Y_value = []
    for i in range(es_bin_num):
        keep_index.append([])
        keep_Y_value.append([])

    scale_list = np.linspace(
        min(XP[var_name]), max(XP[var_name]), es_bin_num + 1
    )
    for i in range(len(Exp)):
        for j in range(es_bin_num):
            if (
                XP[var_name][i] >= scale_list[j]
                and XP[var_name][i] <= scale_list[j + 1]
            ):
                keep_index[j].append(i)
                keep_Y_value[j].append(y[i])
                break
    for i in range(es_bin_num):
        l = order_ascending(keep_Y_value[i])
        for j in range(len(keep_index[i])):
            ii = keep_index[i][j]
            if l[j] % 2 == 0:
                explain_values_list[ii] = l[j]
            else:
                explain_values_list[ii] = -l[j]
    explain_marker = dict(
        size=4,
        opacity=0.6,
        color=X[var_name],
        colorscale="Bluered_r",
        colorbar=dict(thickness=20, title=var_name),
    )
    return [explain_values_list, explain_marker]


# ----

def createRegionsBtns():
    # Validation button to create a region
    validateRegionBtn = v.Btn(
        class_="ma-3",
        children=[
            v.Icon(class_="mr-3", children=["mdi-check"]),
            "Validate the selection",
        ],
    )
    # Delete All regions button
    deleteAllRegionsBtn = v.Btn(
        class_="ma-3",
        children=[
            v.Icon(class_="mr-3", children=["mdi-trash-can-outline"]),
            "Delete the selected regions",
        ],
    )

    # Two button group
    regionsBtnsGroup = v.Layout(
        class_="ma-3 d-flex flex-row",
        children=[validateRegionBtn, v.Spacer(), deleteAllRegionsBtn],
    )
    regionsBtnsView = widgets.VBox([regionsBtnsGroup])
    return validateRegionBtn, deleteAllRegionsBtn, regionsBtnsView

def createNewFeatureRuleGUI(X: pd.DataFrame, newRule, column, binNumber, fig_size):
    newRule[0] = float(newRule[0])
    newRule[4] = float(newRule[4])
    new_valider_change = v.Btn(
        class_="ma-3",
        children=[
            v.Icon(class_="mr-2", children=["mdi-check"]),
            "Validate the change",
        ],
    )

    new_slider_skope = v.RangeSlider(
        class_="ma-3",
        v_model=[newRule[0]  , newRule[4]  ],
        min=newRule[0]  ,
        max=newRule[4]  ,
        step=0.01,
        label=newRule[2],
    )

    new_histogram = FigureWidget(
        data=[
            Histogram(
                X,
                bingroup=1,
                nbinsx=binNumber,
                marker_color="grey",
            )
        ]
    )
    new_histogram.update_layout(
        barmode="overlay",
        bargap=0.1,
        width=0.9 * int(fig_size),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
    )

    new_histogram.add_trace(
        Histogram(
            X,
            bingroup=1,
            nbinsx=binNumber,
            marker_color="LightSkyBlue",
            opacity=0.6,
        )
    )
    new_histogram.add_trace(
        Histogram(
            X,
            bingroup=1,
            nbinsx=binNumber,
            marker_color="blue",
        )
    )
    return new_valider_change, new_slider_skope, new_histogram

def createSettingsMenu(children, text):
    settingsMenu = v.Menu(
        v_slots=[
            {
                "name": "activator",
                "variable": "props",
                "children": v.Btn(
                    v_on="props.on",
                    icon=True,
                    size="x-large",
                    children=[v.Icon(children=["mdi-cogs"], size="large")],
                    class_="ma-2 pa-3",
                    elevation="3",
                ),
            }
        ],
        children=[
            v.Card(
                class_="pa-4",
                rounded=True,
                children=[children],
                min_width="500",
            )
        ],
        v_model=False,
        close_on_content_click=False,
        offset_y=True,
    )

    settingsMenu.v_slots[0]["children"].children = [
        wrap_in_a_tooltip(
            settingsMenu.v_slots[0]["children"].children[0],
            text,
        )
    ]
    return settingsMenu

def computeExplanationsCard(ourSHAPProgLinearColumn, ourLIMEProgLinearColumn):
    widget = v.Card(
        class_="m-0 p-0",
        elevation="0",
        children=[
            v.Tabs(
                class_="w-100",
                v_model="tabs",
                children=[
                    v.Tab(value="one", children=["SHAP (computed)"]),
                    v.Tab(value="two", children=["LIME (computed)"]),
                ],
            ),
            v.CardText(
                class_="w-100",
                children=[
                    v.Window(
                        class_="w-100",
                        v_model="tabs",
                        children=[
                            v.WindowItem(value=0, children=[ourSHAPProgLinearColumn]),
                            v.WindowItem(value=1, children=[ourLIMEProgLinearColumn]),
                        ],
                    )
                ],
            ),
        ],
    )
    return widget

def create_slide_dataset(name, i, type, taille, commentaire, sensible, infos):
    types=['None', 'int64', 'float64', 'str', 'bool']
    infos = v.Card(
        class_="ma-0 pa-0",
        elevation=0,
        children=[
            v.CardText(
                children=[
                    v.Html(tag="h3", children=["Min: " + str(infos[0])]),
                    v.Html(tag="h3", children=["Max: " + str(infos[1])]),
                    v.Html(tag="h3", children=["Mean: " + str(infos[2])]),
                    v.Html(tag="h3", children=["Standard deviation: " + str(infos[3])])
                ]
            ),]
    )
    if str(type) not in types:
        types.append(str(type))
    slide = v.SlideItem(
        style_="max-width: 12%; min-width: 12%;",
        children=[
            v.Card(
                elevation=3,
                class_="grow ma-2",
                children=[
                    v.CardTitle(
                        class_="mx-5 mb-0 pb-0",
                        children=[v.TextField(v_model=name, class_="mb-0 pb-0", label=f"Name: (init: {name})", value=i)]
                    ),
                    v.CardText(class_="d-flex flex-column align-center", children=[
                        v.Menu(
                            location="top",
                            v_slots=[
                                {
                                    "name": "activator",
                                    "variable": "props",
                                    "children": v.Btn(
                                        class_="mt-0 pt-0 mb-5",
                                        v_on="props.on",
                                        icon=True,
                                        size="x-large",
                                        children=[v.Icon(children=["mdi-information-outline"])],
                                        elevation=2,
                                    ),
                                }
                            ],
                            children=[
                                v.Card(
                                    class_="pa-4",
                                    rounded=True,
                                    children=[infos],
                                )
                            ],
                            v_model=False,
                            close_on_content_click=True,
                            offset_y=True,
                        ),
                        v.Row(class_='d-flex flex-row justify-center',
                            children=[
                                v.Select(
                                items=types,
                                v_model=str(type),
                                label=f"Type: (init: {type})",
                                style_="max-width: 40%",
                                value=i
                                ),
                                v.Btn(
                                    icon=True,
                                    elevation=2,
                                    class_="mt-2 ml-2",
                                    children=[wrap_in_a_tooltip(v.Icon(children=["mdi-check"]), "Validate the change")],
                                    value=i
                                ),]),
                        v.Checkbox(
                            v_model=sensible,
                            label="Sensible feature",
                            color="red",
                            class_ = str(i)
                        ),
                        v.Textarea(
                            variant="outlined",
                            v_model=commentaire,
                            label="Comments:",
                            style_="max-width: 90%",
                            rows="3",
                            value=i
                            ),
                        v.Html(
                            tag="p",
                            children=[f"{i}/{taille}"],
                        )
                    ]),
                ],
            )
        ],
    )
    if sensible:
        slide.children[0].color = "red lighten-5"
    return slide


class RuleVariableRefiner :
    """
        A RuleVariableRefiner is a piece of GUI (accordionGrp) that allows the user to refine a rule by selecting a variable and values for this variable.
        It displays the distribution for this variable as well as a beswarm of the explained values.
        The user can use the slider to change the rule.


        _widget : the graph of nested widgets that make up the RuleVariableRefiner
        _variable : the variable that is being refined
        _skope_list : a List of v.CardText that contains the rules that are being refined
        _skope_changed : callable of the GUI parent
    """

    def __init__(self, variable : Variable, skope_changed : callable, skope_list: list = None) :
        self._skope_changed = skope_changed
        self._variable  = variable
        self._skope_list = skope_list
        self._widget = v.ExpansionPanels( # accordionGrp # 0
            class_="ma-2 mb-1",
            children=[
                v.ExpansionPanel( # 00
                    disabled = False,
                    children=[
                        v.ExpansionPanelHeader( # 000
                            children=
                            ["X1"]
                            ),
                        v.ExpansionPanelContent( # 001
                            children=[
                                widgets.HBox( # accordion # 001 0 
                                    [ 
                                        widgets.VBox( # histoCtrl # 001 00
                                            [   
                                                v.Layout( # skopeSliderGroup # 001 000 
                                                    children=[
                                                        v.TextField( # 001 000 0 
                                                            style_="max-width:100px",
                                                            v_model=1, # min value of the slider
                                                            hide_details=True,
                                                            type="number",
                                                            density="compact",
                                                        ),
                                                        v.RangeSlider( # skopeSlider # 001 000 1
                                                            class_="ma-3",
                                                            v_model=[-1, 1],
                                                            min=-10e10,
                                                            max=10e10,
                                                            step=0.01,
                                                        )                                                                    
                                                        ,
                                                        v.TextField(  # 001 000 2
                                                            style_="max-width:100px",
                                                            v_model=5, # max value of the slider
                                                            hide_details=True,
                                                            type="number",
                                                            density="compact",
                                                            step="0.1",
                                                        ),
                                                    ],
                                                ),
                                                FigureWidget( # histogram # 001 001
                                                    data=[ 
                                                        Histogram(
                                                            x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')),
                                                            bingroup=1, 
                                                            nbinsx=50, 
                                                            marker_color="grey"
                                                            )
                                                    ]
                                                ),
                                                widgets.HBox( # validateSkopeChangeBtnAndCheck  # 001 002
                                                    [
                                                        v.Btn( # validateSkopeChangeBtn # 001 002 0
                                                            class_="ma-3",
                                                            children=[
                                                                v.Icon(class_="mr-2", children=["mdi-check"]),
                                                                "Validate the changes",
                                                            ],
                                                        ), 
                                                        v.Checkbox(  # realTimeUpdateCheck # 001 002 1
                                                                v_model=False, label="Real-time updates on the figures", class_="ma-3"
                                                            )
                                                    ]
                                                )
                                            ] # end VBox # 001 00
                                        ),
                                        widgets.VBox( # beeswarmGrp #001 01
                                            [
                                                v.Row( # bs1ColorChoice # 001 010
                                                    class_="pt-3 mt-0 ml-4",
                                                    children=[
                                                        "Value of Xi",
                                                        v.Switch( # 001 0101
                                                            class_="ml-3 mr-2 mt-0 pt-0",
                                                            v_model=False,
                                                            label="",
                                                        ),
                                                        "Current selection",
                                                    ],
                                                ),
                                                FigureWidget( # beeswarm # 001 011
                                                    data=[Scatter(
                                                        x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')), 
                                                        y=[0] * 100, 
                                                        mode="markers")]
                                                )
                                            ],
                                            layout=Layout(margin = "0px 0px 0px 20px")
                                            ), 
                                        v.Col( # rightSide # 001 02
                                            children=[
                                                v.Btn( # deleteSkopeBtn # 001 020
                                                    class_="ma-2 ml-4 pa-1",
                                                    elevation="3",
                                                    icon=True,
                                                    children=[v.Icon(children=["mdi-delete"])],
                                                    disabled=True,
                                                ),
                                                v.Checkbox( # isContinuousChck # 001 021
                                                    v_model=True, 
                                                    label="is continuous?"
                                                    )
                                                ],
                                            class_="d-flex flex-column align-center justify-center",
                                        )
                                    ],
                                    layout=Layout(align_explanationsMenuDict="center"),
                                ) # End HBox 001 0
                                ]               
                        ), # End ExpansionPanelContent 001
                    ]
                ), # End ExpansionPanel 00 
            ] 
        )
        # We vire the input event on the skopeSlider (0010001)
        widget_at_address(self._widget, "0010001").on_event("input", skope_rule_changed)
        # We vire the click event on validateSkopeChangeBtn (010020)
        widget_at_address(self._widget, "010020").on_event("click", skope_slider_changed)

    def hide_beeswarm(self, hide : bool):
        # We retrieve the beeswarmGrp (VBox)
        widget_at_address(self._widget, "00101").disabled = hide
    

    def skope_slider_changed(*change):
        # we just call skope_changed @GUI
        self._skope_changed()


    def get_widget(self) -> v.ExpansionPanels:
        return self._widget


    def redraw_both_graphs(self):
        # We update the refiner's histogram :
        with widget_at_address(self._widget, "001001").batch_update():
            widget_at_address(self._widget, "001001").data[0].x = \
                self._ds.get_full_values()[self._selection.get_vs_rules()[self._variable.get_col_index][2]]
        
        # We update the refiner's beeswarm :
        if widget_at_address(self._widget, "001011").v_model : # TODO Why do we check ?
            with widget_at_address(self._widget, "001011").batch_update():
                pass
                # TODO to understand
                # TODO : maybe the refiner could handle its colors itself
                # y_color = [0] * self._gui._ds.get_length()
                # if i == rule_index:
                #     indexs = (
                #         self._ds.get_full_values()
                #         .index[
                #             self._ds.getXValues()[
                #                 self._selection.getVSRules()[i][2]
                #             ].between(min, max)
                #         ]
                #         .tolist()
                #     )
                # else:
                #     indexs = (
                #         self._ds.get_full_values().index[
                #             self._ds.getXValues()[
                #                 self._selection.getVSRules()[i][2]
                #             ].between(
                #                 self._selection.getVSRules()[i][0],
                #                 self._selection.getVSRules()[i][4],
                #             )
                #         ].tolist()
                #     )
                # for j in range(
                #     len(self._xds.get_full_values(self._explanationES[0]))
                # ):
                #     if j in total_list:
                #         y_color[j] = "blue"
                #     elif j in indexs:
                #         y_color[j] = "#85afcb"
                #     else:
                #         y_color[j] = "grey"
                # widget_at_address(self._widget, "001011").data[0].marker.color = y_color
        
    def skope_rule_changed(widget, event, data):
        # when the value of a slider is modified, the histograms and graphs are modified
        if widget.__class__.__name__ == "RangeSlider":
            # We set the text before the slider (0010000) to the min value of the slider
            widget_at_address(self._widget, "0010000").v_model = widget_at_address(self._widget, "0010001").v_model[0]
            # We set the text after the slider (0010002) to the min value of the slider
            widget_at_address(self._widget, "0010002").v_model = widget_at_address(self._widget, "0010001").v_model[1]
        else:
            if (
                widget_at_address(self._widget, "0010000").v_model == ""
                or widget_at_address(self._widget, "0010002").v_model == ""
            ):
                # If no value, we return
                return
            else:
                # Inversely, we set the slider to the values after the texts
                widget_at_address(self._widget, "0010001").v_model = [
                    float(widget_at_address(self._widget, "0010000").v_model), # min
                    float(widget_at_address(self._widget, "0010002").v_model), # max
                ]
        
        new_list = [
            g
            for g in list(
                self._gui.get_dataset().get_full_values()[self._gui.get_selection().getVSRules()[0][2]].values
            )
            if g >= widget_at_address(self._widget, "0010001").v_model[0] and g <= widget_at_address(self._widget, "0010001").v_model[1]
        ]

        # We updat the histogram (001001)
        with widget_at_address(self._widget, "001001").batch_update():
            widget_at_address(self._widget, "001001").data[1].x = new_list
        
        # TODO : what is _activate_histograms
        if self._activate_histograms:
            self._gui.update_histograms_with_rules(widget_at_address(self._widget, "0010001").v_model[0], widget_at_address(self._widget, "0010001").v_model[1], 0)

        # If realTimeUpdateCheck (0010021) is checked :
        if widget_at_address(self._widget, "0010021").v_model:
            # We update rules with the skopeSlider (0010001) values  
            self._selection.getVSRules()[0][0] = float(
                deepcopy(widget_at_address(self._widget, "0010021").v_model[0]) # min
            )
            self._selection.getVSRules()[0][4] = float(
                deepcopy(widget_at_address(self._widget, "0010021").v_model[1]) # max
            )
            widget_at_address(self._gui.get_app_graph(), "30500101").children = create_rule_card(
                self._selection.ruleListToStr()
            ) 
            self._gui.update_histograms_with_rules()

    def get_class_selector(self, min : int = 1, max : int = -1, fig_size :int =700) -> v.Layout :
            valuesList = list(set(self._gui.get_dataset().getVariableValue(self._variable)))
            widgetList = []
            for value in valuesList:
                if value <= max and value >= min:
                    inside = True
                else:
                    inside = False
                widget = v.Checkbox(
                    class_="ma-4",
                    v_model=inside,
                    label=str(value).replace("_", " "),
                )
                widgetList.append(widget)
            row = v.Row(class_ = "ml-6 ma-3", children = widgetList)
            text = v.Html(tag="h3", children=["Select the values of the feature " + self._variable.getSymbol()])
            return v.Layout(class_= "d-flex flex-column align-center justify-center", style_="width: "+str(int(fig_size)-70)+"px; height: 303px", children=[v.Spacer(), text, row])

    def real_time_changed(*args):
        """ If changed, we invert the validate button """
        widget_at_address(self._widget, "0010020").disabled = not widget_at_address(self._widget, "0010020").disabled
    
        # See realTimeUpdateCheck (0010021)
        widget_at_address(self._widget, "0010021").on_event("change", real_time_changed)

    def beeswarm_color_changed(*args): 
        """ If changed, we invert the showScake value """
        # See beeswarm :
        show_scale = widget_at_address(self._widget, "001011").data[0].marker[showscale]
        show_scale = widget_at_address(self._widget, "001011").update_traces(marker=dict(showscale=not show_scale))
    
        # See bsColorChoice[,v.Switch] (0010101)
        widget_at_address(self._widget, "0010101").on_event("change", beeswarm_color_changed)


    def continuous_check_changed(widget, event, data): 
        features = [
            self._selection.getVSRules()[i][2]
            for i in range(len(self._selection.getVSRules()))
        ]
        aSet = []
        for i in range(len(features)):
            if features[i] not in aSet:
                aSet.append(features[i])

        index = features.index(aSet[2])
        if widget.v_model :
            # TODO : understand
            # We define accordion (0010) children as histoCtrl (00100) + list (accordion(0010).children[1])
            widget_at_address(self._widget, "0010").children = [widget_at_address(self._widget, "00100")] + list(widget_at_address(self._widget, "0010").children[1:])
            count = 0
            for i in range(len(self._gui.get_selection().getVSRules())):
                if (
                    self._gui.get_selection().getVSRules()[i - count][2]
                    == self._selection.getVSRules()[index][2]
                    and i - count != index
                ):
                    self._gui.get_selection().getVSRules().pop(i - count)
                    count += 1
            # We set skopeSlider (0010001) values
            self._gui.get_selection().getVSRules()[index][0] = widget_at_address(self._widget, "0010001").v_model[0]
            self._gui.get_selection().getVSRules()[index][4] = widget_at_address(self._widget, "0010001").v_model[1]
            
            self._skope_list = create_rule_card(self._selection.ruleListToStr())
        else:
            class_selector = self.get_class_selector()
            widget_at_address(self._widget, "0010").children = [class_selector] + list(
                widget_at_address(self._widget, "0010").children[1:]
            )
            aSet = []
            for i in range(len(self.get_class_selector().children[2].children)):
                if class_selector.children[2].children[i].v_model:
                    aSet.append(
                        int(class_selector.children[2].children[i].label)
                    )
            if len(aSet) == 0:
                widget.v_model = True
                return
            column = deepcopy(self._gui.get_selection().getVSRules()[index][2])
            count = 0
            for i in range(len(self._gui.get_selection().getVSRules())):
                if self._gui.get_selection().getVSRules()[i - count][2] == column:
                    self._gui.get_selection().getVSRules().pop(i - count)
                    count += 1
            ascending = 0  
            for item in aSet:
                self._gui.get_selection().getVSRules().insert(
                    index + ascending, [item - 0.5, "<=", column, "<=", item + 0.5]
                )
                ascending += 1
            self._skope_list = create_rule_card(self._gui.get_selection().ruleListToStr())

        # We wire the "change" event on the isContinuousChck (001021)
        widget_at_address(self._widget, "001021").on_event("change", continuous_check_changed)


class HighDimExplorer :
    """
        An HighDimExplorer displays one or several high dim Dataframes on a scatter plot.
        It uses several dimension reduction techniques, through the DimReduction class.
        It can display in or 2 dimensions.

        Implemntation details :
        It handes projections computation itself when needed.
        But, it asks GUI when another dataframe is asked for.
        It stores dataframes with the ProjectedValues class.
        It stored the current projection method but not the dimension
        Attributes are mostly privates (underscorred) since they are not meant to be used outside of the class.

        Attributes :
        _gui : GUI parent
        _pv_list: list # a list of one or several ProjectedValues (PV)
        _current_pv : int, stores the index of current PV in the _pv_list
        selection_changed = a GUi's callable that is called when the selection changes
        new_explain_method_selected = a GUi's callable that is called when the explanation changes

        Widgets :
        _values_select: v.Select
            None if len(_pv_list) = 1
            The labels of its items are transmitted by GUI at construction
        _compute_menu : v.Menu
            None if len(_pv_list) = 1
            Triggers the provision of other dataframes
        _figure : FigureWidget
            Plotly scatter plot
        _projection_select : v.Select, with mutliple dimreduc methods
        _projection_sliders : v.Dialogs, optional parameters for dimreduc methods

    """

    def __init__(self, gui: GUI, space_name:str, values_list : list, label_list : list, init_proj: int, init_dim: int, selection_changed: callable, new_explain_method_selected: callable = None):
        """
        Instantiate a new HighDimExplorer.

        Selected parameters :
            values_list : list of pd.Dataframes. Stored in ProjectedValues
            space_name : str, the name of the space explored. Stored in a widget
            label_list : list of str. 
                Stored in a Select widget
                if len(label_list) = 1, the widget is not created, and the label is ignord
            init_proj, init_dim : int, int, used to initialize widgets
        """ 
        self._gui = gui
        
        self._pv_list = []
        for values in values_list :
            if values is not None :
                self._pv_list.append(ProjectedValues(values))
                logger.debug(f"HDE.init: new PV added to my list")
            else :
                self._pv_list.append(None)
        
        self._current_pv = 0
        for i in range(len(self._pv_list)):
            if self._pv_list[i] is not None :
                self._current_pv = i
                logger.debug(f"HDE.init: _current_pv set to {i}")
                break
        
        if len(self._pv_list) == 1 :
            self._values_select = None
            self._compute_menu = None
        else:
            select_items = {}
            for i in range(len(self._pv_list)):
                select_items.apped({'text': label_list[i], 'disabled': self._pv_list[i] is None})

            self._values_select = v.Select(
                label="Explanation method",
                items= select_items,
                class_="ma-2 mt-1 ml-6",
                style_="width: 150px",
                disabled= False,
                )
            self._compute_menu(
                v.Menu(
                    
            )

        self._projection_select = v.Select(
            label="Projection in the " + space_name,
            items=DimReducMethod.dimreduc_methods_as_str_list(),
            style_="width: 150px",
        )
        # Since HDE is responsible for storing its current proj, we check init value :
        if init_proj not in DimReducMethod.dimreduc_methods_as_list() :
            raise ValueError(f"HDE.init: {init_proj} is not a valid projection method code")
        self._projection_select.v_model = DimReducMethod.dimreduc_method_as_str(init_proj)
        self._projection_slides = widgets.VBox(
            """ A GUI for the user to set the current proj parameters """
            # TODO : are the below paramaters suited for all proj methods ?
            # It looks like it's PaCMAP specific
            [   
            v.Slider(
                v_model=10, min=5, max=30, step=1, label="Number of neighbours"
            ),
            v.Html(class_="ml-3", tag="h3", children=["machin"]),
            v.Slider(
                v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio"
            ),
            v.Html(class_="ml-3", tag="h3", children=["truc"]),
            v.Slider(
                v_model=2, min=0.1, max=5, step=0.1, label="FP ratio"
            ),
            v.Html(class_="ml-3", tag="h3", children=["bidule"]),
            ],
        )

        self._figure = FigureWidget(
            data=Scatter(
                x=self._get_x(), 
                y=self._get_y(), 
                mode="markers", 
                marker=dict( 
                    color=self._gui.y,
                    colorscale="Viridis",
                    colorbar=dict(
                        title="y",
                        thickness=20
                        ),
                    ), 
                customdata=self._gui.y, 
                hovertemplate= '%{customdata:.3f}'
            )
            )
        self._figure.data[0].on_selection(self.dots_lasso_selected)

    # ---- Methods ------

    def set_dimension(self, dim : int) :
        """
        At init, dim is 2
        At runtime, GUI calls this function, then we update our figure each time
        """
        pass
    
    def redraw(self, opacity_values: pd.Series, color: pd.Series, size: int) :
        proj_values = self._ds.get_proj_values(self._current_projection, self._current_dim)
        with self._figure.batch_update():
                self._figure.data[0].marker.opacity = opacity_values
                self._figure.data[0].marker.color = color
                self._figure.layout.width = size
                self._figure.data[0].customdata = color
                self._figure.data[0].x, self._figure.data[0].y = (projValues[0]), (projValues[1])
                if self._current_dim == DimReducMethod.DIM_THREE:
                    self._figure.data[0].z = projValues[2]

    def dots_lasso_selected(self, trace, points, selector, *args):
        """ Called whenever the user selects dots on the scatter plot """
        # We just use a callback to inform the GUI :
        self.selection_changed(
            Selection(self._ds.get_full_values(), Selection.LASSO),
            config.VS if not self._is_explain_explorer else config.ES
        )

    def show(self) :
        display(self._projection_select)
        display(self._projection_sliders)
        display(self._figure)

# ---------- End of AntakiaExplorer class ----------

def get_splash_graph():
    """
        Returns the splash screen graph.
    """
    return v.Layout(
            class_="d-flex flex-column align-center justify-center",
            children=[
                widgets.Image(
                    value=widgets.Image._load_file_value(files("antakia.assets").joinpath("logo_antakia.png")), layout=Layout(width="230px")
                ), 
                v.Row(
                    style_="width:85%;",
                    children=[
                        v.Col(
                            children=[
                                v.Html(
                                    tag="h3",
                                    class_="mt-2 text-right",
                                    children=["Computation of explanation values"],
                                )
                            ]   
                        ),
                        v.Col(
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
                            ]
                        ),
                        v.Col(
                            children=[
                                v.TextField(
                                    variant="plain",
                                    v_model="0.00% [0/?] - 0m0s (estimated time : /min /s)",
                                    readonly=True,
                                    class_="mt-0 pt-0",
                                    )
                            ]
                        ),
                    ],
                ),
                v.Row(
                    style_="width:85%;",
                    children=[
                        v.Col(
                            children=[
                                v.Html(
                                    tag="h3",
                                    class_="mt-2 text-right",
                                    children=["Computation of dimension reduction values"],
                                )
                            ]
                        ),
                        v.Col(class_="mt-3", children=[
                            v.ProgressLinear(
                                style_="width: 80%",
                                class_="py-0 mx-5",
                                v_model=0,
                                color="primary",
                                height="15",
                                striped=True,
                            )
                        ]),
                        v.Col(
                            children=[
                                v.TextField(
                                    variant="plain",
                                    v_model="0.00% [0/?] - 0m0s (estimated time : /min /s)",
                                    readonly=True,
                                    class_="mt-0 pt-0",
                                    )
                            ]
                        ),
                    ],
                ), 
            ]
        )

def get_app_graph():
    return widgets.VBox(
        [
            v.AppBar( # 0
                elevation="4",
                class_="ma-4",
                rounded=True,
                children=[
                    v.Layout(
                        children=[
                            widgets.Image(
                                value=open(files("antakia.assets").joinpath("logo_ai-vidence.png"), "rb").read(), 
                                height=str(864 / 20) + "px", 
                                width=str(3839 / 20) + "px"
                            )
                            ],
                        class_="mt-1",
                    ),
                    v.Html(tag="h2", children=["AntakIA"], class_="ml-3"), # 01
                    v.Spacer(),
                    v.Btn( # backupBtn # 03 
                        icon=True, children=[v.Icon(children=["mdi-content-save"])], elevation=0
                    ),
                    v.Btn( # settingsBtn # 04
                        icon=True, children=[v.Icon(children=["mdi-tune"])], elevation=0
                    ),
                    v.Dialog( # 05
                        children=[
                            v.Card( # 050
                                children=[
                                    v.CardTitle( # 050 0
                                        children=[
                                            v.Icon(class_="mr-5", children=["mdi-cogs"]),
                                            "Settings",
                                        ]
                                    ),
                                    v.CardText( # 050 1
                                        children=[
                                        v.Row( # 050 10
                                            children=[
                                                v.Slider( # figureSizeSlider # 050 100
                                                    style_="width:20%",
                                                    v_model=700,
                                                    min=200,
                                                    max=1200,
                                                    label="With of both scattered plots (in pixels)",
                                                ), 
                                            widgets.IntText(
                                                value="700", disabled=True, layout=Layout(width="40%")
                                            )
                                            ],
                                            ),
                                        ]
                                        ),
                                        ]
                                    ),
                                ]
                    ),
                    v.Btn( # gotoWebBtn # 06
                        icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
                    ),
                ],
            ), 
            widgets.HBox( # 1
                [
                    v.Row( # 10
                        class_="ma-3",
                        children=[
                            v.Icon(children=["mdi-numeric-2-box"]),
                            v.Icon(children=["mdi-alpha-d-box"]),
                            v.Switch( # Dim switch # 102
                                class_="ml-3 mr-2",
                                v_model=False,
                                label="",
                            ),
                            v.Icon(children=["mdi-numeric-3-box"]),
                            v.Icon(children=["mdi-alpha-d-box"]),
                        ],
                    ),
                    v.Layout( # 11
                        class_="pa-2 ma-2",
                        elevation="3",
                            children=[
                                    v.Icon( # 110
                                        children=["mdi-format-color-fill"], class_="mt-n5 mr-4"
                                    ),
                                    v.BtnToggle( # colorChoiceBtnToggle # 111
                                        mandatory=True,
                                        v_model="Y",
                                        children=[
                                            v.Btn( # 1110
                                                icon=True,
                                                children=[v.Icon(children=["mdi-alpha-y-circle-outline"])],
                                                value="y",
                                                v_model=True,
                                            ),
                                            v.Btn( # 1111
                                                icon=True,
                                                children=[v.Icon(children=["mdi-alpha-y-circle"])],
                                                value="y^",
                                                v_model=True,
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[2]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-minus-box-multiple"])],
                                                value="residual",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[3]
                                                icon=True,
                                                children=[v.Icon(children="mdi-lasso")],
                                                value="current selection",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[4]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-ungroup"])],
                                                value="regions",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[5]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-select-off"])],
                                                value="not selected",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[6]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-star"])],
                                                value="auto",
                                            ),
                                        ],
                                    ),
                                    v.Btn( # opacityBtn # 112
                                        icon=True,
                                        children=[v.Icon(children=["mdi-opacity"])],
                                        class_="ma-2 ml-6 pa-3",
                                        elevation="3",
                                    ),
                                    v.Select( # explanationSelect # 113
                                        label="Explanation method",
                                        items=[
                                            {'text': "SHAP (imported)", 'disabled': True },
                                            {'text': "SHAP (computed)", 'disabled': True },
                                            {'text': "LIME (imported)", 'disabled': True },
                                            {'text': "LIME (computed)", 'disabled': True }
                                            ],
                                        class_="ma-2 mt-1 ml-6",
                                        style_="width: 150px",
                                        disabled = False,
                                    )
                                ],
                    ),
                    v.Layout( # 12
                        class_="mt-3",
                        children=[
                            widgets.HBox( # 120
                                [
                                    v.Select( # projSelectVS # 1200
                                        label="Projection in the VS :",
                                        items=DimReducMethod.dimreduc_methods_as_str_list(),
                                        style_="width: 150px",
                                    ),
                                    v.Layout( # 1201
                                        children=[
                                            v.Menu( # projSettingsMenuVS # 12010
                                                v_slots=[
                                                {
                                                    "name": "activator",
                                                    "variable": "props",
                                                    "children": v.Btn(
                                                        v_on="props.on",
                                                        icon=True,
                                                        size="x-large",
                                                        children=[v.Icon(children=["mdi-cogs"], size="large")],
                                                        class_="ma-2 pa-3",
                                                        elevation="3",
                                                    ),
                                                }
                                                ],
                                                children=[
                                                    v.Card( # app_graph.children[1].children[2].children[0].children[1].children[0].childen[0] 
                                                        class_="pa-4",
                                                        rounded=True,
                                                        children=[
                                                            widgets.VBox([ # ProjVS sliders
                                                                v.Slider(
                                                                    v_model=10, min=5, max=30, step=1, label="Number of neighbours"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                v.Slider(
                                                                    v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                v.Slider(
                                                                    v_model=2, min=0.1, max=5, step=0.1, label="FP ratio"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
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
                                    widgets.HBox( # app_graph.children[1].children[2].children[0].children[2]
                                        [ 
                                        v.ProgressCircular( # VSBusyBox # app_graph.children[1].children[2].children[0].children[2].children[0]
                                            indeterminate=True, color="blue", width="6", size="35", class_="mx-4 my-3"
                                        )
                                        ]),
                                ]
                            ),
                            widgets.HBox(  # 121
                                [
                                    v.Select( # projSelectES # 121 0
                                        label="Projection in the ES :",
                                        items=DimReducMethod.dimreduc_methods_as_str_list(),
                                        style_="width: 150px",
                                    ),
                                    v.Layout( # 1211
                                        children=[
                                        ]),
                                    widgets.HBox( # 121 2
                                        [
                                        v.ProgressCircular( # ESBusyBox # 121 20
                                            indeterminate=True, color="blue", width="6", size="35", class_="mx-4 my-3"
                                        )
                                        ]),
                                ]
                            ),
                        ],
                    )
                ],
                layout=Layout(
                    width="100%",
                    display="flex",
                    flex_flow="row",
                    justify_content="space-around",
                ),
            ),
            widgets.VBox( # 2
                [
                widgets.HBox( # 20
                    [
                    widgets.VBox( # 200
                            [
                            widgets.HTML("<h3>Values Space<h3>"), # 2000
                            FigureWidget( # _leftVSFigure or _ae_vs.get_figure_widget() # 2001
                                data=Scatter(
                                    x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')),
                                    y=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))['A'],
                                    mode="markers", 
                                    marker= dict( 
                                        color=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))['A'],
                                        colorscale="Viridis",
                                        colorbar=dict(
                                            title="y",
                                            thickness=20,
                                        ),
                                    ), 
                                    customdata=dict( 
                                        color=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))['A'],
                                        colorscale="Viridis",
                                        colorbar=dict(
                                            title="y",
                                            thickness=20,
                                        ),
                                    )["color"],
                                    hovertemplate = '%{customdata:.3f}')
                            )
                            ],
                            layout=Layout(
                                display="flex", align_items="center", margin="0px 0px 0px 0px"
                            ),
                    )
                    ,
                    widgets.VBox( #  #201
                            [
                            widgets.HTML("<h3>Explanations Space<h3>"),  # 201 0
                            FigureWidget( #_rightESFigure # 201 1
                                data=Scatter(
                                    x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')),
                                    y=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))['A'],
                                mode="markers", 
                                marker=
                                    dict( 
                                        color=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))['A'],
                                        colorscale="Viridis"
                                    ), 
                                customdata=dict( 
                                    color=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))['A'],
                                    colorscale="Viridis",
                                    colorbar=dict(
                                        title="y",
                                        thickness=20,
                                    ),
                                    )["color"],
                                hovertemplate = '%{customdata:.3f}')
                            )
                            ],
                            layout=Layout(
                                display="flex", align_items="center", margin="0px 0px 0px 0px"
                            )
                    )
                    ],
                    layout=Layout(width="100%")
                    )
            ]    
            ),
            v.Container( # antakiaMethodCard # 3
                fluid = True,
                children=[
                    v.Tabs( # 30
                        v_model=0, # default active tab
                        children=
                        [
                            v.Tab(children=["1. Selection"]),  # 300
                            v.Tab(children=["2. Refinement"]), # 301
                            v.Tab(children=["3. Sub-model"]), # 302
                            v.Tab(children=["4. Regions"]) # 303
                        ] 
                        + 
                        [
                            v.TabItem(  # Tab 1) = tabOneSelectionColumn ? Selection # 304
                                children=[
                                    v.Card( # selectionCard # 304 0
                                        class_="ma-2",
                                        elevation=0,
                                        children=[
                                            v.Layout( # 304 00
                                                children=[
                                                    v.Icon(children=["mdi-lasso"]), # 304 000
                                                    v.Html( # 304 001
                                                        class_="mt-2 ml-4",
                                                        tag="h4",
                                                        children=[
                                                            "0 point selected : use the lasso tool on the figures above or use the auto-selection tool below"
                                                        ],
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    v.ExpansionPanels( # out_accordion # 304 1
                                        class_="ma-2",
                                        children=[
                                            v.ExpansionPanel( # 304 10
                                                children=[
                                                    v.ExpansionPanelHeader(children=["Data selected"]), # 304 100
                                                    v.ExpansionPanelContent( # 304 101
                                                        children=[
                                                        v.Alert( # out_selec_all # 304 101 0
                                                            max_height="400px",
                                                            style_="overflow: auto",
                                                            elevation="0",
                                                            children=[
                                                                v.Row( # 304 101 00
                                                                    class_="d-flex flex-row justify-space-between",
                                                                    children=[
                                                                        v.Layout( # out_selec # 304 101 000
                                                                            style_="min-width: 47%; max-width: 47%",
                                                                            children=[
                                                                                v.Html( # out_selec # 
                                                                                    tag="h4",
                                                                                    children=["Select points on the figure to see their values ​​here"],
                                                                                )
                                                                            ],
                                                                        ),
                                                                        v.Divider(class_="ma-2", vertical=True), # 304 101 001
                                                                        v.Layout( # out_selec_SHAP # 304 101 002
                                                                            style_="min-width: 47%; max-width: 47%",
                                                                            children=[
                                                                                v.Html( # 304 101 002 0
                                                                                    tag="h4",
                                                                                    children=[
                                                                                        "Select points on the figure to see their SHAP values ​​here"
                                                                                    ],
                                                                                )
                                                                            ],
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                        ),
                                                        ]),
                                                ]
                                            )
                                        ],
                                    ),
                                    v.Layout( # clusterGrp # 304 2
                                        class_="d-flex flex-row",
                                        children=[
                                            v.Btn( # findClusterBtn # 304 20
                                                class_="ma-1 mt-2 mb-0",
                                                elevation="2",
                                                children=[v.Icon(children=["mdi-magnify"]), "Find clusters"],
                                            ),
                                            v.Checkbox( # clusterCheck # 304 21
                                                v_model=True, label="Optimal number of clusters :", class_="ma-3"
                                            ),
                                            v.Slider( # clustersSlider # 304 22
                                                style_="width : 30%",
                                                class_="ma-3 mb-0",
                                                min=2,
                                                max=20,
                                                step=1,
                                                v_model=3,
                                                disabled=True,
                                            ),
                                            v.Html( # clustersSliderTxt # 304 23
                                                tag="h3",
                                                class_="ma-3 mb-0",
                                                children=["Number of clusters #"],
                                            ),
                                        ],
                                    ),
                                    v.ProgressLinear( # loadingClustersProgLinear # 304 3
                                        indeterminate=True, class_="ma-3", style_="width : 100%"
                                    ),
                                    v.Row( # clusterResults # 304 4
                                        children=[
                                            v.Layout(
                                                class_="flex-grow-0 flex-shrink-0",
                                                children=[
                                                    v.Btn(class_="d-none", elevation=0, disabled=True
                                                    )], # 304 40
                                            ),
                                            v.Layout(  # 304 41
                                                class_="flex-grow-1 flex-shrink-0",
                                                children=[ # A v.DataTable is inserted here by the app. Will be : # cluster_results_table # 304 420
                                                    ],
                                            ),
                                        ],
                                    ),
                                    v.Layout( # magicGUI 304 5
                                        class_="d-flex flex-row justify-center align-center",
                                        children=[
                                            v.Spacer(), # 304 50
                                            v.Btn( # magicBtn # findClusterBtn # 304 51
                                                    class_="ma-3",
                                                    children=[
                                                        v.Icon(children=["mdi-creation"], class_="mr-3"),
                                                        "Magic button",
                                                    ],
                                            ),
                                            v.Checkbox( # # magicCheckBox 304 52
                                                v_model=True, label="Demonstration mode", class_="ma-4"), 
                                            v.TextField( # 304 53
                                                class_="shrink",
                                                type="number",
                                                label="Time between the steps (ds)",
                                                v_model=10,
                                            ),
                                            v.Spacer(), # 304 54
                                        ],
                                    )
                                ]
                            ), 
                            v.TabItem( # Tab 2) = tabTwoSkopeRulesColumn ? Refinement # 305
                                children=[
                                    v.Col( # 305 0
                                        children=[
                                            widgets.VBox( # skopeBtnsGrp # 305 00
                                                [
                                                v.Layout( # skopeBtns # 305 000
                                                    class_="d-flex flex-row",
                                                    children=[
                                                        v.Btn( # validateSkopeBtn # 305 000 0
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(class_="mr-2", children=["mdi-auto-fix"]),
                                                                "Skope-Rules",
                                                            ],
                                                        ),
                                                        v.Btn( # reinitSkopeBtn # 305 000 1
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(class_="mr-2", children=["mdi-skip-backward"]),
                                                                "Come back to the initial rules",
                                                            ],
                                                        ),
                                                        v.Spacer(), # 305 000 2
                                                        v.Checkbox( # beeSwarmCheck # 305 000 3
                                                            v_model=True,
                                                            label="Show Shapley's beeswarm plots",
                                                            class_="ma-1 mr-3",
                                                        )
                                                        ,
                                                    ],
                                                ),
                                                v.Layout( # skopeText # skopeBtns # 305 001
                                                    class_="d-flex flex-row",
                                                    children=[
                                                        v.Card( # ourVSSkopeText # skopeBtns # 305 001 0
                                                            style_="width: 50%;",
                                                            class_="ma-3",
                                                            children=[
                                                                v.Row(  # 30500100
                                                                    class_="ml-4",
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), # 305 010 00
                                                                        v.CardTitle(children=["Rules applied to the Values Space"]), # 305 010 01
                                                                        v.Spacer(), # 305 010 02
                                                                        v.Html( # 305 010 03
                                                                            class_="mr-5 mt-5 font-italic",
                                                                            tag="p",
                                                                            children=["precision = /"],
                                                                        ),
                                                                    ],
                                                                ),
                                                                v.Card( # ourVSSkopeCard # 305 001 01
                                                                    class_="mx-4 mt-0",
                                                                    elevation=0,
                                                                    children=[
                                                                        v.CardText(
                                                                            children=[
                                                                                v.Row(
                                                                                    class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                                                                                    children=[
                                                                                        "Waiting for the skope-rules to be applied...",
                                                                                    ],
                                                                                )
                                                                            ]
                                                                        )
                                                                    ],
                                                                )
                                                            ],
                                                        ),
                                                        v.Card( # ourESSkopeText # 305 001 1
                                                            style_="width: 50%;",
                                                            class_="ma-3",
                                                            children=[
                                                                v.Row( # 305 001 10
                                                                    class_="ml-4",
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), # 305 001 100
                                                                        v.CardTitle(children=["Rules applied on the Explanatory Space"]), 
                                                                        v.Spacer(),
                                                                        v.Html( # 305 001 103
                                                                            class_="mr-5 mt-5 font-italic",
                                                                            tag="p",
                                                                            children=["precision = /"],
                                                                        ),
                                                                    ],
                                                                ),
                                                                v.Card( # ourESSkopeCard # 305 001 11
                                                                    class_="mx-4 mt-0",
                                                                    elevation=0,
                                                                    # style_="width: 100%;",
                                                                    children=[
                                                                        v.CardText(
                                                                            children=[
                                                                                v.Row(
                                                                                    class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                                                                                    children=[
                                                                                        "Waiting for the Skope-rules to be applied...",
                                                                                    ],
                                                                                )
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                        )
                                                    ]
                                                ) # End v.Layout / skopeText
                                            ]
                                            ), # End VBox / skopeBtnsGrp
                                            widgets.VBox( # skopeAccordion # 305 01
                                                children=[ # RuleVariableRefiner objects are inserted here by the app
                                                ],
                                                layout=Layout(width="100%", height="auto"),
                                            ), # End of VBox 30501
                                            v.Row( #addButtonsGrp # 305 02
                                                children=[
                                                    v.Btn( # addSkopeBtn # 305 020
                                                        class_="ma-4 pa-2 mb-1",
                                                        children=[v.Icon(children=["mdi-plus"]), "Add a rule"],
                                                    ), 
                                                    v.Select( # addAnotherFeatureWgt # 305 021
                                                        class_="mr-3 mb-0",
                                                        explanationsMenuDict=["/"],
                                                        v_model="/",
                                                        style_="max-width : 15%",
                                                    ), 
                                                    v.Spacer(), # 305 022
                                                    v.Btn( # addMapBtn # 305 023
                                                        class_="ma-4 pa-2 mb-1",
                                                        children=[v.Icon(class_="mr-4", children=["mdi-map"]), "Display the map"],
                                                        color="white",
                                                        disabled=True,
                                                    ),
                                                    ]
                                            ),
                                            ]
                                    )
                                ]
                            ), 
                            v.TabItem( # Tab 3) = tabThreeSubstitutionVBox ? # 306
                                children=[
                                        widgets.VBox( # 306 0
                                            [
                                                v.ProgressLinear( # loadingModelsProgLinear # 306 00
                                                    indeterminate=True,
                                                    class_="my-0 mx-15",
                                                    style_="width: 100%;",
                                                    color="primary",
                                                    height="5",
                                                ), 
                                                v.SlideGroup( # subModelslides # 306 01
                                                    v_model=None,
                                                    class_="ma-3 pa-3",
                                                    elevation=4,
                                                    center_active=True,
                                                    show_arrows=True,
                                                    children=
                                                    [
                                                        v.SlideItem( # 306 010 # dummy SlideItem. Will be replaced by the app
                                                            # style_="width: 30%",
                                                            children=[
                                                                v.Card(
                                                                    class_="grow ma-2",
                                                                    children=[
                                                                        v.Row(
                                                                            class_="ml-5 mr-4",
                                                                            children=[
                                                                                v.Icon(children=["a name"]),
                                                                                v.CardTitle(
                                                                                    children=["model foo"]
                                                                                ),
                                                                            ],
                                                                        ),
                                                                        v.CardText(
                                                                            class_="mt-0 pt-0",
                                                                            children=["Model's score"],
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
                            v.TabItem( # Tab 4) = tabFourRegionListVBox # 307
                                children=[
                                    v.Col( # 307 0
                                    children=[
                                        widgets.VBox( # 307 00
                                            [
                                                v.Btn( # 307 000
                                                        class_="ma-4 pa-2 mb-1",
                                                        children=[v.Icon(class_="mr-4", children=["mdi-map"]), "Validate the region"],
                                                        color="white",
                                                        disabled=True,
                                                )
                                        ]
                                        ),         
                                    ]
                                    )
                                ]
                            )
                        ]
                    )
                ],
                class_="mt-0",
                outlined=True
            )
        ]
    )