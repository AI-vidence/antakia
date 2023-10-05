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
from plotly.graph_objects import FigureWidget, Histogram, Scatter

from antakia.data import Dataset, DimReducMethod, ExplanationDataset, ExplanationMethod, Model, Variable
from antakia.utils import confLogger
from antakia.gui import GUI

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()


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
                        children=[model.__class__.__name__]"]
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

def wrap_in_a_tooltip(widget, text) -> v.Tooltip:
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
    return wrapped_widget


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
    expValues = expDS.getFullValues(explainMethod)
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

def 

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

def createNewFeatureRuleGUI(gui, newRule, column, binNumber, fig_size):
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
                x=gui.atk.dataset.X[column].values,
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
            x=gui.atk.dataset.X[column].values,
            bingroup=1,
            nbinsx=binNumber,
            marker_color="LightSkyBlue",
            opacity=0.6,
        )
    )
    new_histogram.add_trace(
        Histogram(
            x=gui.atk.dataset.X[column].values,
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
        _gui : the parent GUI object
        _variable : the variable that is being refined
        _skope_list : a List of v.CardText that contains the rules that are being refined
    """

    def __init__(self, gui : GUI, variable : Variable, skope_list: list = None) :
        self._gui = gui
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
        # We core the click event on validateSkopeChangeBtn (010020)
        widget_at_address(self._widget, "010020").on_event("click", skope_slider_changed)


        def skope_slider_changed(*change):
            # We retrive the skopeSlider (0010001) min value
            self._gui.get_selection().getVSRules()[2][0] = float(widget_at_address(self._widget,"0010001").v_model[0])
            # We retrive the skopeSlider (0010001) max value
            self._gui.get_selection().getVSRules()[2][4] = float(widget_at_address(self._widget,"0010001").v_model[1])
            # We redefined ourVSSkopeCard (30500101)
            widget_at_address(self._gui.get_app_graph(), "30500101").children = create_rule_card()

            self._gui.update_graph_with_rules()
            self._gui.update_submodels_scores(None)

        

        def get_widget(self) -> v.ExpansionPanels :
            return self._widget
        
        
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
                    self._gui.get_dataset().getFullValues()[self._gui.get_selection().getVSRules()[0][2]].values
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



class AntakiaExplorer :
    """
        An AntakiaExplorer displays a 2D or 3D projection of the dataset, using a scatter plot viesw.
        An AntakiaExplorer is a ipywidget with a menu to change the dimension reduction ethod.
        It relies on the Plotly JS library to display the scatter plot.

    
    _figure : FigureWidget
    _projectionSelect : v.Select
    _projectionSliders : v.Dialog
    _isExplainExplorer : bool
    _explanationSelect : v.Select # if _isExplainExplorer is True
    _explainComputeMenu : v.Menu # if _isExplainExplorer is True
    _ds : Dataset # even if _explainExplorer we need _ds for the Y values
    _xds : ExplanationDataset # if _isExplainExplorer is True
    _currentDim : int # May be 2 or 3
    _currentProjection : int # refers to DimReduction constants
    _currentExplanation : int # refers to Explanation constants. Only relevant if _isExplainExplorer is True
    _currentOrigin : int # refers to ExplanationDataset constants. Only relevant if _isExplainExplorer is True

    _sideStr : str
    """

    def __init__(self, ds : Dataset, xds : ExplanationDataset, isExplainExplorer : bool):
        """
        Instantiate a new AntakiaExplorer.
        """ 
        self._isExplainExplorer = isExplainExplorer
        if self._isExplainExplorer :
            self._xds = xds
            self._explanationSelect = v.Select(
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
            self._explainComputeMenu = v.Menu(
                    v_slots=[
                                {
                                    "name": "activator",
                                    "variable": "props",
                                    "children": v.Btn(
                                        v_on="props.on",
                                        icon=True,
                                        size="x-large",
                                        # children=[wrap_in_a_stooltip(v.Icon(children=["mdi-timer-sand"], size="large"), "Time of computing")],
                                        # children=v.Icon(children=["mdi-timer-sand"], size="large"),
                                        class_="ma-2 pa-3",
                                        elevation="3",
                                    ),
                                }
                        ],
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
                                        v.WindowItem(value=0, children=[
                                            v.Col( #This is Tab "one" content
                                                class_="d-flex flex-column align-center",
                                                children=[
                                                        v.Html(
                                                            tag="h3",
                                                            class_="mb-3",
                                                            children=["Compute SHAP values"],
                                                    ),
                                                    v.ProgressLinear(
                                                        style_="width: 80%",
                                                        v_model=0,
                                                        color="primary",
                                                        height="15",
                                                        striped=True,
                                                    ),
                                                    v.TextField(
                                                        class_="w-100",
                                                        style_="width: 100%",
                                                        v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
                                                        readonly=True,
                                                    ),
                                                    v.Btn(
                                                        children=[v.Icon(class_="mr-2", children=["mdi-calculator-variant"]), "Compute values"], #SHAP compute button
                                                        class_="ma-2 ml-6 pa-3",
                                                        elevation="3",
                                                        v_model="lion",
                                                        color="primary",
                                                    ),
                                                ],
                                            )
                                        ]),
                                        v.WindowItem(value=1, children=[
                                            v.Col( #This is Tab "two" content
                                                class_="d-flex flex-column align-center",
                                                children=[
                                                        v.Html(
                                                            tag="h3",
                                                            class_="mb-3",
                                                            children=["Compute LIME values"],
                                                    ),
                                                    v.ProgressLinear( # LIME progress bar we'll have to update
                                                        style_="width: 80%",
                                                        v_model=0,
                                                        color="primary",
                                                        height="15",
                                                        striped=True,
                                                    ),
                                                    v.TextField(
                                                        class_="w-100",
                                                        style_="width: 100%",
                                                        v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
                                                        readonly=True,
                                                    ),
                                                    v.Btn(
                                                        children=[v.Icon(class_="mr-2", children=["mdi-calculator-variant"]), "Compute values"], # LIME compute button
                                                        class_="ma-2 ml-6 pa-3",
                                                        elevation="3",
                                                        v_model="panthère",
                                                        color="primary",
                                                    ),
                                                ],
                                            )
                                            ]),
                                    ],
                                )
                            ],
                        ),
                        ],
                        v_model=False,
                        close_on_content_click=False,
                        offset_y=True,
                        )
            self._currentExplanation = ExplanationMethod.SHAP # Shall we set _explanationSelect ?
            self._currentOrigin = ExplanationDataset.IMPORTED
            self._sideStr="ES"
        else :
            self._sideStr="VS"
        self._ds = ds
        self._projectionSelect = v.Select(
            label="Projection in the :"+self._sideStr,
            items=DimReducMethod.getDimReducMethodsAsStrList(),
            style_="width: 150px",
        )
        self._projectionSliders = widgets.VBox([
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

        self._currentProjection = None

        self._markers = dict( 
            color=self._ds.getYValues(Dataset.REGULAR),
            colorscale="Viridis",
            colorbar=dict(
                title="y",
                thickness=20,
            ),
        )

        #  ourESMarkerDict = dict(
        #     color=self._ds.getYValues(Dataset.REGULAR), 
        #     colorscale="Viridis")

    #   ourVS3DMarkerDict = dict(color=self._ds.getYValues(Dataset.REGULAR), colorscale="Viridis", colorbar=dict(thickness=20,), size=3,)
    #     ourES3DMarkerDict = dict(color=self._ds.getYValues(Dataset.REGULAR), colorscale="Viridis", size=3)



        self._figure = FigureWidget(
            data=Scatter(
                x=self._getX(), 
                y=self._getY(), 
                mode="markers", 
                marker=self._markers, 
                customdata=self._markers["color"], 
                hovertemplate = '%{customdata:.3f}')
        )
        self._currentDim = 2

    # ---- getter & setters ------

    def isExplainExplorer(self) -> bool :
        return self._isExplainExplorer
    
    def getProjectionSelect(self) -> v.Select :
        return self._projectionSelect   

    def getProjectionSliders(self) -> widgets.VBox :
        return self._projectionSliders 

    def getExplanationSelect(self) -> v.Select :
        return self._explanationSelect
    
    def getExplainComputeMenu(self) -> v.Menu :
        return self._explainComputeMenu

    def getFigureWidget(self)-> widgets.Widget :
        return self._figure

    def setDimension(self, dim : int) : 
        self._currentDim = dim

    
    # ----------------  
    
    def _getX(self) -> pd.DataFrame :
        if self._isExplainExplorer :
            return self._xds.getFullValues(self._currentExplanation, self._currentOrigin)
        else :
            return self._ds.getFullValues()

    def _getY(self) -> pd.Series :
        return self._ds.getYValues()


def widget_at_address(graph : Widget, address : str) :
    """ Returns the widget at the given address in the graph.
        Ex: widget_at_address(graph, "012") returns graph.children[0].children[1].children[2]
        Implemented recursively.
    """
    if len(address) > 1 :
        return widget_at_address(graph.children[int(address[0])])
    else :
        return graph.children[int(address[0])]
        



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
        v.AppBar( # gui.children[0]
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
                v.Html(tag="h2", children=["AntakIA"], class_="ml-3"), # gui.children[0].children[1]
                v.Spacer(),
                v.Btn( # backupBtn #gui.children[0].children[3]
                    icon=True, children=[v.Icon(children=["mdi-content-save"])], elevation=0
                ),
                v.Btn( # settingsBtn #gui.children[0].children[4]
                    icon=True, children=[v.Icon(children=["mdi-tune"])], elevation=0
                ),
                v.Dialog(
                    children=[
                        v.Card(
                            children=[
                                v.CardTitle(
                                    children=[
                                        v.Icon(class_="mr-5", children=["mdi-cogs"]),
                                        "Settings",
                                    ]
                                ),
                                v.CardText(children=[
                                    v.Row(children=[
                                        v.Slider(
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
                v.Btn( # gotoWebBtn
                    icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
                ),
            ],
        ), 
        widgets.HBox( # gui.children[1]
            [
                v.Row( # gui.children[1].children[0]
                    class_="ma-3",
                    children=[
                        v.Icon(children=["mdi-numeric-2-box"]),
                        v.Icon(children=["mdi-alpha-d-box"]),
                        v.Switch( # Dim switch # gui.children[1].children[0].children[2]
                            class_="ml-3 mr-2",
                            v_model=False,
                            label="",
                        ),
                        v.Icon(children=["mdi-numeric-3-box"]),
                        v.Icon(children=["mdi-alpha-d-box"]),
                    ],
                ),
                v.Layout( # gui.children[1].children[1]
                    class_="pa-2 ma-2",
                    elevation="3",
                        children=[
                                v.Icon(
                                    children=["mdi-format-color-fill"], class_="mt-n5 mr-4"
                                ),
                                v.BtnToggle( # colorChoiceBtnToggle
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
                                ),
                                v.Btn( # opacityBtn
                                    icon=True,
                                    children=[v.Icon(children=["mdi-opacity"])],
                                    class_="ma-2 ml-6 pa-3",
                                    elevation="3",
                                ),
                                v.Select( # explanationSelect
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
                v.Layout( # gui.children[1].children[2]
                    class_="mt-3",
                    children=[
                        widgets.HBox(
                            [
                                v.Select( # projSelectVS
                                    label="Projection in the VS :",
                                    items=DimReducMethod.getDimReducMethodsAsStrList(),
                                    style_="width: 150px",
                                ),
                                v.Layout(children=[
                                    v.Menu( # projSettingsMenuVS
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
                                    ]),
                                widgets.HBox([ 
                                    v.ProgressCircular( # VSBusyBox
                                        indeterminate=True, color="blue", width="6", size="35", class_="mx-4 my-3"
                                    )
                                    ]),
                            ]
                        ),
                        widgets.HBox(
                            [
                                v.Select( # projSelectES
                                    label="Projection in the ES :",
                                    items=DimReducMethod.getDimReducMethodsAsStrList(),
                                    style_="width: 150px",
                                ),
                                v.Layout(children=[
                                    v.Menu( # projSettingsMenuES
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
                                                children=[
                                                    widgets.VBox([ # ProjES sliders
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
                                    ]),
                                widgets.HBox([
                                    v.ProgressCircular( # ESBusyBox
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
        widgets.VBox([
            widgets.HBox([
                widgets.VBox(
                        [
                        widgets.HTML("<h3>Values Space<h3>"),
                        FigureWidget(
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
                widgets.VBox(
                        [
                        widgets.HTML("<h3>Explanations Space<h3>"),
                        FigureWidget(
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
        v.Container(
            fluid = True,
            children=[
                v.Tabs(
                                    v_model=0, # default active tab
                                    children=
                                    [
                                        v.Tab(children=["1. Selection"]), 
                                        v.Tab(children=["2. Refinement"]), 
                                        v.Tab(children=["3. Sub-model"]), 
                                        v.Tab(children=["4. Regions"])
                                    ] 
                                    + 
                                    [
                                        v.TabItem(  # Tab 1) Selection
                                            children=[
                                                v.Card( # selectionCard
                                                    class_="ma-2",
                                                    elevation=0,
                                                    children=[
                                                        v.Layout(
                                                            children=[
                                                                v.Icon(children=["mdi-lasso"]),
                                                                v.Html(
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
                                                v.ExpansionPanels( # out_accordion
                                                    class_="ma-2",
                                                    children=[
                                                        v.ExpansionPanel(
                                                            children=[
                                                                v.ExpansionPanelHeader(children=["Data selected"]),
                                                                v.ExpansionPanelContent(
                                                                    children=[
                                                                    v.Alert(
                                                                        max_height="400px",
                                                                        style_="overflow: auto",
                                                                        elevation="0",
                                                                        children=[
                                                                            v.Row(
                                                                                class_="d-flex flex-row justify-space-between",
                                                                                children=[
                                                                                    v.Layout(
                                                                                        style_="min-width: 47%; max-width: 47%",
                                                                                        children=[
                                                                                            v.Html(
                                                                                                tag="h4",
                                                                                                children=["Select points on the figure to see their values ​​here"],
                                                                                            )
                                                                                        ],
                                                                                    ),
                                                                                    v.Divider(class_="ma-2", vertical=True),
                                                                                    v.Layout(
                                                                                        style_="min-width: 47%; max-width: 47%",
                                                                                        children=[
                                                                                            v.Html(
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
                                                v.Layout( # clusterGrp
                                                    class_="d-flex flex-row",
                                                    children=[
                                                        v.Btn(
                                                            class_="ma-1 mt-2 mb-0",
                                                            elevation="2",
                                                            children=[v.Icon(children=["mdi-magnify"]), "Find clusters"],
                                                        ),
                                                        v.Checkbox(
                                                            v_model=True, label="Optimal number of clusters :", class_="ma-3"
                                                        ),
                                                        v.Slider(
                                                            style_="width : 30%",
                                                            class_="ma-3 mb-0",
                                                            min=2,
                                                            max=20,
                                                            step=1,
                                                            v_model=3,
                                                            disabled=True,
                                                        ),
                                                        v.Html(
                                                            tag="h3",
                                                            class_="ma-3 mb-0",
                                                            children=["Number of clusters #"],
                                                        ),
                                                    ],
                                                ),
                                                v.ProgressLinear( # loadingClustersProgLinear
                                                    indeterminate=True, class_="ma-3", style_="width : 100%"
                                                ),
                                                v.Row( # clusterResults
                                                    children=[
                                                        v.Layout(
                                                            class_="flex-grow-0 flex-shrink-0", children=[
                                                                v.RadioGroup(
                                                                    v_model=None,
                                                                    class_="mt-10 ml-7",
                                                                    style_="width : 10%",
                                                                    children=[
                                                                        "1", "2", "3"
                                                                    ],
                                                                ),
                                                                ]
                                                        ),
                                                        v.Layout(
                                                            class_="flex-grow-0 flex-shrink-0", children=[
                                                                v.Col(
                                                                    class_="mt-10 mb-2 ml-0 d-flex flex-column justify-space-between",
                                                                    style_="width : 10%",
                                                                    children=[
                                                                        "4", "5", "6"
                                                                    ],
                                                                )
                                                                ]
                                                        ),
                                                        v.Layout(
                                                            class_="flex-grow-1 flex-shrink-0",
                                                            children=[
                                                                v.DataTable(
                                                                    class_="w-100",
                                                                    style_="width : 100%",
                                                                    v_model=[],
                                                                    show_select=False,
                                                                    # headers=columns,
                                                                    # explanationsMenuDict=new_df.to_dict("records"),
                                                                    item_value="Region #",
                                                                    item_key="Region #",
                                                                    hide_default_footer=True,
                                                                ),
                                                                ],
                                                        ),
                                                    ],
                                                ),
                                                v.Layout(
                                                    class_="d-flex flex-row justify-center align-center",
                                                    children=[
                                                        v.Spacer(),
                                                            v.Btn(
                                                                class_="ma-3",
                                                                children=[
                                                                    v.Icon(children=["mdi-creation"], class_="mr-3"),
                                                                    "Magic button",
                                                                ],
                                                        ),
                                                        v.Checkbox(v_model=True, label="Demonstration mode", class_="ma-4"),
                                                        v.TextField(
                                                            class_="shrink",
                                                            type="number",
                                                            label="Time between the steps (ds)",
                                                            v_model=10,
                                                        ),
                                                        v.Spacer(),
                                                    ],
                                                )
                                            ]
                                        ), 
                                        v.TabItem( # Tab 2) Refinement
                                            children=[
                                                v.Col(
                                                    children=[
                                                        widgets.VBox( # skopeBtnsGrp
                                                            [
                                                            v.Layout( # skopeBtns
                                                                class_="d-flex flex-row",
                                                                children=[
                                                                    v.Btn( # validateSkopeBtn
                                                                        class_="ma-1",
                                                                        children=[
                                                                            v.Icon(class_="mr-2", children=["mdi-auto-fix"]),
                                                                            "Skope-Rules",
                                                                        ],
                                                                    ),
                                                                    v.Btn( # reinitSkopeBtn
                                                                        class_="ma-1",
                                                                        children=[
                                                                            v.Icon(class_="mr-2", children=["mdi-skip-backward"]),
                                                                            "Come back to the initial rules",
                                                                        ],
                                                                    ),
                                                                    v.Spacer(),
                                                                    v.Checkbox( # beeSwarmCheck
                                                                        v_model=True,
                                                                        label="Show Shapley's beeswarm plots",
                                                                        class_="ma-1 mr-3",
                                                                    )
                                                                    ,
                                                                ],
                                                            ),
                                                            v.Layout( # skopeText
                                                                class_="d-flex flex-row",
                                                                children=[
                                                                    v.Card( # ourVSSkopeText
                                                                        style_="width: 50%;",
                                                                        class_="ma-3",
                                                                        children=[
                                                                            v.Row(
                                                                                class_="ml-4",
                                                                                children=[
                                                                                    v.Icon(children=["mdi-target"]),
                                                                                    v.CardTitle(children=["Rules applied to the Values Space"]),
                                                                                    v.Spacer(),
                                                                                    v.Html(
                                                                                        class_="mr-5 mt-5 font-italic",
                                                                                        tag="p",
                                                                                        children=["precision = /"],
                                                                                    ),
                                                                                ],
                                                                            ),
                                                                            v.Card( # ourVSCard
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
                                                                    v.Card( # ourESSkopeText
                                                                        style_="width: 50%;",
                                                                        class_="ma-3",
                                                                        children=[
                                                                            v.Row(
                                                                                class_="ml-4",
                                                                                children=[
                                                                                    v.Icon(children=["mdi-target"]),
                                                                                    v.CardTitle(children=["Rules applied on the Explanatory Space"]),
                                                                                    v.Spacer(),
                                                                                    v.Html(
                                                                                        class_="mr-5 mt-5 font-italic",
                                                                                        tag="p",
                                                                                        children=["precision = /"],
                                                                                    ),
                                                                                ],
                                                                            ),
                                                                            v.Card( # ourESCard
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
                                                        ),
                                                        widgets.VBox( # skopeAccordion
                                                            children=[
                                                                v.ExpansionPanels( # accordionGrp1
                                                                    class_="ma-2 mb-1",
                                                                    children=[
                                                                        v.ExpansionPanel(
                                                                            disabled = False,
                                                                            children=[
                                                                                v.ExpansionPanelHeader(
                                                                                    children=
                                                                                    ["X1"]
                                                                                    ),
                                                                                v.ExpansionPanelContent(
                                                                                    children=[
                                                                                        widgets.HBox( # accordion1
                                                                                            [ 
                                                                                                widgets.VBox( # histo1Ctrl
                                                                                                    [   
                                                                                                        v.Layout( # skopeSliderGroup1
                                                                                                            children=[
                                                                                                                v.TextField(
                                                                                                                    style_="max-width:100px",
                                                                                                                    v_model=1, # min value of the slider
                                                                                                                    hide_details=True,
                                                                                                                    type="number",
                                                                                                                    density="compact",
                                                                                                                ),
                                                                                                                v.RangeSlider( # skopeSlider
                                                                                                                    class_="ma-3",
                                                                                                                    v_model=[-1, 1],
                                                                                                                    min=-10e10,
                                                                                                                    max=10e10,
                                                                                                                    step=0.01,
                                                                                                                )
                                                                                                                
                                                                                                                ,
                                                                                                                v.TextField(
                                                                                                                    style_="max-width:100px",
                                                                                                                    v_model=5, # max value of the slider
                                                                                                                    hide_details=True,
                                                                                                                    type="number",
                                                                                                                    density="compact",
                                                                                                                    step="0.1",
                                                                                                                ),
                                                                                                            ],
                                                                                                        ),
                                                                                                        FigureWidget( # histogram1
                                                                                                            data=[
                                                                                                                Histogram(
                                                                                                                    x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')),
                                                                                                                    bingroup=1, 
                                                                                                                    nbinsx=50, 
                                                                                                                    marker_color="grey"
                                                                                                                    )
                                                                                                            ]
                                                                                                        ),
                                                                                                        widgets.HBox( # validateSkopeChangeBtnAndCheck1
                                                                                                            [
                                                                                                                v.Btn(
                                                                                                                    class_="ma-3",
                                                                                                                    children=[
                                                                                                                        v.Icon(class_="mr-2", children=["mdi-check"]),
                                                                                                                        "Validate the changes",
                                                                                                                    ],
                                                                                                                ), 
                                                                                                                v.Checkbox(
                                                                                                                        v_model=False, label="Real-time updates on the figures", class_="ma-3"
                                                                                                                    )
                                                                                                                ]
                                                                                                        )
                                                                                                        ]
                                                                                                ),
                                                                                                widgets.VBox( # beeswarmGrp1
                                                                                                    [
                                                                                                        v.Row( # bs1ColorChoice
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
                                                                                                        ),
                                                                                                        FigureWidget( # beeswarm1
                                                                                                            data=[Scatter(
                                                                                                                x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')), 
                                                                                                                y=[0] * 100, 
                                                                                                                mode="markers")]
                                                                                                        )
                                                                                                    ],
                                                                                                    layout=Layout(margin = "0px 0px 0px 20px")
                                                                                                    ), 
                                                                                                v.Col( # rightSide1
                                                                                                    children=[
                                                                                                        v.Btn( # deleteSkopeBtn1
                                                                                                            class_="ma-2 ml-4 pa-1",
                                                                                                            elevation="3",
                                                                                                            icon=True,
                                                                                                            children=[v.Icon(children=["mdi-delete"])],
                                                                                                            disabled=True,
                                                                                                        ),
                                                                                                        v.Checkbox( # isContinuousChck1
                                                                                                            v_model=True, 
                                                                                                            label="is continuous?"
                                                                                                            )
                                                                                                        ],
                                                                                                    class_="d-flex flex-column align-center justify-center",
                                                                                                )
                                                                                                ],
                                                                                            layout=Layout(align_explanationsMenuDict="center"),
                                                                                        )
                                                                                    ]
                                                                                    ),
                                                                            ]
                                                                        ),
                                                                        v.ExpansionPanels( # accordionGrp2
                                                                            class_="ma-2 mb-1",
                                                                            children=[
                                                                                v.ExpansionPanel(
                                                                                    disabled = True,
                                                                                    children=[
                                                                                        v.ExpansionPanelHeader(children=["X2"]),
                                                                                        # v.ExpansionPanelContent(children=[accordion2]),
                                                                                    ]
                                                                                )
                                                                            ],
                                                                        ),
                                                                        v.ExpansionPanels( # accordionGrp3
                                                                            class_="ma-2 mb-1",
                                                                            children=[
                                                                                v.ExpansionPanel(
                                                                                    disabled = True,
                                                                                    children=[
                                                                                        v.ExpansionPanelHeader(children=["X3"]),
                                                                                        # v.ExpansionPanelContent(children=[accordion3]),
                                                                                    ]
                                                                                )
                                                                            ],
                                                                        ),
                                                                        ],
                                                                    layout=Layout(width="100%", height="auto"),
                                                                    ),
                                                            ]
                                                        ),
                                                        v.Row( #addButtonsGrp
                                                            children=[
                                                                v.Btn(
                                                                    class_="ma-4 pa-2 mb-1",
                                                                    children=[v.Icon(children=["mdi-plus"]), "Add a rule"],
                                                                ), 
                                                                # addAnotherFeatureWgt
                                                                v.Select(
                                                                    class_="mr-3 mb-0",
                                                                    explanationsMenuDict=["/"],
                                                                    v_model="/",
                                                                    style_="max-width : 15%",
                                                                ), v.Spacer(), 
                                                                v.Btn(
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
                                        v.TabItem(
                                            children=[
                                                    widgets.VBox(
                                                        [
                                                            v.ProgressLinear( # loadingModelsProgLinear
                                                                indeterminate=True,
                                                                class_="my-0 mx-15",
                                                                style_="width: 100%;",
                                                                color="primary",
                                                                height="5",
                                                            ), 
                                                            v.SlideGroup( 
                                                                v_model=None,
                                                                class_="ma-3 pa-3",
                                                                elevation=4,
                                                                center_active=True,
                                                                show_arrows=True,
                                                                children=
                                                                [
                                                                    v.SlideItem( # dummy SlideItem. Will be replaced by the app
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
                                        v.TabItem(
                                            children=[
                                                v.Col(
                                                children=[
                                                    widgets.VBox(
                                                        [

                                                            ]
                                                    ),         
                                                ]
                                                )
                                            ]
                                        )
                                    ]
                                )
            ],
            # class_="mt-0",
            outlined=True
        )
    ]
)