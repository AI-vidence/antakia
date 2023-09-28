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

from antakia.data import Dataset, DimReducMethod, ExplanationDataset, ExplanationMethod
from antakia.utils import confLogger

logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()


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
def createSkopeCard():
    ourVSCard = v.Card(
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

    ourVSSkopeText = v.Card(
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
            ourVSCard,
        ],
    )

    # Text with skope info on the ES
    ourESCard = v.Card(
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
    )

    ourESSkopeText = v.Card(
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
            ourESCard,
        ],
    )

    # Text with Skope info on the VS and ES
    skopeText = v.Layout(
        class_="d-flex flex-row", children=[ourVSSkopeText, ourESSkopeText]
    )

    return skopeText, ourVSSkopeText, ourESSkopeText, ourVSCard, ourESCard

# ------

# def createFeatureSelector(ds : Dataset, colName, min=1, max=-1, fig_size=700):
#     # TODO understand what this does
#     featureList = list(set(ds.getXValues(Dataset.CURRENT)[colName]))
#     returnList = []
#     for i in range(len(featureList)):
#         if featureList[i] <= max and featureList[i] >= min:
#             le_bool = True
#         else:
#             le_bool = False
#         widget = v.Checkbox(
#             class_="ma-4",
#             v_model=le_bool,
#             label=str(featureList[i]).replace("_", " "),

#         )
#         returnList.append(widget)
#     row = v.Row(class_ = "ml-6 ma-3", children=returnList)
#     text = v.Html(tag="h3", children=["Select the values of the feature " + colName])
#     return v.Layout(class_= "d-flex flex-column align-center justify-center", style_="width: "+str(int(fig_size)-70)+"px; height: 303px", children=[v.Spacer(), text, row])

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


def createValidateChangeBtn():
    widget = v.Btn(
        class_="ma-3",
        children=[
            v.Icon(class_="mr-2", children=["mdi-check"]),
            "Validate the changes",
        ],
    )
    return widget

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
                v_model=skopeSlider.v_model[0],
                hide_details=True,
                type="number",
                density="compact",
            ),
            skopeSlider,
            v.TextField(
                style_="max-width:100px",
                v_model=skopeSlider.v_model[1],
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

def deleteSkopeBtn():
    widget = v.Btn(
        class_="ma-2 ml-4 pa-1",
        elevation="3",
        icon=True,
        children=[v.Icon(children=["mdi-delete"])],
        disabled=True,
    )
    return widget

def skopeAccordion(text, dans_accordion):
    widget = v.ExpansionPanels(
        class_="ma-2 mb-1",
        children=[
            v.ExpansionPanel(
                disabled = True,
                children=[
                    v.ExpansionPanelHeader(children=[text]),
                    v.ExpansionPanelContent(children=[dans_accordion]),
                ]
            )
        ],
    )
    return widget

def createRuleCard(chaine, is_class=False):
    chaine_carac = str(chaine).split()
    taille = int(len(chaine_carac) / 5)
    l = []
    for i in range(taille):
        l.append(
            v.CardText(
                children=[
                    v.Row(
                        class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                        children=[
                            chaine_carac[5 * i],
                            v.Icon(children=["mdi-less-than-or-equal"]),
                            chaine_carac[5 * i + 2],
                            v.Icon(children=["mdi-less-than-or-equal"]),
                            chaine_carac[5 * i + 4],
                        ],
                    )
                ]
            )
        )
        if i != taille - 1:
            if chaine_carac[5 * i + 2] == chaine_carac[5 * (i+1) + 2]:
                l.append(v.Layout(class_="ma-n2 pa-0 d-flex flex-row justify-center align-center", children=[v.Html(class_="ma0 pa-0", tag="i", children=["or"])]))
            else:
                l.append(v.Divider())
    return l

# ----

def createSelectionInfoCard():
    # TODO : to be localized
    initialText = "About the current selection : \n"
    startInitialText = (
        "About the current selection : \n0 point selected (0% of the overall data)"
    )
    # Text that will take the value of text_base + information on the selection
    selectionText = widgets.Textarea(
        value=initialText,
        placeholder="Infos",
        description="",
        disabled=True,
        layout=Layout(width="100%"),
    )

    selectionCard = v.Card(
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
    )

    validateSkopeBtn = v.Btn(
        class_="ma-1",
        children=[
            v.Icon(class_="mr-2", children=["mdi-auto-fix"]),
            "Skope-Rules",
        ],
    )

    # Button that allows you to return to the initial rules in part 2. Skope-rules
    reinitSkopeBtn = v.Btn(
        class_="ma-1",
        children=[
            v.Icon(class_="mr-2", children=["mdi-skip-backward"]),
            "Come back to the initial rules",
        ],
    )
    return initialText, startInitialText, selectionText, selectionCard, validateSkopeBtn, reinitSkopeBtn

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

class AntakiaExplorer :
    """
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
        Instantiate a new AntakiaExplorer. Should not be used directly. Instead, people should use static methods : createVSExplorer and createEXExplorer.
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

    def  setDimension(self, dim : int) : 
        self._currentDim = dim

    
    # ----------------  
    
    def _getX(self) -> pd.DataFrame :
        if self._isExplainExplorer :
            return self._xds.getFullValues(self._currentExplanation, self._currentOrigin)
        else :
            return self._ds.getFullValues()

    def _getY(self) -> pd.Series :
        return self._ds.getYValues()

