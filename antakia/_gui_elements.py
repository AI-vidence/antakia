"""
Gui elements for antakia
"""

import ipywidgets as widgets
from ipywidgets import Layout
import ipyvuetify as v
import pandas as pd
import numpy as np
import webbrowser
import time
from importlib.resources import files
import json

def add_tooltip(widget, text):
    # function that allows you to add a tooltip to a widget
    wid = v.Tooltip(
        bottom=True,
        v_slots=[
            {
                "name": "activator",
                "variable": "tooltip",
                "children": widget,
            }
        ],
        children=[text],
    )
    widget.v_on = "tooltip.on"
    return wid

def ProgressLinear():
    widget = v.ProgressLinear(
            style_="width: 80%",
            class_="py-0 mx-5",
            v_model=0,
            color="primary",
            height="15",
            striped=True,
        )
    return widget

def TotalProgress(texte, element):
    widget = v.Row(
            style_="width:85%;",
            children=[
                v.Col(
                    children=[
                        v.Html(
                            tag="h3",
                            class_="mt-2 text-right",
                            children=[texte],
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

def color_choice():
    couleur_radio = v.BtnToggle(
        color="blue",
        mandatory=True,
        v_model="Y",
        children=[
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-alpha-y-circle-outline"])],
                value="Y",
                v_model=True,
            ),
            v.Btn(
                icon=True,
                children=[v.Icon(children=["mdi-alpha-y-circle"])],
                value="Y^",
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
    couleur_radio.children[0].children = [
        add_tooltip(couleur_radio.children[0].children[0], "Real values")
    ]
    couleur_radio.children[1].children = [
        add_tooltip(couleur_radio.children[1].children[0], "Predicted values")
    ]
    couleur_radio.children[2].children = [
        add_tooltip(couleur_radio.children[2].children[0], "Residuals")
    ]
    couleur_radio.children[3].children = [
        add_tooltip(
            couleur_radio.children[3].children[0],
            "Selected points",
        )
    ]
    couleur_radio.children[4].children = [
        add_tooltip(couleur_radio.children[4].children[0], "Region created")
    ]
    couleur_radio.children[5].children = [
        add_tooltip(
            couleur_radio.children[5].children[0],
            "Points that belong to no region",
        )
    ]
    couleur_radio.children[6].children = [
        add_tooltip(
            couleur_radio.children[6].children[0],
            "Automatic dyadic-clustering result",
        )
    ]
    return couleur_radio

def create_menu_bar():
    """
    Create the menu bar"""
    fig_size = v.Slider(
        style_="width:20%",
        v_model=700,
        min=200,
        max=1200,
        label="Size of the figures (in pixels)",
    )

    fig_size_text = widgets.IntText(
        value="700", disabled=True, layout=Layout(width="40%")
    )

    widgets.jslink((fig_size, "v_model"), (fig_size_text, "value"))

    fig_size_et_texte = v.Row(children=[fig_size, fig_size_text])

    bouton_save = v.Btn(
        icon=True, children=[v.Icon(children=["mdi-content-save"])], elevation=0
    )
    bouton_save.children = [
        add_tooltip(bouton_save.children[0], "Backup management")
    ]
    bouton_settings = v.Btn(
        icon=True, children=[v.Icon(children=["mdi-tune"])], elevation=0
    )
    bouton_settings.children = [
        add_tooltip(bouton_settings.children[0], "Settings of the GUI")
    ]
    bouton_website = v.Btn(
        icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
    )
    bouton_website.children = [
        add_tooltip(bouton_website.children[0], "AI-Vidence's website")
    ]

    bouton_website.on_event(
        "click", lambda *args: webbrowser.open_new_tab("https://ai-vidence.com/")
    )

    data_path = files("antakia.assets").joinpath("logo_ai-vidence.png")
    with open(data_path, "rb") as f:
        logo = f.read()
    logo_aividence1 = widgets.Image(
        value=logo, height=str(864 / 20) + "px", width=str(3839 / 20) + "px"
    )
    logo_aividence1.layout.object_fit = "contain"
    logo_aividence = v.Layout(
        children=[logo_aividence1],
        class_="mt-1",
    )

    # function to open the dialog of the parameters of the size of the figures in particular (before finding better)
    dialogue_size = v.Dialog(
        children=[
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            v.Icon(class_="mr-5", children=["mdi-cogs"]),
                            "Settings",
                        ]
                    ),
                    v.CardText(children=[fig_size_et_texte]),
                        ]
                    ),
                ]
    )
    dialogue_size.v_model = False

    def ouvre_dialogue(*args):
        dialogue_size.v_model = True

    bouton_settings.on_event("click", ouvre_dialogue)

    # menu bar (logo, links etc...)
    barre_menu = v.AppBar(
        elevation="4",
        class_="ma-4",
        rounded=True,
        children=[
            logo_aividence,
            v.Spacer(),
            bouton_save,
            bouton_settings,
            dialogue_size,
            bouton_website,
        ],
    )

    return barre_menu, fig_size, bouton_save


def dialog_save(bouton_save, texte, table_save, save_regions):
    # view a selected backup
    visu_save = v.Btn(
        class_="ma-4",
        children=[v.Icon(children=["mdi-eye"])],
    )
    visu_save.children = [
        add_tooltip(visu_save.children[0], "Visualize the selected backup")
    ]
    delete_save = v.Btn(
        class_="ma-4",
        children=[
            v.Icon(children=["mdi-trash-can"]),
        ],
    )
    delete_save.children = [
        add_tooltip(delete_save.children[0], "Delete the selected backup")
    ]
    new_save = v.Btn(
        class_="ma-4",
        children=[
            v.Icon(children=["mdi-plus"]),
        ],
    )
    new_save.children = [
        add_tooltip(new_save.children[0], "Create a new backup")
    ]

    nom_sauvegarde = v.TextField(
        label="Name of the backup",
        v_model="Default name",
        class_="ma-4",
    )# save backups locally
    bouton_save_all = v.Btn(
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

    # part to save locally: choice of name, location, etc...
    partie_local_save = v.Col(
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
            bouton_save_all,
            out_save,
        ],
    )

    # card to manage backups, which opens
    carte_save = v.Card(
        elevation=0,
        children=[
            v.Html(tag="h3", children=[texte]),
            table_save,
            v.Row(
                children=[
                    v.Spacer(),
                    visu_save,
                    delete_save,
                    v.Spacer(),
                    new_save,
                    nom_sauvegarde,
                    v.Spacer(),
                ]
            ),
            v.Divider(class_="mt mb-5"),
            partie_local_save,
        ],
    )

    dialogue_save = v.Dialog(
        children=[
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            v.Icon(class_="mr-5", children=["mdi-content-save"]),
                            "Backup management",
                        ]
                    ),
                    v.CardText(children=[carte_save]),
                ],
                width="100%",
            )
        ],
        width="50%",
    )
    dialogue_save.v_model = False

    def ouvre_save(*args):
        dialogue_save.v_model = True

    bouton_save.on_event("click", ouvre_save)

    def fonction_save_all(*args):
        emplacement = partie_local_save.children[1].children[1].v_model
        fichier = partie_local_save.children[1].children[2].v_model
        if len(emplacement) == 0 or len(fichier) == 0:
            out_save.color = "error"
            out_save.children = ["Please fill in all fields!"]
            out_save.v_model = True
            time.sleep(3)
            out_save.v_model = False
            return
        destination = emplacement + "/" + fichier + ".json"
        destination = destination.replace("//", "/")
        destination = destination.replace(" ", "_")

        for i in range(len(save_regions)):
            save_regions[i]["liste"] = list(save_regions[i]["liste"])
            save_regions[i]["sub_models"] = save_regions[i][
                "sub_models"
            ].__class__.__name__
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

    bouton_save_all.on_event("click", fonction_save_all)
    
    return [dialogue_save, carte_save, delete_save, nom_sauvegarde, visu_save, new_save]

def figure_and_text(fig, text):
    widget = widgets.VBox(
        [text, fig],
        layout=Layout(
            display="flex", align_items="center", margin="0px 0px 0px 0px"
        ),
    )
    return widget