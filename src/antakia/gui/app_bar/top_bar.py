import ipyvuetify as v
import requests
from ipywidgets import widgets
import webbrowser
from importlib.resources import files
from antakia.gui.helpers.metadata import metadata


class TopBar:
    def __init__(self):
        # We count the number of times this GUI has been initialized

        self._build_widget()

    def _build_widget(self):
        star_btn = v.Btn(color='primary', children=['Star Antakia'])
        star_btn.on_event('click', self.open_web)

        self.star_dialog = v.Dialog(
            width='500',
            v_model=False,
            children=[
                v.Card(children=[
                    v.CardTitle(class_='headline gray lighten-2', primary_title=True, children=[
                        "Do you like AntakIA 😀 ?"
                    ]),
                    v.CardText(children=[
                        v.Html(
                            tag="p",
                            children=["Please star us on Github if you like our work!"],
                        ),
                        star_btn
                    ])
                ])
            ]
        )

        self.star_dialog.on_event('keydown.stop', lambda *args: None)  # close dialog on escape

        self.logo = v.Sheet(
            children=[
                widgets.Image(
                    value=self.get_logo(),
                    height=str(864 / 20) + "px",
                    width=str(3839 / 20) + "px",
                ),
                self.star_dialog
            ],
        )

        self.widget = v.AppBar(  # Top bar # 0
            class_="white",
            children=[
                self.logo,
                v.Sheet(children=self.get_version_text()),
                v.Sheet(class_='flex-fill align-stretch'),  # 02
                v.Sheet(children=[v.Menu(  # 03 # Menu for the figure width
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
                            children=[],
                            min_width="500",
                        )
                    ],
                    v_model=False,
                    close_on_content_click=False,
                    offset_y=True,
                )]),  # End V.Menu
            ],  # End AppBar children
        )  # End AppBar

        self.logo.on_event('click', self.open_web)

    def open_web(self, *args):
        webbrowser.open('https://github.com/AI-vidence/antakia')

    def open(self):
        self.star_dialog.v_model = True

    def get_logo(self):
        try:
            url = 'https://drive.ai-vidence.com/s/6PQaTGEGD6iXHEp/download/antakia_horizontal.png'
            response = requests.get(url)
            if response.status_code < 300:
                return response.content
        except:
            pass
        file = files("antakia").joinpath("assets/logo_antakia_horizontal.png")  # type: ignore
        return open(file, 'rb').read()

    def get_version_text(self):
        current_version = metadata.current_version
        if metadata.is_latest_version():
            return 'v' + current_version
        else:
            return 'v' + current_version + ' - a new version is available !'
