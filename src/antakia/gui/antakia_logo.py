import ipyvuetify as v
from ipywidgets import widgets
import webbrowser
from importlib.resources import files


class AntakiaLogo:
    def __init__(self):
        self.build_widget()

    def build_widget(self):
        star_btn = v.Btn(color='primary', children=['Star Antakia'])
        star_btn.on_event('click', self.open_web)

        self.dialog = v.Dialog(
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

        self.dialog.on_event('keydown.stop', lambda *args: None)  # close dialog on escape

        self.widget = v.Layout(
            children=[
                widgets.Image(
                    value=open(
                        files("antakia").joinpath("assets/logo_antakia_horizontal.png"),  # type: ignore
                        "rb",
                    ).read(),
                    height=str(864 / 20) + "px",
                    width=str(3839 / 20) + "px",
                ),
                self.dialog
            ],
        )
        self.widget.on_event('click', self.open_web)

    def open_web(self, *args):
        webbrowser.open('https://github.com/AI-vidence/antakia')

    def open(self):
        self.dialog.v_model = True
