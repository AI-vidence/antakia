import ipyvuetify as v
from ipywidgets import Layout, widgets
from importlib.resources import files

from antakia.gui.helpers.progress_bar import ProgressBar, MultiStepProgressBar


class SplashScreen:
    def __init__(self, X):
        self.X = X
        self._build_widget()

    def _build_widget(self):
        exp_progressbar_wgt = v.ProgressLinear(  # 110
            style_="width: 80%",
            class_="py-0 mx-5",
            v_model=0,
            color="primary",
            height="15",
            striped=True,
        )
        self.exp_progressbar = ProgressBar(
            exp_progressbar_wgt,
            unactive_color="light blue",
            reset_at_end=False
        )

        proj_progressbar_wgt = v.ProgressLinear(  # 110
            style_="width: 80%",
            class_="py-0 mx-5",
            v_model=0,
            color="primary",
            height="15",
            striped=True,
        )
        self.proj_progressbar = MultiStepProgressBar(
            proj_progressbar_wgt,
            steps=2,
            unactive_color="light blue",
            reset_at_end=False
        )

        self.exp_txt = v.TextField(  # 120
            variant="plain",
            v_model="",
            readonly=True,
            class_="mt-0 pt-0",
        )
        self.proj_txt = v.TextField(  # 120
            variant="plain",
            v_model="",
            readonly=True,
            class_="mt-0 pt-0",
        )

        self.widget = v.Layout(
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
                                exp_progressbar_wgt
                            ],
                        ),
                        v.Col(  # #12
                            children=[
                                self.exp_txt
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
                                proj_progressbar_wgt
                            ],
                        ),
                        v.Col(  # 22
                            children=[
                                self.proj_txt
                            ]
                        ),
                    ],
                ),
            ],
        )

    def set_exp_msg(self, msg):
        self.exp_txt.v_model = msg

    def set_proj_msg(self, msg):
        self.proj_txt.v_model = msg
