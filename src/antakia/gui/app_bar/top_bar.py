import webbrowser
from importlib.resources import files

import ipyvuetify as v
from ipywidgets import widgets

from antakia.gui.helpers.metadata import metadata
from antakia.gui.theme import theme


class TopBar:
    def __init__(self):
        self._dark_mode = False
        self._build_widget()
        # Register for theme changes
        theme.add_observer(self._on_theme_change)

    def _build_widget(self):
        star_btn = v.Btn(color="primary", children=["Star Antakia"])
        star_btn.on_event("click", self.open_web)

        self.star_dialog = v.Dialog(
            width="500",
            v_model=False,
            children=[
                v.Card(
                    children=[
                        v.CardTitle(
                            class_="headline gray lighten-2",
                            primary_title=True,
                            children=["Do you like AntakIA ?"],
                        ),
                        v.CardText(
                            children=[
                                v.Html(
                                    tag="p",
                                    children=["Please star us on Github if you like our work!"],
                                ),
                                star_btn,
                            ]
                        ),
                    ]
                )
            ],
        )

        self.star_dialog.on_event("keydown.stop", lambda *args: None)

        self.logo = v.Sheet(
            class_="transparent",
            children=[
                widgets.Image(
                    value=self.get_logo(),
                    height="43px",
                    width="192px",
                ),
                self.star_dialog,
            ],
        )

        # Dark mode toggle button
        self.dark_mode_icon = v.Icon(children=["mdi-weather-night"])
        self.dark_mode_btn = v.Btn(
            icon=True,
            children=[self.dark_mode_icon],
            class_="ma-1",
            elevation="0",
        )
        self.dark_mode_btn.on_event("click", self._toggle_dark_mode)

        # Help button
        self.help_btn = v.Btn(
            icon=True,
            children=[v.Icon(children=["mdi-help-circle-outline"])],
            class_="ma-1",
            elevation="0",
        )

        # Settings menu
        self.settings_menu = v.Menu(
            v_slots=[
                {
                    "name": "activator",
                    "variable": "props",
                    "children": v.Btn(
                        v_on="props.on",
                        icon=True,
                        children=[v.Icon(children=["mdi-cog-outline"])],
                        class_="ma-1",
                        elevation="0",
                    ),
                }
            ],
            children=[
                v.Card(
                    class_="pa-4",
                    rounded=True,
                    children=[
                        v.CardTitle(children=["Settings"]),
                        v.Divider(),
                        v.CardText(
                            children=[
                                v.Html(tag="p", children=["Configuration options coming soon..."]),
                            ]
                        ),
                    ],
                    min_width="300",
                )
            ],
            v_model=False,
            close_on_content_click=False,
            offset_y=True,
        )

        # Version badge
        self.version_chip = v.Chip(
            small=True,
            outlined=True,
            class_="ml-2",
            children=[self.get_version_text()],
        )

        self.widget = v.AppBar(
            class_="white elevation-1",
            dense=True,
            children=[
                self.logo,
                self.version_chip,
                v.Spacer(),
                self.dark_mode_btn,
                self.help_btn,
                self.settings_menu,
            ],
        )

        self.logo.on_event("click", self.open_web)

    def _toggle_dark_mode(self, widget, event, data):
        """Toggle dark mode."""
        is_dark = theme.toggle_dark_mode()
        self._update_dark_mode_icon(is_dark)

    def _update_dark_mode_icon(self, is_dark: bool):
        """Update icon based on dark mode state."""
        icon = "mdi-weather-sunny" if is_dark else "mdi-weather-night"
        self.dark_mode_icon.children = [icon]
        # Update AppBar background
        if is_dark:
            self.widget.class_ = "grey darken-4 elevation-1"
        else:
            self.widget.class_ = "white elevation-1"

    def _on_theme_change(self, theme_instance):
        """Called when theme changes."""
        self._update_dark_mode_icon(theme_instance.dark_mode)

    def open_web(self, *args):
        webbrowser.open("https://github.com/AI-vidence/antakia")

    def open(self):
        self.star_dialog.v_model = True

    def get_logo(self):
        # try:
        #     url = 'https://drive.ai-vidence.com/s/6PQaTGEGD6iXHEp/download/antakia_horizontal.png'
        #     response = requests.get(url)
        #     if response.status_code < 300:
        #         return response.content
        # except:
        #     pass
        file = files("antakia").joinpath("assets/logo_antakia_horizontal.png")  # type: ignore
        return open(file, "rb").read()

    def get_version_text(self):
        # Override to show RC version for local development
        display_version = "6.0 RC7"

        if metadata.is_latest_version():
            return f"v{display_version}"
        else:
            return f"v{display_version} (update available)"
