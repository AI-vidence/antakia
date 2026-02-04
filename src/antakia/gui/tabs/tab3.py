from typing import Callable, List

import ipyvuetify as v
import pandas as pd
from antakia_core.data_handler import ModelRegion
from antakia_core.utils import BASE_COLOR

from antakia.config import AppConfig
from antakia.gui.graphical_elements.sub_model_table import SubModelTable
from antakia.gui.helpers.data import DataStore
from antakia.gui.helpers.progress_bar import ProgressBar
from antakia.gui.tabs.batch_substitution import BatchSubstitutionWidget
from antakia.gui.tabs.model_explorer import ModelExplorer
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class Tab3:
    headers = [
        {
            "text": column,
            "sortable": True,
            "value": column,
            # "class": "primary white--text",\
        }
        for column in ["Sub-model", "MSE", "MAE", "R2", "delta"]
    ]

    def __init__(
        self, data_store: DataStore, validate_callback: Callable, display_model_data: Callable
    ):
        self.data_store = data_store
        self.validate_callback = validate_callback
        self.display_model_data = display_model_data
        # Pass data_store for current X (survives outlier removal), original model for PDP comparison
        self.model_explorer = ModelExplorer(self.data_store, original_model=self.data_store.model)
        self.region: ModelRegion | None = None
        self.substitution_model_training = False  # tab 3 : training flag

        # Batch substitution widget
        self.batch_widget = BatchSubstitutionWidget(
            data_store=data_store,
            validate_callback=validate_callback,
        )
        self.is_batch_mode = False
        self.is_overview_mode = True  # Start with overview of all regions

        self._build_widget()
        self.progress_bar = ProgressBar(self.progress_wgt, indeterminate=True, reset_at_end=True)
        self.progress_bar(100)

    def _build_widget(self):
        """
        build the tab3 widget - part of init method
        Returns
        -------

        """
        self.validate_model_btn = v.Btn(
            v_on="tooltip.on",
            class_="ma-1 green white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-check"],
                ),
                "Validate sub-model",
            ],
        )
        self.model_table = SubModelTable(
            headers=self.headers,
            items=[],
        )

        # Region selector dropdown
        self.region_select = v.Select(
            v_model=None,
            items=[],
            label="Select region",
            dense=True,
            outlined=True,
            style_="max-width: 250px",
            class_="mr-3",
        )
        self.region_select.on_event("change", self._on_region_select_changed)

        self.region_prefix_wgt = v.Html(class_="mr-2", tag="h3", children=["Region"])
        self.region_chip_wgt = v.Chip(
            color=BASE_COLOR,
            children=["-"],
        )  # 450001
        self.region_title = v.Html(
            class_="ml-2", tag="h3", children=["No region selected for substitution"]
        )  # 450002
        self.progress_wgt = v.ProgressLinear(  # 450110
            style_="width: 100%",
            class_="mt-4",
            v_model=0,
            height="15",
            indeterminate=True,
            color="blue",
        )
        self.training_log_wgt = v.Textarea(
            label="Log des calculs",
            v_model="",
            readonly=True,
            outlined=True,
            dense=True,
            rows=4,
            hide_details=True,
            class_="mt-2",
            style_="font-family: monospace; font-size: 0.85em",
        )
        self.widget = [
            v.Col(
                children=[
                    v.Row(  # Row1 : Title and validate button
                        class_="d-flex",
                        children=[
                            v.Col(  # Col1 - region selector and info
                                class_="col-9",
                                children=[
                                    v.Sheet(
                                        class_="ma-1 d-flex flex-row align-center",
                                        children=[
                                            self.region_select,
                                            self.region_prefix_wgt,
                                            self.region_chip_wgt,
                                            self.region_title,
                                        ],
                                    )
                                ],
                            ),
                            v.Col(  # Col2 - buttons
                                class_="col-3",
                                children=[
                                    v.Row(
                                        class_="flex-column",
                                        children=[
                                            v.Tooltip(
                                                bottom=True,
                                                v_slots=[
                                                    {
                                                        "name": "activator",
                                                        "variable": "tooltip",
                                                        "children": self.validate_model_btn,
                                                    }
                                                ],
                                                children=["Chose this submodel"],
                                            )
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    v.Row(  # Row2 : Progress bar + log
                        class_=" flex-column align-center",
                        children=[
                            v.Col(class_="col-12", children=[self.progress_wgt]),
                            v.Col(class_="col-12", children=[self.training_log_wgt]),
                        ],
                    ),
                    v.Row(  # Row3 : Model table (gauche) + Model explorer PDP (droite)
                        children=[
                            v.Col(class_="col-5", children=[self.model_table]),
                            v.Col(class_="col-7", children=[self.model_explorer.widget]),
                        ],
                    ),
                ]
            )
        ]

        # Store reference to single-region widget content
        self.single_region_content = self.widget[0]
        self.progressbar_widget = self.widget[0].children[1]
        row3 = self.widget[0].children[2]  # Row3 : table (gauche) + PDP (droite)
        self.model_table_widget = row3.children[0]  # Col : tableau des candidats
        self.model_explorer_widget = row3.children[1]  # Col : PDP + Feature Importance

        # Overview panel: all regions + tesselle status
        self.overview_summary_wgt = v.Html(tag="p", class_="ma-2 grey--text", children=[""])
        self.overview_list_container = v.Container(fluid=True, class_="pa-0")
        self.overview_content = v.Container(
            fluid=True,
            children=[
                v.Row(
                    class_="ma-2",
                    children=[
                        v.Html(
                            tag="h4",
                            class_="ma-0",
                            children=["Vue d'ensemble des tesselles"],
                        ),
                    ],
                ),
                v.Row(children=[self.overview_summary_wgt]),
                v.Row(children=[v.Col(children=[self.overview_list_container])]),
            ],
        )

        # Add "Vue d'ensemble" button to detail view header
        self.back_to_overview_btn = v.Btn(
            small=True,
            outlined=True,
            class_="ma-1 mr-2",
            children=[
                v.Icon(small=True, children=["mdi-arrow-left"]),
                "Vue d'ensemble",
            ],
        )
        self.back_to_overview_btn.on_event("click", self._switch_to_overview_mode)
        # Insert back button in Row1 Col1
        self.single_region_content.children[0].children[0].children[0].children[0].children.insert(
            0, self.back_to_overview_btn
        )

        # Create main container that can switch between overview, single and batch mode
        self.main_container = v.Container(fluid=True, children=[self.overview_content])
        self.widget = [self.main_container]

        self.model_table_widget.hide()  # masque le tableau des candidats jusqu'à entraînement

        # We wire a select event on the 'substitution table' :
        self.model_table.set_callback(self._sub_model_selected_callback)

        # We wire a click event on the "validate sub-model" button :
        self.validate_model_btn.on_event("click", self._validate_sub_model)
        self.update()

    @property
    def selected_sub_model(self):
        return self.model_table.selected

    @selected_sub_model.setter
    def selected_sub_model(self, value):
        self.model_table.selected = value

    def update_region(self, region: ModelRegion, train=True):
        """
        method to update the region of substitution
        Parameters
        ----------
        region: region to substitute
        train: if True, train models if not already trained

        Returns
        -------

        """
        # Switch to single-region mode if in batch or overview mode
        if self.is_batch_mode:
            self._switch_to_single_mode()
        elif self.is_overview_mode:
            self.is_overview_mode = False
            self.main_container.children = [self.single_region_content]

        self.region = region

        # Update region selector to reflect current region
        self._update_region_selector()

        if self.region is not None:
            # Check if models are already trained
            models_trained = (
                hasattr(self.region, "perfs")
                and self.region.perfs is not None
                and len(self.region.perfs) > 0
            )

            if models_trained:
                # Models already trained, just update UI
                self.progressbar_widget.hide()
                self.model_table_widget.show()
                self.substitution_model_training = False
                self.update()
            elif train:
                # Need to train models
                self.substitution_model_training = True
                self.progressbar_widget.show()
                self.model_table_widget.hide()
                self.progress_bar(0)
                self.training_log_wgt.v_model = ""
                self.update()

                def log_callback(msg: str):
                    current = self.training_log_wgt.v_model or ""
                    self.training_log_wgt.v_model = current + msg + "\n"
                    self.update()

                # Train substitution models with log
                self.region.train_substitution_models(
                    task_type=self.data_store.problem_category,
                    progress_callback=log_callback,
                )
                log_callback("Terminé.")
                self.progressbar_widget.hide()
                self.model_table_widget.show()
                self.progress_bar(100)
                self.substitution_model_training = False
                self.update()
            else:
                # No train, no existing models
                self.update()
        else:
            self.update()

    def start_batch_substitution(self, regions: List[ModelRegion]):
        """
        Start batch substitution for multiple regions.

        Trains substitution models for all regions in parallel
        with progressive UI updates.

        Parameters
        ----------
        regions: list of ModelRegion to substitute
        """
        self._switch_to_batch_mode()
        self.batch_widget.start_batch(regions)

    def _switch_to_batch_mode(self):
        """Switch UI to batch substitution mode."""
        self.is_batch_mode = True
        self.is_overview_mode = False
        self.main_container.children = self.batch_widget.widget

    def _switch_to_single_mode(self):
        """Switch UI to single-region (detail) mode."""
        self.is_batch_mode = False
        self.is_overview_mode = False
        self.main_container.children = [self.single_region_content]

    def _switch_to_overview_mode(self, *args):
        """Switch UI to overview mode (all regions + tesselle status)."""
        self.is_batch_mode = False
        self.is_overview_mode = True
        self.region = None
        self.selected_sub_model = []
        self.model_explorer.reset()
        self.display_model_data(None, None)
        self.main_container.children = [self.overview_content]
        self._update_overview()

    def _switch_to_detail_mode(self, region: ModelRegion):
        """Switch from overview to detail view for a specific region."""
        self.is_overview_mode = False
        self.main_container.children = [self.single_region_content]
        self.update_region(region, train=True)
        self.display_model_data(region, None)

    def _get_tesselle_status_color(self, region) -> tuple[str, str]:
        """
        Return (status_label, color) for overview display.
        - Green: tesselle validée
        - Green/orange/red: candidats avec gain possible (delta) ou pas
        - Grey: à traiter
        """
        has_tesselle = getattr(region, "validated", False)
        has_candidats = (
            hasattr(region, "perfs")
            and region.perfs is not None
            and len(region.perfs) > 0
        )
        if has_tesselle:
            return "✓", "green"
        if has_candidats:
            best_delta = float(region.perfs["delta"].min())
            if best_delta < -0.01:
                return f"○ Δ{best_delta:+.2f}", "green"  # Gain possible
            if best_delta > 0.01:
                return f"○ Δ{best_delta:+.2f}", "red"  # Perte
            return f"○ Δ{best_delta:+.2f}", "orange"  # Neutre
        return "—", "grey"

    def _update_overview(self):
        """Update the overview list with all regions and tesselle status."""
        region_set = self.data_store.region_set
        list_items = []
        n_with_tesselle = 0
        n_candidats = 0

        for region in region_set.regions.values():
            has_tesselle = getattr(region, "validated", False)
            has_candidats = (
                hasattr(region, "perfs")
                and region.perfs is not None
                and len(region.perfs) > 0
            )
            if has_tesselle:
                n_with_tesselle += 1
            elif has_candidats:
                n_candidats += 1

            model_str = "-"
            if has_tesselle and hasattr(region, "interpretable_models"):
                sel = region.interpretable_models.selected_model
                if sel:
                    model_str = region.interpretable_models.selected_model_str()

            name = getattr(region, "name", "") or f"Région {region.num}"
            status_label, status_color = self._get_tesselle_status_color(region)
            pts = region.num_points()

            def make_click_handler(reg):
                def handler(*args, _r=reg):
                    stats_logger.log("tesselle_overview_region_click", {"region": _r.num})
                    self._switch_to_detail_mode(_r)

                return handler

            status_chip = v.Chip(
                small=True,
                color=status_color,
                class_="mr-2 white--text" if status_color != "grey" else "mr-2",
                children=[status_label],
            )
            li = v.ListItem(
                class_="mb-1",
                children=[
                    v.ListItemContent(
                        children=[
                            v.ListItemTitle(
                                children=[
                                    status_chip,
                                    f"Région {region.num}: {name} — {pts} pts",
                                ]
                            ),
                            v.ListItemSubtitle(
                                children=[f"Modèle: {model_str}"] if model_str != "-" else []
                            ),
                        ]
                    ),
                    v.ListItemAction(children=[v.Icon(small=True, children=["mdi-chevron-right"])]),
                ],
            )
            li.on_event("click", make_click_handler(region))
            list_items.append(li)

        total = len(list_items)
        if total == 0:
            self.overview_summary_wgt.children = [
                "Aucune région. Créez des régions dans l'onglet Régions."
            ]
            self.overview_list_container.children = []
        else:
            parts = [f"{n_with_tesselle} avec tesselle"]
            if n_candidats:
                parts.append(f"{n_candidats} candidat(s)")
            legend = " (vert=tesselle/gain, orange=neutre, rouge=perte, gris=à traiter)"
            self.overview_summary_wgt.children = [
                f"{', '.join(parts)} / {total} total{legend} — "
                f"Cliquez sur une région pour travailler sur la parcelle"
            ]
            self.overview_list_container.children = [
                v.List(class_="py-0", dense=True, children=list_items)
            ]

    def update(self):
        if self.is_overview_mode:
            self._update_overview()
            return
        self._update_region_selector()
        self._update_substitution_prefix()
        self._update_substitution_title()
        self._update_model_table()
        self._update_selected()
        self._update_validate_btn()

    def _update_region_selector(self):
        """Update the region dropdown with available regions."""
        region_set = self.data_store.region_set
        items = []

        # Iterate over region objects (values), not keys
        for region in region_set.regions.values():
            label = f"Region {region.num}"
            if hasattr(region, "name") and region.name:
                label = f"{region.num}: {region.name}"
            items.append(
                {
                    "text": f"{label} ({region.num_points()} pts)",
                    "value": region.num,
                }
            )

        self.region_select.items = items

        # Set current selection
        if self.region is not None:
            self.region_select.v_model = self.region.num
        elif items:
            # Don't auto-select, let user choose
            pass

    @log_errors
    def _on_region_select_changed(self, widget, event, data):
        """Called when user selects a region from dropdown."""
        if data is None:
            return

        with Log("region_select_changed", 2):
            region_num = data
            region = self.data_store.region_set.get(region_num)

            if region is not None and region != self.region:
                stats_logger.log("substitute_region_changed", {"region": region_num})
                # Update region - will train only if not already trained
                self.update_region(region, train=True)
                # Update display
                self.display_model_data(region, None)

    def _update_substitution_prefix(self):
        # Region prefix text
        self.region_prefix_wgt.class_ = "mr-2 black--text" if self.region else "mr-2 grey--text"
        # v.Chip
        self.region_chip_wgt.color = self.region.color if self.region else BASE_COLOR
        self.region_chip_wgt.children = [str(self.region.num)] if self.region else ["-"]

    def _update_model_table(self):
        # Afficher les modèles même si région "trop petite" (avertissement affiché, substitution permise)
        if (
            self.substitution_model_training
            or not self.region
            or len(self.region.perfs) == 0
        ):
            self.model_table.items = []
        else:

            def series_to_str(series: pd.Series) -> pd.Series:
                return series.apply(lambda x: f"{x:.2f}")

            perfs = self.region.perfs
            stats_logger.log("substitute_model", {"best_perf": perfs["delta"].min()})
            for col in perfs.columns:
                if col != "delta_color":
                    perfs[col] = series_to_str(perfs[col])
            perfs = perfs.reset_index().rename(columns={"index": "Sub-model"})
            headers = [
                {
                    "text": column,
                    "sortable": False,
                    "value": column,
                }
                for column in perfs.drop("delta_color", axis=1).columns
            ]
            self.model_table.headers = headers
            self.model_table.items = perfs.to_dict("records")

    def _update_selected(self):
        if self.region and self.region.interpretable_models.selected_model:
            # we set to selected model if any
            self.model_table.selected = [
                {"Sub-model": self.region.interpretable_models.selected_model}
            ]
            self.model_explorer.update_selected_model(self.region.get_selected_model(), self.region)
        else:
            # clear selection if new region:
            self.model_explorer.reset()
            self.model_table.selected = []

    def _update_substitution_title(self):
        title = self.region_title
        title.tag = "h3"
        if self.substitution_model_training:
            # We tell to wait ...
            title.class_ = "ml-2 grey--text italic "
            title.children = ["Sub-models are being evaluated ..."]
            # We clear items int the SubModelTable
        elif not self.region:  # no region provided
            title.class_ = "ml-2 grey--text italic "
            title.children = ["No region selected for substitution"]
        elif self.region.num_points() < AppConfig.ATK_MIN_POINTS_NUMBER:  # region is too small
            title.class_ = "ml-2 orange--text"
            title.children = [
                f"Region too small for substitution ! ({self.region.num_points()} pts) — substitution allowed anyway.",
            ]
        elif len(self.region.perfs) == 0:  # model not trained
            title.class_ = "ml-2 red--text"
            title.children = ["click on substitute button to train substitution models"]
        else:
            # We have results
            title.class_ = "ml-2 black--text"
            title.children = [
                f"{self.region.name}, "
                f"{self.region.num_points()} points, {100 * self.region.dataset_cov():.1f}% of the dataset"
            ]

    def _update_validate_btn(self):
        self.validate_model_btn.disabled = len(self.selected_sub_model) == 0

    def _sub_model_selected_callback(self, data):
        """
        callback on model selection - updates the model explorer
        Parameters
        ----------
        data

        Returns
        -------

        """
        with Log("_sub_model_selected_callback", 2):
            is_selected = bool(data["value"])
            # We use this GUI attribute to store the selected sub-model
            self.selected_sub_model = [data["item"]]
            model_name = data["item"]["Sub-model"]
            self.validate_model_btn.disabled = not is_selected
            if is_selected:
                self.model_explorer.update_selected_model(
                    self.region.get_model(model_name), self.region
                )
                self.display_model_data(self.region, self.region.train_residuals(model_name))
            else:
                self.display_model_data(self.region, None)
                self.model_explorer.reset()

    @log_errors
    def _validate_sub_model(self, *args):
        """
        callback called on model validation
        Parameters
        ----------
        args

        Returns
        -------

        """
        # We get the sub-model data from the SubModelTable:
        # get_widget(self.widget,"45001").items[self.validated_sub_model]
        with Log("_validate_sub_model", 2):
            self.validate_model_btn.disabled = True

            stats_logger.log(
                "validate_sub_model", {"model": self.selected_sub_model[0]["Sub-model"]}
            )

            # We udpate the region
            self.region.select_model(self.selected_sub_model[0]["Sub-model"])
            self.region.validate()
            # empty selected region
            self.region = None
            self.selected_sub_model = []
            # Show tab 2
            self.validate_callback()
            self.progressbar_widget.show()
            self.model_table_widget.hide()  # masque le tableau des candidats (PDP reste visible via model_explorer_widget)
