from functools import partial
from typing import Callable

import ipyvuetify as v
import ipywidgets as widgets
import numpy as np
import pandas as pd
from antakia.gui.helpers.skope_multiview import find_descriptive_rules
from antakia_core.data_handler import RegionSet
from sklearn.ensemble import IsolationForest

# AutoCluster is optional (Cython module, may need recompilation)
try:
    from auto_cluster import AutoCluster

    AUTO_CLUSTER_AVAILABLE = True
except ImportError as e:
    AUTO_CLUSTER_AVAILABLE = False
    print(f"[WARNING] auto_cluster not available: {e}")
    print("[WARNING] Auto-clustering feature will be disabled")

from antakia.config import AppConfig
from antakia.gui.graphical_elements.color_table import ColorTable
from antakia.gui.components.color_manager import ColorManager, ALL_PALETTES, color_manager
from antakia.gui.helpers.data import DataStore
from antakia.gui.helpers.progress_bar import ProgressBar
from antakia.gui.high_dim_exp.projected_values_selector import ProjectedValuesSelector
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class Tab2:
    region_headers = [
        {
            "text": column,
            "sortable": False,
            "value": column,
        }
        for column in ["Region", "Rules", "Average", "Points", "% dataset", "Sub-model"]
    ]

    def __init__(
        self,
        data_store: DataStore,
        vs_pvs: ProjectedValuesSelector,
        es_pvs: ProjectedValuesSelector,
        edit_callback: Callable,
        update_callback: Callable,
        substitute_callback: Callable,
    ):
        self.data_store = data_store
        self.vs_pvs = vs_pvs
        self.es_pvs = es_pvs
        self.edit_callback = partial(edit_callback, self)
        self.update_callback = partial(update_callback, self)
        self.substitute_callback = partial(substitute_callback, self)
        self.auto_cluster_running = False

        self._build_widget()

    def _build_widget(self):
        self.region_table_wgt = ColorTable(  # 44001
            headers=self.region_headers,
            items=[],
        )
        self.stats_wgt = v.Html(  # 44002
            tag="p",
            class_="ml-2 mb-2",
            children=["0 region, 0% of the dataset"],
        )
        # Légende : delta (gain/perte) et sur-apprentissage - composants pour affichage correct
        self._legend_wgt = v.Row(
            class_="ml-2 mb-2 align-center flex-wrap",
            style_="font-size: 11px; color: #666;",
            children=[
                v.Html(tag="span", class_="mr-1", children=["Δ :"]),
                v.Html(
                    tag="span",
                    class_="mr-1",
                    style_="background:#a8e6cf;padding:1px 6px;border-radius:3px;",
                    children=[" gain "],
                ),
                v.Html(
                    tag="span",
                    class_="mr-1",
                    style_="background:#ffd3a8;padding:1px 6px;border-radius:3px;",
                    children=[" perte "],
                ),
                v.Html(
                    tag="span",
                    class_="mr-2",
                    style_="background:#fff3a8;padding:1px 6px;border-radius:3px;",
                    children=[" neutre "],
                ),
                v.Html(tag="span", class_="mr-1", children=["| Sur-apprentissage :"]),
                v.Tooltip(
                    bottom=True,
                    v_slots=[
                        {"name": "activator", "variable": "tooltip", "children": v.Html(tag="span", class_="mr-1", style_="cursor:help;", children=["✓ faible"])}
                    ],
                    children=["Risque faible (écart train/test < 5%)"],
                ),
                v.Tooltip(
                    bottom=True,
                    v_slots=[
                        {"name": "activator", "variable": "tooltip", "children": v.Html(tag="span", class_="mr-1", style_="cursor:help;", children=["⚠ modéré"])}
                    ],
                    children=["Risque modéré (écart 5-15%)"],
                ),
                v.Tooltip(
                    bottom=True,
                    v_slots=[
                        {"name": "activator", "variable": "tooltip", "children": v.Html(tag="span", class_="mr-1", style_="cursor:help;", children=["⚠⚠ élevé"])}
                    ],
                    children=["Risque élevé (écart > 15%)"],
                ),
            ],
        )
        # Status widget for showing operation feedback
        self.status_wgt = v.Alert(
            type="info",
            dense=True,
            text=True,
            class_="ml-2 mb-2",
            v_model=False,  # Hidden by default
            children=[""],
        )
        
        # Color palette controls
        self.color_manager = color_manager
        palette_items = [
            {"text": "Modern", "value": "modern"},
            {"text": "Pastel", "value": "pastel"},
            {"text": "Vibrant", "value": "vibrant"},
            {"text": "Categorical", "value": "categorical"},
            {"text": "Earth Tones", "value": "earth"},
        ]
        self.palette_select = v.Select(
            v_model="modern",
            items=palette_items,
            label="Color Palette",
            dense=True,
            hide_details=True,
            class_="ml-2",
            style_="max-width: 150px;",
        )
        self.refresh_colors_btn = v.Btn(
            icon=True,
            small=True,
            class_="ml-1",
            children=[v.Icon(small=True, children=["mdi-refresh"])],
        )
        self.palette_preview = v.Html(
            tag="div",
            class_="d-flex ml-2",
            children=[],
        )
        self.color_controls = v.Row(
            class_="ml-2 mb-2 align-center",
            children=[
                self.palette_select,
                v.Tooltip(
                    bottom=True,
                    v_slots=[{"name": "activator", "variable": "tooltip", "children": self.refresh_colors_btn}],
                    children=["Refresh colors"],
                ),
                self.palette_preview,
            ],
        )
        
        self.substitute_btn = v.Btn(  # 4401000
            v_on="tooltip.on",
            class_="ml-3 mt-8 green white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-swap-horizontal-circle-outline"],
                ),
                "Substitute",
            ],
        )
        self.substitute_all_btn = v.Btn(  # Batch substitute
            v_on="tooltip.on",
            class_="ml-1 mt-8 teal white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-swap-horizontal-bold"],
                ),
                "Substitute All",
            ],
        )
        self.divide_btn = v.Btn(  # 4401100
            v_on="tooltip.on",
            class_="ml-3 mt-8",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-content-cut"],
                ),
                "Divide",
            ],
        )
        self.edit_btn = v.Btn(  # 4401100
            v_on="tooltip.on",
            class_="ml-3 mt-3",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-pencil"],
                ),
                "Edit",
            ],
        )
        self.merge_btn = v.Btn(  # 4401200
            v_on="tooltip.on",
            class_="ml-3 mt-3",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-table-merge-cells"],
                ),
                "Merge",
            ],
        )
        self.delete_btn = v.Btn(  # 4401300
            v_on="tooltip.on",
            class_="ml-3 mt-3",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-delete"],
                ),
                "Delete",
            ],
        )
        self.detect_outliers_btn = v.Btn(  # Outlier detection
            v_on="tooltip.on",
            class_="ml-3 mt-3 orange white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-alert-circle-outline"],
                ),
                "Detect Outliers",
            ],
        )
        self.outlier_method_select = v.Select(
            v_model="iqr",
            items=[
                {"text": "IQR", "value": "iqr"},
                {"text": "Z-Score", "value": "zscore"},
                {"text": "Isolation Forest", "value": "isolation_forest"},
            ],
            dense=True,
            hide_details=True,
            style_="max-width: 180px;",
            class_="mt-0",
        )
        self.auto_cluster_btn = v.Btn(  # 4402000
            v_on="tooltip.on",
            class_="mt-8 primary",
            children=[
                v.Icon(  # 44020000
                    class_="mr-2",
                    children=["mdi-auto-fix"],
                ),
                "Auto-clustering",
            ],
        )
        self.cluster_num_wgt = v.Slider(  # 4402100
            v_on="tooltip.on",
            class_="mt-10 mx-2",
            v_model=6,
            min=2,
            max=12,
            thumb_color="blue",  # marker color
            step=1,
            thumb_label="always",
        )
        self.auto_cluster_checkbox = v.Checkbox(  # 440211
            class_="px-3", v_model=True, label="Automatic number of clusters"
        )
        self.multiview_rules_checkbox = v.Checkbox(
            class_="px-3",
            v_model=self.data_store.multiview_rules_enabled,
            label="Règles descriptives multi-view (RC7)",
        )
        self.multiview_rules_checkbox.on_event("change", self._on_multiview_rules_toggle)
        self.auto_cluster_progress = v.ProgressLinear(  # 440212
            style_="width: 100%",
            class_="px-3",
            v_model=0,
            color="primary",
            height="15",
        )

        # AC confirmation dialog - created at init so it's in the widget tree (required for ipyvuetify)
        self._ac_title_wgt = v.CardTitle(children=[""])
        self._ac_message_wgt = v.CardText(
            children=[""],
            style_="white-space: pre-wrap;",
        )
        self._ac_cancel_btn = v.Btn(children=["Annuler"], text=True, class_="mr-2")
        self._ac_confirm_btn = v.Btn(
            children=["Confirmer"],
            color="primary",
            class_="white--text",
        )
        self._ac_dialog = v.Dialog(
            v_model=False,
            max_width="500px",
            persistent=True,
            content_class="d-flex justify-center align-center",
            transition="dialog-transition",
            children=[
                v.Card(
                    class_="elevation-4",
                    children=[
                        self._ac_title_wgt,
                        self._ac_message_wgt,
                        v.CardActions(
                            children=[
                                v.Spacer(),
                                self._ac_cancel_btn,
                                self._ac_confirm_btn,
                            ]
                        ),
                    ]
                )
            ],
        )

        self.widget = [
            v.Row(  # 440
                class_="d-flex flex-row",
                children=[
                    v.Col(  # v.Sheet Col 1 # 4400
                        class_="col-8",
                        children=[
                            v.Html(  # 44000
                                tag="h3",
                                class_="ml-2",
                                children=["Parcelles :"],
                            ),
                            self.region_table_wgt,
                            self._legend_wgt,
                            self.stats_wgt,
                            self.color_controls,
                            self.status_wgt,
                        ],
                    ),  # End Col 1
                    v.Col(  # v.Sheet Col 2 = buttons #4401
                        class_="col-2",
                        style_="min-width: 280px;",
                        children=[
                            v.Row(  # 44010 - Substitute buttons
                                class_="flex-row",
                                children=[
                                    v.Tooltip(  # 440100
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.substitute_btn,
                                            }
                                        ],
                                        children=[
                                            "Find an explicable surrogate model on this region"
                                        ],
                                    ),
                                    v.Tooltip(  # 440101 - Batch substitute
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.substitute_all_btn,
                                            }
                                        ],
                                        children=["Substitute all selected regions in parallel"],
                                    ),
                                ],
                            ),
                            v.Row(  # 44011
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440110
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.divide_btn,
                                            }
                                        ],
                                        children=["Divide a region into sub-regions"],
                                    )
                                ],
                            ),
                            v.Row(  # 44011
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440110
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.edit_btn,
                                            }
                                        ],
                                        children=["Edit region's rules"],
                                    )
                                ],
                            ),
                            v.Row(  # 44012
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440120
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.merge_btn,
                                            }
                                        ],
                                        children=["Merge regions"],
                                    )
                                ],
                            ),
                            v.Row(  # 44013
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440130
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.delete_btn,
                                            }
                                        ],
                                        children=["Delete region"],
                                    )
                                ],
                            ),
                            v.Divider(class_="my-2"),
                            v.Row(  # Outlier detection row
                                class_="flex-column mt-2",
                                children=[
                                    v.Tooltip(
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.detect_outliers_btn,
                                            }
                                        ],
                                        children=[
                                            "Detect outliers and create an 'Outliers' region"
                                        ],
                                    ),
                                    self.outlier_method_select,
                                ],
                            ),
                        ],  # End v.Sheet Col 2 children
                    ),  # End v.Sheet Col 2 = buttons
                    v.Col(  # v.Sheet Col 3 # 4402
                        class_="col-2 px-6",
                        style_="size: 50%",
                        children=[
                            v.Row(  # 44020
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440200
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.auto_cluster_btn,
                                            }
                                        ],
                                        children=["Find homogeneous regions in both spaces"],
                                    )
                                ],
                            ),
                            v.Row(  # 44021
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440210
                                        bottom=True,
                                        v_slots=[
                                            {
                                                "name": "activator",
                                                "variable": "tooltip",
                                                "children": self.cluster_num_wgt,
                                            }
                                        ],
                                        children=["Number of clusters you expect to find"],
                                    ),
                                    self.auto_cluster_checkbox,
                                    self.multiview_rules_checkbox,
                                    self.auto_cluster_progress,
                                ],
                            ),
                        ],  # End v.Sheet Col 3 children
                    ),  # End v.Sheet Col 3
                ],  # End v.Sheet children
            ),  # End v.Sheet
            self._ac_dialog,  # Must be in tree for ipyvuetify
        ]

        self.wire()
        self.update_region_table()

    def wire(self):
        self.region_table_wgt.set_callback(self.region_selected)
        self.substitute_btn.on_event("click", self.substitute_clicked)
        self.substitute_all_btn.on_event("click", self.substitute_all_clicked)
        # Color controls
        self.palette_select.on_event("change", self._on_palette_change)
        self.refresh_colors_btn.on_event("click", self._on_refresh_colors)
        # Initialize palette preview
        self._update_palette_preview()
        self.edit_btn.on_event("click", self.edit_region_clicked)
        self.divide_btn.on_event("click", self.divide_region_clicked)
        self.merge_btn.on_event("click", self.merge_region_clicked)
        self.detect_outliers_btn.on_event("click", self.detect_outliers_clicked)
        self.delete_btn.on_event("click", self.delete_region_clicked)
        self.auto_cluster_btn.on_event("click", self.auto_cluster_clicked)
        self.auto_cluster_checkbox.v_model = True
        self.auto_cluster_checkbox.on_event("change", self.checkbox_auto_cluster_clicked)
        self.cluster_num_wgt.on_event("change", self.num_cluster_changed)
        self._ac_cancel_btn.on_event("click", self._ac_dialog_cancel)
        self._ac_confirm_btn.on_event("click", self._ac_dialog_confirm)

    @property
    def selected_regions(self):
        return self.region_table_wgt.selected

    @selected_regions.setter
    def selected_regions(self, value):
        self.region_table_wgt.selected = value
        self.update_btns()

    @property
    def region_set(self):
        return self.data_store.region_set

    def update_stats(self):
        region_stats = self.region_set.stats()
        str_stats = [
            f"{region_stats['regions']} {'regions' if region_stats['regions'] > 1 else 'region'}",
            f"{region_stats['points']} points",
            f"{region_stats['coverage']}% of the dataset",
            f"{region_stats['delta_score']:.2f} subst score",
        ]
        self.stats_wgt.children = [", ".join(str_stats)]
    
    def show_status(self, message: str, type_: str = "info"):
        """Show a status message in the interface."""
        self.status_wgt.type = type_
        self.status_wgt.children = [message]
        self.status_wgt.v_model = True
    
    def hide_status(self):
        """Hide the status message."""
        self.status_wgt.v_model = False
    
    # ==================== Color Management ==================== #
    
    def _update_palette_preview(self):
        """Update the palette preview display."""
        palette_name = self.palette_select.v_model
        colors = ALL_PALETTES.get(palette_name, ALL_PALETTES["modern"])[:6]
        
        # Create color swatches
        swatches = []
        for color in colors:
            swatches.append(
                v.Html(
                    tag="div",
                    style_=f"width: 18px; height: 18px; background-color: {color}; border-radius: 3px; margin-right: 3px;",
                )
            )
        self.palette_preview.children = swatches
    
    def _on_palette_change(self, widget, event, data):
        """Handle palette selection change."""
        palette_name = self.palette_select.v_model
        self.color_manager.set_palette(palette_name)
        self._update_palette_preview()
        self._reassign_region_colors()
        self.show_status(f"🎨 Palette changed to '{palette_name}'", "info")
    
    def _on_refresh_colors(self, widget, event, data):
        """Handle refresh colors button click."""
        self.color_manager.reset()
        self._reassign_region_colors()
        self.show_status("🎨 Colors refreshed", "info")
    
    def _reassign_region_colors(self):
        """Reassign colors to all regions using current palette."""
        for region in self.region_set.regions.values():
            new_color = self.color_manager.get_color(region.num)
            region.color = new_color
        self.update_region_table()
        self.update_callback()
    
    def _get_region_color(self, region_num: int, parent_region: int = None) -> str:
        """Get color for a region, optionally based on parent for shading."""
        return self.color_manager.get_color(region_num, parent_region)

    def update_region_table(self):
        """
        Called to empty / fill the RegionDataTable and refresh plots
        """
        self.region_set.sort(by="size", ascending=False)
        temp_items = self.region_set.to_dict()

        # Enrich items with delta info and colors
        for item in temp_items:
            region_num = item.get("Region")
            if region_num == "-":  # left out region
                item["delta"] = None
                item["delta_color"] = None
                item["overfit_risk"] = None
                continue
                
            region = self.region_set.get(region_num)
            if region is None:
                item["delta"] = None
                item["delta_color"] = None
                item["overfit_risk"] = None
                continue
                
            # Get delta and color from perfs if available
            if hasattr(region, 'interpretable_models') and region.interpretable_models.selected_model:
                perfs = region.perfs
                if perfs is not None and len(perfs) > 0 and region.interpretable_models.selected_model in perfs.index:
                    perf = perfs.loc[region.interpretable_models.selected_model]
                    delta = perf.get('delta', 0)
                    item["delta"] = float(delta) if delta is not None else None
                    item["delta_color"] = self._get_delta_color(delta)
                    
                    # Overfitting risk from precomputed overfit_gap
                    overfit_gap = perf.get('overfit_gap', None)
                    if overfit_gap is not None:
                        item["overfit_risk"] = self._get_overfit_indicator(float(overfit_gap))
                    else:
                        item["overfit_risk"] = None
                else:
                    item["delta"] = None
                    item["delta_color"] = None
                    item["overfit_risk"] = None
            else:
                item["delta"] = None
                item["delta_color"] = None
                item["overfit_risk"] = None

        # We populate the ColorTable :
        self.region_table_wgt.items = temp_items

        self.update_stats()
        self.update_btns()
    
    def _get_delta_color(self, delta: float) -> str:
        """
        Get color for delta value (gain/perte en performance).
        
        Delta négatif = gain (modèle meilleur que l'original) → vert
        Delta positif = perte (modèle moins bon) → rouge/orange
        Proche de zéro = neutre → jaune
        """
        if delta is None:
            return "#808080"  # Gris
        
        # Seuils plus sensibles pour distinguer gain/perte
        clamped = max(-0.5, min(0.5, delta))
        
        if clamped < -0.02:
            # Gain : dégradé vert (plus le delta est négatif, plus vert foncé)
            intensity = min(1.0, abs(clamped) * 3)
            r = int(168 - intensity * 122)
            g = int(230 - intensity * 26)
            b = int(207 - intensity * 94)
            return f"#{r:02x}{g:02x}{b:02x}"
        elif clamped > 0.02:
            # Perte : dégradé rouge/orange
            intensity = min(1.0, clamped * 3)
            r = int(255 - intensity * 24)
            g = int(211 - intensity * 135)
            b = int(168 - intensity * 108)
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            # Neutre : jaune pastel
            return "#fff3a8"
    
    def _get_overfit_indicator(self, gap: float) -> str:
        """
        Get overfitting risk indicator.
        
        gap: absolute difference between train and test performance
        """
        if gap < 0.05:
            return "✓"  # Low risk
        elif gap < 0.15:
            return "⚠"  # Medium risk
        else:
            return "⚠⚠"  # High risk

    @log_errors
    def _on_multiview_rules_toggle(self, widget, event, data):
        self.data_store.multiview_rules_enabled = bool(
            self.multiview_rules_checkbox.v_model
        )

    def checkbox_auto_cluster_clicked(self, widget, event, data):
        """
        Called when the user clicks on the 'auto-cluster' checkbox
        """
        self.update_btns()

    @log_errors
    def auto_cluster_clicked(self, *args):
        """
        Called when the user clicks on the 'auto-cluster' button.
        
        Behavior depends on selected regions:
        - No region selected: AC on entire dataset (with confirmation if regions exist)
        - 1 region selected: AC only within that region
        - N regions selected: AC successively in each region (with confirmation)
        """
        with Log("auto_cluster", 2):
            num_selected = len(self.selected_regions)
            num_existing = len(self.region_set)
            
            # Determine behavior based on selection
            if num_selected == 0:
                # No region selected → AC on entire dataset
                if num_existing > 0:
                    # Existing regions will be erased → need confirmation
                    self._show_ac_confirmation_dialog(
                        title="⚠️ Confirmation requise",
                        message=f"L'auto-clustering va s'exécuter sur tout le dataset et effacer les {num_existing} région(s) existante(s).\n\nCette action est irréversible.",
                        on_confirm=lambda: self._execute_auto_cluster_global()
                    )
                    return
                else:
                    # No existing regions, proceed directly
                    self._execute_auto_cluster_global()
            elif num_selected == 1:
                # 1 region selected → AC only within that region
                region = self.region_set.get(self.selected_regions[0]["Region"])
                self._execute_auto_cluster_in_region(region)
            else:
                # Multiple regions selected → AC in each, with confirmation
                self._show_ac_confirmation_dialog(
                    title="⚠️ Confirmation",
                    message=f"L'auto-clustering va s'exécuter successivement dans les {num_selected} régions sélectionnées.\n\nLes autres régions resteront inchangées.",
                    on_confirm=lambda: self._execute_auto_cluster_in_selected_regions()
                )
    
    def _show_ac_confirmation_dialog(self, title: str, message: str, on_confirm: Callable):
        """Show a confirmation dialog before destructive AC operations."""
        self._ac_confirm_callback = on_confirm
        self._ac_title_wgt.children = [title]
        self._ac_message_wgt.children = [message]
        self._ac_dialog.v_model = True
    
    def _ac_dialog_cancel(self, *args):
        """Called when user cancels AC dialog."""
        self._ac_dialog.v_model = False

    def _ac_dialog_confirm(self, *args):
        """Called when user confirms AC dialog."""
        self._ac_dialog.v_model = False
        if hasattr(self, "_ac_confirm_callback") and self._ac_confirm_callback:
            self._ac_confirm_callback()
    
    def _execute_auto_cluster_global(self):
        """Execute auto-clustering on entire dataset (clearing existing regions)."""
        self._start_ac_ui()
        
        # Clear existing regions
        self.region_set.clear_unvalidated()
        
        # AC on all points
        region_set_mask = self.region_set.mask
        not_rules_indexes_list = ~region_set_mask
        cluster_num = self._get_cluster_num()
        
        stats_logger.log(
            "auto_cluster",
            {
                "mode": "global",
                "cluster_num": cluster_num,
                "vs_proj": str(self.vs_pvs.current_proj),
                "es_proj": str(self.es_pvs.current_proj),
            },
        )
        
        self._compute_auto_cluster(
            not_rules_indexes_list, cluster_num,
            target_manually_specified=not self.auto_cluster_checkbox.v_model
        )
        self._finish_ac_ui()
    
    def _execute_auto_cluster_in_region(self, region):
        """Execute auto-clustering only within a single region."""
        self.show_status(f"🔄 Auto-clustering region {region.num} ({region.num_points()} points)...", "info")
        
        self._start_ac_ui()
        
        if region.num_points() < AppConfig.ATK_MIN_POINTS_NUMBER:
            self.show_status(f"⚠️ Region {region.num} too small ({region.num_points()} < {AppConfig.ATK_MIN_POINTS_NUMBER} points)", "warning")
            self._finish_ac_ui()
            return
        
        stats_logger.log(
            "auto_cluster",
            {
                "mode": "in_region",
                "region_num": region.num,
                "cluster_num": self._get_cluster_num(),
            },
        )
        
        # Save parent color for shading
        parent_color = region.color if hasattr(region, 'color') else None
        parent_num = region.num
        
        # Copy the mask before removing the region
        mask = region.mask.copy()
        n_points_in_mask = mask.sum()
        
        # Remove the region (and its color assignment)
        old_num_regions = len(self.region_set)
        self.color_manager.remove_color(region.num)
        self.region_set.remove(region.num)
        
        self.show_status(f"🔄 Clustering {n_points_in_mask} points from region {region.num}...", "info")
        
        # Get cluster num from slider (for subdivision, ensure at least 2)
        cluster_num = self._get_cluster_num(for_subdivision=True)
        
        # Compute auto-cluster
        self._compute_auto_cluster(
            mask, cluster_num, parent_color=parent_color,
            target_manually_specified=not self.auto_cluster_checkbox.v_model
        )
        
        new_regions_count = len(self.region_set) - old_num_regions + 1
        self.show_status(f"✅ Region {region.num} split into {new_regions_count} sub-regions", "success")
        self._finish_ac_ui()
    
    def _execute_auto_cluster_in_selected_regions(self):
        """Execute auto-clustering successively in each selected region."""
        self._start_ac_ui()
        
        # Get all selected regions (copy list since we'll modify region_set)
        regions_to_process = [
            self.region_set.get(r["Region"]) 
            for r in self.selected_regions
        ]
        
        stats_logger.log(
            "auto_cluster",
            {
                "mode": "in_multiple_regions",
                "num_regions": len(regions_to_process),
                "cluster_num": self._get_cluster_num(),
            },
        )
        
        for region in regions_to_process:
            if region.num_points() >= AppConfig.ATK_MIN_POINTS_NUMBER:
                parent_color = region.color if hasattr(region, 'color') else None
                mask = region.mask.copy()
                self.color_manager.remove_color(region.num)
                self.region_set.remove(region.num)
                cluster_num = self._get_cluster_num(for_subdivision=True)
                self._compute_auto_cluster(
                    mask, cluster_num, parent_color=parent_color,
                    target_manually_specified=not self.auto_cluster_checkbox.v_model
                )
        
        self._finish_ac_ui()
    
    def _start_ac_ui(self):
        """Start AC visual feedback."""
        self.auto_cluster_running = True
        self.auto_cluster_btn.disabled = True
        self.auto_cluster_btn.loading = True
        self.auto_cluster_btn.children = [
            v.Icon(class_="mr-2", children=["mdi-loading mdi-spin"]),
            "Computing...",
        ]
        self.auto_cluster_progress.indeterminate = True
        self.auto_cluster_progress.color = "primary"
    
    def _finish_ac_ui(self):
        """Finish AC and restore UI."""
        self.update_region_table()
        self.update_callback()
        
        # Restore button state
        self.auto_cluster_btn.loading = False
        self.auto_cluster_btn.children = [
            v.Icon(class_="mr-2", children=["mdi-auto-fix"]),
            "Auto-clustering",
        ]
        self.auto_cluster_progress.indeterminate = False
        self.auto_cluster_progress.v_model = 0
        self.auto_cluster_progress.color = "grey"
        
        # Clear selection and re-enable
        self.clear_selected_regions()
        self.auto_cluster_running = False
        self.update_btns()

    def num_cluster_changed(self, *args):
        """
        Called when the user changes the number of clusters
        """
        with Log("num_cluster_changed", 2):
            self.update_btns()

    def _get_cluster_num(self, for_subdivision: bool = False):
        """
        Get the target number of clusters.
        
        Parameters
        ----------
        for_subdivision : bool
            If True, always return a fixed number (not "auto") to ensure division happens
        """
        if self.auto_cluster_checkbox.v_model:
            if for_subdivision:
                # For subdivision, don't use "auto" - force at least 2 clusters
                cluster_num = 2
            else:
                cluster_num = "auto"
        else:
            slider_value = self.cluster_num_wgt.v_model
            cluster_num = slider_value - len(self.region_set)
            if cluster_num <= 2:
                cluster_num = 2
        
        # For subdivision, ensure at least 2 clusters
        if for_subdivision and isinstance(cluster_num, int):
            cluster_num = max(cluster_num, 2)
        
        return cluster_num

    def _compute_auto_cluster(self, not_rules_indexes_list, cluster_num="auto", parent_color: str = None, target_manually_specified: bool = False):
        """
        Compute auto-clustering on the given mask.
        
        Parameters
        ----------
        not_rules_indexes_list : pd.Series
            Boolean mask of points to cluster
        cluster_num : int or "auto"
            Target number of clusters
        parent_color : str, optional
            If provided, new regions will be shades of this color
        target_manually_specified : bool
            If True, the target number of clusters was explicitly set by the user (slider).
            Only in that case do we display it in the status message.
        """
        if not AUTO_CLUSTER_AVAILABLE:
            self.show_status("❌ Auto-clustering module not available", "error")
            return

        # Count True values in the mask (not the length of the Series)
        n_points_to_cluster = not_rules_indexes_list.sum() if hasattr(not_rules_indexes_list, 'sum') else sum(not_rules_indexes_list)
        
        if target_manually_specified and isinstance(cluster_num, int):
            self.show_status(f"🔄 Clustering {n_points_to_cluster} points (target: {cluster_num} clusters)...", "info")
        else:
            self.show_status(f"🔄 Clustering {n_points_to_cluster} points...", "info")
        
        if n_points_to_cluster > AppConfig.ATK_MIN_POINTS_NUMBER:
            vs_compute = int(not self.vs_pvs.is_computed(dim=3))
            es_compute = int(not self.es_pvs.is_computed(dim=3))
            steps = 1 + vs_compute + es_compute

            progress_bar = ProgressBar(self.auto_cluster_progress)
            pb1, pb2, pb3 = progress_bar.split(
                [vs_compute / steps * 100, (vs_compute + es_compute) / steps * 100]
            )
            vs_proj_3d_df = self.vs_pvs.get_current_X_proj(3, progress_callback=pb1)

            es_proj_3d_df = self.es_pvs.get_current_X_proj(3, progress_callback=pb2)

            ac = AutoCluster(self.data_store.X, pb3)
            n_points = not_rules_indexes_list.sum()
            
            if n_points > 50:
                try:
                    found_regions = ac.compute(
                        vs_proj_3d_df.loc[not_rules_indexes_list],
                        es_proj_3d_df.loc[not_rules_indexes_list],
                        cluster_num,
                    )  # type: ignore
                    self.show_status(f"✅ Found {len(found_regions)} clusters", "success")
                except Exception as e:
                    self.show_status(f"❌ Clustering error: {str(e)}", "error")
                    import traceback
                    traceback.print_exc()
                    found_regions = RegionSet(self.data_store.X)
            else:
                self.show_status(f"⚠️ Too few points ({n_points}), creating single region", "warning")
                found_regions = RegionSet(self.data_store.X)
                found_regions.add_region(
                    mask=pd.Series(
                        [True] * n_points, index=vs_proj_3d_df.loc[not_rules_indexes_list].index
                    ),
                    auto_cluster=True,
                )
            
            old_count = len(self.region_set)
            self.region_set.extend(found_regions)
            
            # Assign colors to new regions
            new_region_nums = [r.num for r in self.region_set.regions.values() 
                               if r.num > old_count or r.num not in [rr.num for rr in list(self.region_set.regions.values())[:old_count]]]
            
            if parent_color and new_region_nums:
                # Generate shades of parent color for subdivisions
                from antakia.gui.components.color_manager import generate_shades
                shades = generate_shades(parent_color, len(found_regions))
                for region, shade in zip(found_regions.regions.values(), shades):
                    region.color = shade
                    region._color = shade
            else:
                # Assign new colors from palette
                for region in found_regions.regions.values():
                    color = self.color_manager.get_color(region.num)
                    region.color = color
                    region._color = color
            
            self.show_status(f"✅ Created {len(self.region_set) - old_count} new regions (total: {len(self.region_set)})", "success")
            progress_bar(100)
        else:
            self.show_status(f"⚠️ Not enough points ({n_points_to_cluster} <= {AppConfig.ATK_MIN_POINTS_NUMBER})", "warning")

    def update_btns(self, current_operation=None):
        selected_region_nums = [x["Region"] for x in self.selected_regions]
        if current_operation:
            if current_operation["type"] == "select":
                selected_region_nums.append(current_operation["region_num"])
            elif current_operation["type"] == "unselect":
                selected_region_nums.remove(current_operation["region_num"])
        num_selected_regions = len(selected_region_nums)
        if num_selected_regions:
            first_region = self.region_set.get(selected_region_nums[0])
            enable_div = (num_selected_regions == 1) and bool(
                first_region.num_points() >= AppConfig.ATK_MIN_POINTS_NUMBER
            )
        else:
            enable_div = False

        # substitute - single region
        self.substitute_btn.disabled = num_selected_regions != 1

        # substitute all - always enabled if there are regions
        self.substitute_all_btn.disabled = len(self.region_set.regions) == 0

        # edit
        self.edit_btn.disabled = num_selected_regions != 1

        # divide
        self.divide_btn.disabled = not enable_div

        # merge
        enable_merge = num_selected_regions > 1
        self.merge_btn.disabled = not enable_merge

        # delete
        self.delete_btn.disabled = num_selected_regions == 0

        # auto_cluster
        self.auto_cluster_btn.disabled = self.auto_cluster_running
        self.cluster_num_wgt.disabled = bool(self.auto_cluster_checkbox.v_model)

    def region_selected(self, data):
        with Log("region_selected", 2):
            operation = {
                "type": "select" if data["value"] else "unselect",
                "region_num": data["item"]["Region"],
            }
            self.update_btns(operation)

    def clear_selected_regions(self):
        self.selected_regions = []
        self.update_btns(None)

    @log_errors
    def edit_region_clicked(self, *args):
        """
        Called when the user clicks on the 'divide' (region) button
        """
        with Log("edit_region", 2):
            stats_logger.log("edit_region")
            # we recover the region to sudivide
            region = self.region_set.get(self.selected_regions[0]["Region"])
            self.edit_callback(region)

    @log_errors
    def divide_region_clicked(self, *args):
        """
        Called when the user clicks on the 'divide' (region) button
        """
        with Log("divide_region", 2):
            stats_logger.log("divide_region")
            # we recover the region to sudivide
            region = self.region_set.get(self.selected_regions[0]["Region"])
            if region.num_points() > AppConfig.ATK_MIN_POINTS_NUMBER:
                # Save parent color for shading
                parent_color = region.color if hasattr(region, 'color') else None
                
                # Then we delete the region in self.region_set
                mask = region.mask.copy()
                self.color_manager.remove_color(region.num)
                self.region_set.remove(region.num)
                
                # we compute the subregions and add them to the region set
                cluster_num = self._get_cluster_num(for_subdivision=True)
                self.show_status(f"🔄 Dividing region {region.num}...", "info")
                self._compute_auto_cluster(
                    mask, cluster_num, parent_color=parent_color,
                    target_manually_specified=not self.auto_cluster_checkbox.v_model
                )
            # There is no more selected region
            self.clear_selected_regions()
            self.update_region_table()
            self.update_callback()

    @log_errors
    def merge_region_clicked(self, *args):
        """
        Called when the user clicks on the 'merge' (regions) button
        """
        with Log("merge_region", 2):
            selected_regions = [self.region_set.get(r["Region"]) for r in self.selected_regions]
            stats_logger.log("merge_region", {"num_regions": len(selected_regions)})
            mask = None
            for region in selected_regions:
                if mask is None:
                    mask = region.mask
                else:
                    mask |= region.mask

            # compute descriptive rules (multi-view RC7 si activé)
            vs_rules, _, score_dict, _ = find_descriptive_rules(
                mask,
                self.data_store.X,
                self.data_store.X_exp,
                variables=self.data_store.variables,
                multiview=self.data_store.multiview_rules_enabled,
                mode=self.data_store.multiview_rules_mode,
            )
            skr_rules_list = vs_rules
            stats_logger.log("merge_region_rules", score_dict)

            # delete regions
            for region in selected_regions:
                self.region_set.remove(region.num)
            # add new region
            if len(skr_rules_list) > 0:
                r = self.region_set.add_region(rules=skr_rules_list)
            else:
                r = self.region_set.add_region(mask=mask)
            self.selected_regions = [{"Region": r.num}]
            self.update_region_table()
            self.update_callback()

    @log_errors
    def delete_region_clicked(self, *args):
        """
        Called when the user clicks on the 'delete' (region) button
        """
        with Log("delete_region", 2):
            stats_logger.log("merge_region", {"num_regions": len(self.selected_regions)})
            for selected_region in self.selected_regions:
                region = self.region_set.get(selected_region["Region"])
                # Then we delete the regions in self.region_set
                self.region_set.remove(region.num)

            # There is no more selected region
            self.clear_selected_regions()
            self.update_region_table()
            self.update_callback()

    @log_errors
    def substitute_clicked(self, widget, event, data):
        """Substitute single selected region (interactive mode via Tab3)."""
        print("[Substitute] Button clicked")
        with Log("substitute_region", 2):
            stats_logger.log("substitute_region")
            region = self.region_set.get(self.selected_regions[0]["Region"])
            print(f"[Substitute] Region {region.num} selected")
            self.substitute_callback(region)

    def substitute_all_clicked(self, widget, event, data):
        """
        Auto-substitute all regions (or selected regions) with best model.
        
        If no regions selected: substitute ALL regions
        If regions selected: substitute only selected regions
        """
        # Visual feedback - show status
        self.show_status("🔄 Substitute All clicked - starting...", "info")
        
        # Visual feedback - change button text to show it was clicked
        original_children = self.substitute_all_btn.children
        self.substitute_all_btn.children = [
            v.Icon(class_="mr-2", children=["mdi-loading mdi-spin"]),
            "Processing..."
        ]
        self.substitute_all_btn.disabled = True
        
        try:
            # Get regions to substitute
            if self.selected_regions:
                regions = [self.region_set.get(r["Region"]) for r in self.selected_regions]
                self.show_status(f"🔄 Processing {len(regions)} selected regions...", "info")
            else:
                # No selection = all regions
                regions = list(self.region_set.regions.values())
                self.show_status(f"🔄 Processing ALL {len(regions)} regions...", "info")
            
            if not regions:
                self.show_status("⚠️ No regions to substitute!", "warning")
                # Restore button
                self.substitute_all_btn.children = original_children
                self.substitute_all_btn.disabled = False
                return
            
            stats_logger.log("substitute_all_regions", {"num_regions": len(regions)})
            
            # Auto-substitute all regions
            self._auto_substitute_regions(regions)
            
            self.show_status(f"✅ Successfully substituted {len(regions)} regions!", "success")
            
        except Exception as e:
            self.show_status(f"❌ Error: {str(e)}", "error")
            import traceback
            traceback.print_exc()
        finally:
            # Restore button state
            self.substitute_all_btn.children = original_children
            self.substitute_all_btn.disabled = len(self.region_set.regions) == 0
    
    def _auto_substitute_regions(self, regions):
        """
        Automatically substitute all given regions with best model.
        
        For each region:
        1. Train all substitution models
        2. Select the best model (lowest delta)
        3. Validate the region
        """
        total = len(regions)
        
        for i, region in enumerate(regions):
            try:
                self.show_status(f"🔄 Region {region.num} ({i+1}/{total}): Training models...", "info")
                
                # Train substitution models
                task_type = self.data_store.problem_category
                region.train_substitution_models(task_type=task_type)
                
                # Get performance table
                perfs = region.perfs
                
                if perfs is None or len(perfs) == 0:
                    self.show_status(f"⚠️ Region {region.num}: No models available", "warning")
                    continue
                
                # Select best model (sorted by delta ascending, so first is best)
                best_model_name = perfs.index[0]
                best_delta = perfs.loc[best_model_name, 'delta'] if 'delta' in perfs.columns else 0
                
                # Select and validate
                region.select_model(best_model_name)
                region.validate()
                
                self.show_status(f"✅ Region {region.num}: {best_model_name} (Δ={best_delta:.3f})", "success")
                
            except Exception as e:
                self.show_status(f"❌ Region {region.num}: {str(e)}", "error")
                import traceback
                traceback.print_exc()
        
        # Update UI
        self.update_region_table()
        self.update_callback()
    
    @log_errors
    def detect_outliers_clicked(self, widget, event, data):
        """
        Detect outliers and create or update the 'Outliers' region.

        Uses the selected method (IQR, Z-Score, or Isolation Forest) to detect
        outliers based on the target variable y and creates a mask-only region
        (no rules, since outliers don't fit clean interval rules).
        
        If an Outliers region already exists, it will be updated instead of
        creating a duplicate.
        """
        # Immediate feedback - button was clicked
        print("=" * 50)
        print("[Detect Outliers] Button clicked!")
        
        with Log("detect_outliers", 1):
            method = self.outlier_method_select.v_model
            print(f"[Detect Outliers] Method selected: {method}")
            
            y = self.data_store.y
            X = self.data_store.X
            
            if y is None:
                print("[Detect Outliers] ERROR: y is None!")
                return
                
            print(f"[Detect Outliers] Method: {method}, Dataset size: {len(y)}")

            outlier_mask = self._detect_outliers(y, X, method)
            n_outliers = int(outlier_mask.sum())

            print(f"[Detect Outliers] Found {n_outliers} outliers ({100*n_outliers/len(y):.1f}%)")

            if n_outliers == 0:
                stats_logger.log("detect_outliers", {"method": method, "count": 0})
                print("[Detect Outliers] No outliers found!")
                return

            stats_logger.log(
                "detect_outliers",
                {"method": method, "count": n_outliers, "pct": 100 * n_outliers / len(y)},
            )

            # Check if an Outliers region already exists
            existing_outlier_region = None
            for region in self.region_set.regions.values():
                if hasattr(region, '_outlier_method'):
                    existing_outlier_region = region
                    print(f"[Detect Outliers] Found existing Outliers region: {region.num}")
                    break
            
            # Remove outlier points from other existing regions
            other_regions = [r for r in self.region_set.regions.values() if r != existing_outlier_region]
            for other_region in other_regions:
                if other_region.mask is not None:
                    # Count how many points will be removed
                    overlap = (other_region.mask & outlier_mask).sum()
                    if overlap > 0:
                        print(f"[Detect Outliers] Removing {overlap} outliers from region {other_region.num}")
                        other_region.mask = other_region.mask & ~outlier_mask
            
            # Method name mapping for display
            method_names = {
                "iqr": "IQR",
                "zscore": "Z-Score", 
                "isolation_forest": "Isolation Forest"
            }
            display_method = method_names.get(method, method)
            outlier_region_name = f"Outliers ({display_method})"
            
            if existing_outlier_region is not None:
                # Update existing outliers region
                print(f"[Detect Outliers] Updating existing Outliers region (was {existing_outlier_region._outlier_method})")
                existing_outlier_region.mask = outlier_mask
                existing_outlier_region._outlier_method = method
                existing_outlier_region.name = outlier_region_name
                region = existing_outlier_region
            else:
                # Create new region with mask only (no rules)
                # This allows excluding scattered outliers that don't follow interval patterns
                region = self.region_set.add_region(
                    mask=outlier_mask,
                    auto_cluster=False,  # Not auto-cluster, it's outliers
                )
                # Mark as outlier region for future detection
                region._outlier_method = method
                region.name = outlier_region_name

            self.clear_selected_regions()
            self.update_region_table()
            self.update_callback()

            # Log result
            Log(f"Detected {n_outliers} outliers ({100*n_outliers/len(y):.1f}%) using {method}", 1)

    def _detect_outliers(self, y: pd.Series, X: pd.DataFrame, method: str) -> pd.Series:
        """
        Detect outliers using the specified method.

        Parameters
        ----------
        y : target variable
        X : features (used for Isolation Forest)
        method : 'iqr', 'zscore', or 'isolation_forest'

        Returns
        -------
        Boolean Series where True = outlier
        """
        if method == "iqr":
            # Interquartile Range method
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return (y < lower) | (y > upper)

        elif method == "zscore":
            # Z-Score method (>3 standard deviations)
            z_scores = np.abs((y - y.mean()) / y.std())
            return z_scores > 3

        elif method == "isolation_forest":
            # Isolation Forest (unsupervised anomaly detection)
            # Uses both X and y for detection
            data = X.copy()
            data["_target"] = y

            iso_forest = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
            predictions = iso_forest.fit_predict(data)

            # -1 = outlier, 1 = inlier
            return pd.Series(predictions == -1, index=y.index)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
