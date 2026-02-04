from functools import partial
from typing import Callable, Optional

import ipyvuetify as v
import numpy as np
import pandas as pd
from antakia_core.compute.skope_rule.skope_rule import skope_rules
from antakia_core.data_handler import Region, RuleSet
from antakia_core.utils import format_data, timeit
from sklearn.ensemble import IsolationForest

from antakia.gui.components.beeswarm_plot import BeeswarmPlot
from antakia.gui.components.feature_dual_view import FeatureDualView
from antakia.gui.components.styled_data_table import StyledDataTable
from antakia.gui.components.realtime_rules import RealtimeRulesDebouncer
from antakia.gui.helpers.data import DataStore
from antakia.gui.tabs.ruleswidget import RulesWidget
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class Tab1:
    EDIT_RULE = "edition"
    CREATE_RULE = "creation"

    def __init__(
        self,
        data_store: DataStore,
        update_callback: Callable,
        validate_rules_callback: Callable,
        retire_outliers_callback: Optional[Callable] = None,
        exp_values=None,
    ):
        self.data_store = data_store
        self._region = Region(self.data_store.X)
        self.update_callback = partial(update_callback, self, "rule_updated")
        self.validate_rules_callback = partial(validate_rules_callback, self, "rule_validated")
        self.retire_outliers_callback = retire_outliers_callback
        self.exp_values = exp_values

        self.X_rounded = None
        self._outlier_mask: Optional[pd.Series] = None

        self.vs_rules_wgt = RulesWidget(self.data_store, True, self.refresh_callback)
        self.es_rules_wgt = RulesWidget(self.data_store, False, self.refresh_callback)

        self._build_widget()

    def _build_widget(self):
        self.find_rule_progress = v.ProgressCircular(
            class_="my-2", color="primary", width="6", indeterminate=False
        )
        self.find_rules_btn = v.Btn(  # 43010 Skope button
            v_on="tooltip.on",
            class_="ma-1 primary white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-axis-arrow"],
                ),
                "Find rules",
            ],
        )
        self.cancel_btn = v.Btn(  # 4302
            class_="ma-1",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-close-circle-outline"],
                ),
                "Cancel",
            ],
        )
        self.undo_btn = v.Btn(  # 4302
            class_="ma-1",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-undo"],
                ),
                "Undo",
            ],
        )
        self.validate_btn = v.Btn(  # 43030 validate
            v_on="tooltip.on",
            class_="ma-1 green white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-check"],
                ),
                "Validate rules",
            ],
        )
        self.title_wgt = v.Row(
            children=[
                v.Html(  # 44000
                    tag="h3",
                    class_="ml-2",
                    children=["Creating new region :"],
                )
            ]
        )
        self.selection_status_str_1 = v.Html(  # 430000
            tag="strong",
            children=["0 points"]
            # 4300000 # selection_status_str_1
        )
        self.selection_status_str_2 = v.Html(  # 43001
            tag="li",
            children=["0% of the dataset"]
            # 430010 # selection_status_str_2
        )
        self.points_count_label = v.Html(
            tag="li",
            style_="font-size: 11px; color: #666;",
            children=["Initial: - pts | Displayed: - pts"],
        )
        # Archetype / typical point
        self.archetype_label = v.Html(
            tag="li",
            style_="font-size: 11px; color: #666;",
            children=["Typical point: -"]
        )
        # Filtre par catégorie pour le tableau Data selected
        self._data_table_category_filter = "all"
        self.data_table_category_select = v.Select(
            v_model="all",
            items=[
                {"text": "Toutes", "value": "all"},
                {"text": "● matched", "value": "matched"},
                {"text": "● rule only", "value": "rule_only"},
                {"text": "● selection only", "value": "selection_only"},
                {"text": "● other", "value": "other"},
            ],
            label="Filtrer par catégorie",
            dense=True,
            hide_details=True,
            style_="max-width: 180px;",
            class_="mb-2",
        )
        self.data_table_category_select.on_event("change", self._on_data_table_filter_changed)

        self.data_table = StyledDataTable(
            headers=[
                {
                    "text": column,
                    "sortable": True,
                    "value": column,
                }
                for column in self.data_store.X.columns
            ],
            items=[],
            disabled=False,
        )
        # CSS pour les couleurs des lignes du tableau Data selected
        _data_table_css = (
            ".atk-row-matched { background-color: rgba(33, 150, 243, 0.25) !important; } "
            ".atk-row-rule-only { background-color: rgba(255, 152, 0, 0.25) !important; } "
            ".atk-row-selection-only { background-color: rgba(244, 67, 54, 0.25) !important; } "
            ".atk-row-other { background-color: rgba(158, 158, 158, 0.2) !important; } "
            ".atk-row-archetype { font-weight: bold !important; }"
        )
        self._data_table_style = v.Html(tag="style", children=[_data_table_css])
        self.data_panel = v.ExpansionPanel(  # 4320 # is enabled or disabled when no selection
            children=[
                v.ExpansionPanelHeader(  # 43200
                    class_="grey lighten-3", children=["Data selected"]
                ),
                v.ExpansionPanelContent(  # 43201
                    children=[
                        self._data_table_style,
                        v.Container(
                            fluid=True,
                            class_="pa-2",
                            children=[
                                self.data_table_category_select,
                                self.data_table,
                            ],
                        ),
                    ],
                ),
            ]
        )
        self._data_panel_expanded = False
        
        # Règles en temps réel
        self.auto_rules_checkbox = v.Checkbox(
            v_model=False,
            label="Règles automatiques",
            dense=True,
            hide_details=True,
            class_="ma-1",
        )
        self._realtime_rules_debouncer = RealtimeRulesDebouncer(
            self._do_compute_skope_rules,
            delay_seconds=0.6,
        )
        
        # Toggle Vue globale / Vue par feature
        self.view_mode_toggle = v.BtnToggle(
            v_model="global",
            mandatory=True,
            dense=True,
            class_="ma-1",
            children=[
                v.Btn(value="global", small=True, children=["Vue globale"]),
                v.Btn(value="feature", small=True, children=["Vue par feature"]),
            ],
        )
        self.view_mode_toggle.on_event("change", self._on_view_mode_changed)

        # Feature dual view (ES | VS par feature)
        self.feature_dual_view = FeatureDualView(self.data_store, height_per_feature=90)
        self.feature_dual_widget = self.feature_dual_view.build_widget()
        self._feature_view_content = (
            [self.feature_dual_widget]
            if self.feature_dual_widget is not None
            else [
                v.Html(
                    tag="p",
                    class_="ml-3 grey--text",
                    children=["SHAP non disponible — calculez les explications d'abord"],
                )
            ]
        )

        # Beeswarm SHAP - vue par feature (lien VS/ES), toujours modèle original
        def _original_model_shap_getter():
            if self.exp_values is None:
                return None
            # SHAP du modèle original : priorité SHAP calculé > Imported
            # Ne pas utiliser "or" : un DataFrame déclencherait "truth value ambiguous"
            shap_val = self.exp_values.explanations.get("SHAP")
            if shap_val is not None:
                return shap_val
            return self.exp_values.explanations.get("Imported")

        self.beeswarm_plot = BeeswarmPlot(
            self.data_store,
            height_per_feature=70,
            original_model_shap_getter=_original_model_shap_getter,
        )
        self.beeswarm_widget = self.beeswarm_plot.build_widget()
        _beeswarm_content = (
            [self.beeswarm_widget]
            if self.beeswarm_widget is not None
            else [
                v.Html(
                    tag="p",
                    class_="ml-3 grey--text",
                    children=["SHAP non disponible — calculez les explications d'abord"],
                )
            ]
        )
        self.beeswarm_panel = v.ExpansionPanel(
            children=[
                v.ExpansionPanelHeader(
                    class_="grey lighten-3",
                    children=["Vue SHAP par feature (beeswarm)"],
                ),
                v.ExpansionPanelContent(children=_beeswarm_content),
            ]
        )
        # Outliers: méthode, détection, retrait
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
            class_="mt-0 mr-2",
        )
        self.detect_outliers_btn = v.Btn(
            v_on="tooltip.on",
            class_="ma-1 orange white--text",
            children=[
                v.Icon(class_="mr-2", children=["mdi-alert-circle-outline"]),
                "Detect",
            ],
        )
        self.retire_outliers_btn = v.Btn(
            v_on="tooltip.on",
            class_="ma-1 red white--text",
            disabled=True,
            children=[
                v.Icon(class_="mr-2", children=["mdi-delete-sweep"]),
                "Remove",
            ],
        )
        self.outlier_status = v.Html(tag="span", class_="ml-2 grey--text", children=["-"])

        # Légende des couleurs pour la sélection et les règles
        self.color_legend = v.Sheet(
            class_="ml-auto mr-3 pa-2 d-flex flex-row align-center",
            style_="font-size: 12px;",
            children=[
                v.Html(tag="span", class_="mr-3 font-weight-bold", children=["Legend:"]),
                v.Html(
                    tag="span", style_="color: blue; margin-right: 12px;", children=["● matched"]
                ),
                v.Html(
                    tag="span",
                    style_="color: orange; margin-right: 12px;",
                    children=["● rule only"],
                ),
                v.Html(
                    tag="span",
                    style_="color: red; margin-right: 12px;",
                    children=["● selection only"],
                ),
                v.Html(tag="span", style_="color: grey;", children=["● other"]),
            ],
        )
        self.widget = [
            self.title_wgt,
            v.Row(  # buttons row # 430
                class_="d-flex flex-row align-top mt-2",
                children=[
                    v.Sheet(  # Selection info # 4300
                        class_="ml-3 mr-3 pa-2 align-top grey lighten-3",
                        style_="width: 25%",
                        elevation=1,
                        children=[
                            v.Html(tag="li", children=[self.selection_status_str_1]),  # 43000
                            self.selection_status_str_2,
                            self.points_count_label,
                            self.archetype_label,
                        ],
                    ),
                    v.Tooltip(  # 4301
                        bottom=True,
                        v_slots=[
                            {
                                "name": "activator",
                                "variable": "tooltip",
                                "children": self.find_rules_btn,
                            }
                        ],
                        children=["Find a rule to match the selection"],
                    ),
                    self.view_mode_toggle,
                    self.auto_rules_checkbox,
                    self.find_rule_progress,
                    self.cancel_btn,
                    self.undo_btn,
                    v.Tooltip(  # 4303
                        bottom=True,
                        v_slots=[
                            {
                                "name": "activator",
                                "variable": "tooltip",
                                "children": self.validate_btn,
                            }
                        ],
                        children=["Promote current rules as a region"],
                    ),
                    self.color_legend,  # Légende des couleurs
                ],
            ),  # End Buttons row
            v.Row(
                class_="d-flex flex-row align-center mt-2 ml-3",
                children=[
                    v.Html(tag="strong", class_="mr-2", children=["Outliers:"]),
                    self.outlier_method_select,
                    v.Tooltip(
                        bottom=True,
                        v_slots=[
                            {"name": "activator", "variable": "tooltip", "children": self.detect_outliers_btn},
                        ],
                        children=["Detect outliers (y or X+y depending on method)"],
                    ),
                    v.Tooltip(
                        bottom=True,
                        v_slots=[
                            {"name": "activator", "variable": "tooltip", "children": self.retire_outliers_btn},
                        ],
                        children=[
                            "Remove outliers from dataset and regenerate graphs. "
                            "Regions (Parcels tab) will be created on displayed points (excluding outliers)."
                        ],
                    ),
                    self.outlier_status,
                ],
            ),
            v.Container(
                fluid=True,
                class_="pa-0",
                children=[
                    v.Row(
                        class_="d-flex flex-row",
                        children=[
                            self.vs_rules_wgt.widget,
                            self.es_rules_wgt.widget,
                        ],
                    ),
                    v.ExpansionPanels(
                        class_="d-flex flex-row",
                        children=[self.data_panel, self.beeswarm_panel],
                    ),
                ],
            ),
        ]
        self._global_content = self.widget[3].children
        self._feature_content = [
            v.Row(
                class_="mt-2",
                children=[v.Col(children=self._feature_view_content)],
            )
        ]
        self._content_container = self.widget[3]
        # get_widget(self.widget[2], "0").disabled = True  # disable datatable
        # We wire the click events
        self.find_rules_btn.on_event("click", self.compute_skope_rules)
        self.undo_btn.on_event("click", self.undo_rules)
        self.cancel_btn.on_event("click", self.cancel_edit)
        self.validate_btn.on_event("click", self.validate_rules)
        self.data_panel.on_event("click", self.data_panel_changed)
        self.detect_outliers_btn.on_event("click", self._detect_outliers_clicked)
        self.retire_outliers_btn.on_event("click", self._retire_outliers_clicked)
        self._refresh_buttons()

    @timeit
    def initialize(self):
        self.vs_rules_wgt.initialize()
        self.es_rules_wgt.initialize()
        self._refresh_points_count()

    @property
    def _valid_selection(self) -> bool:
        return not self.data_store.empty_selection

    @property
    def edit_type(self) -> str:
        return self.CREATE_RULE if self._region.num == -1 else self.EDIT_RULE

    def _refresh_points_count(self):
        """Display initial and displayed point counts (excluding outliers if removed)."""
        n_init = self.data_store.n_initial_points
        n_curr = len(self.data_store.X)
        suffix = " (outliers removed)" if n_init != n_curr else ""
        self.points_count_label.children = [
            f"Initial: {n_init:,} pts | Displayed: {n_curr:,} pts{suffix}"
        ]

    def _refresh_selection_stat_card(self):
        if self._valid_selection:
            n_selected = self.data_store.selection_mask.sum()
            selection_status_str_1 = f"{n_selected} point selected"
            selection_status_str_2 = (
                f"{100 * self.data_store.selection_mask.mean():.2f}% of the  dataset"
            )
            # Compute archetype (typical point)
            archetype_str = self._compute_archetype_str()
        else:
            selection_status_str_1 = "0 point selected"
            selection_status_str_2 = "0% of the  dataset"
            archetype_str = "Typical point: -"
        self.selection_status_str_1.children = [selection_status_str_1]
        self.selection_status_str_2.children = [selection_status_str_2]
        self.archetype_label.children = [archetype_str]
        self._refresh_points_count()
    
    def _compute_archetype_str(self) -> str:
        """Compute the archetype (point closest to centroid) for current selection."""
        import numpy as np
        
        mask = self.data_store.selection_mask
        if mask.sum() == 0:
            return "Typical point: -"
        
        X_sel = self.data_store.X[mask]
        
        if len(X_sel) == 1:
            # Single point selected
            idx = X_sel.index[0]
            return f"Typical point: #{idx}"
        
        # Compute centroid
        centroid = X_sel.mean().values
        
        # Find closest point
        distances = np.linalg.norm(X_sel.values - centroid, axis=1)
        closest_idx = distances.argmin()
        archetype_idx = X_sel.index[closest_idx]
        
        return f"Typical point: #{archetype_idx}"

    def _refresh_title_txt(self):
        if self.edit_type == "creation":
            self.title_wgt.children = [
                v.Html(  # 44000
                    tag="h3",
                    class_="ml-2",
                    children=["Creating new region :"],
                )
            ]
        else:
            region_prefix_wgt = v.Html(
                class_="mr-2", tag="h3", children=["Editing Region"]
            )  # 450000
            region_chip_wgt = v.Chip(
                color=self._region.color,
                children=[str(self._region.num)],
            )
            self.title_wgt.children = [
                v.Sheet(  # 45000
                    class_="ma-1 d-flex flex-row align-center",
                    children=[region_prefix_wgt, region_chip_wgt],
                )
            ]

    def _refresh_beeswarm_content(self):
        """Rebuild beeswarm widget if SHAP available. Refresh rebuilds (Plotly forbids data assign)."""
        if self.data_store.X is None:
            return
        # Réessayer de construire si le widget n'existe pas encore (SHAP peut arriver plus tard)
        if self.beeswarm_widget is None:
            self.beeswarm_widget = self.beeswarm_plot.build_widget()
            if self.beeswarm_widget is not None:
                self.beeswarm_panel.children[1].children = [self.beeswarm_widget]
        if self.beeswarm_widget is not None:
            self.beeswarm_plot.refresh()
            self.beeswarm_panel.children[1].children = [self.beeswarm_widget]

    def _refresh_feature_dual_content(self):
        """Rebuild feature dual view if X_exp just became available."""
        if self.feature_dual_widget is not None:
            return
        if self.data_store.X_exp is not None and self.data_store.X is not None:
            self.feature_dual_widget = self.feature_dual_view.build_widget()
            if self.feature_dual_widget is not None:
                self._feature_view_content = [self.feature_dual_widget]
                self._feature_content[0].children[0].children = self._feature_view_content

    def _data_table_row_class_str(self, category: str, is_archetype: bool) -> str:
        """Construit la chaîne de classes CSS pour une ligne."""
        classes = []
        if category == "matched":
            classes.append("atk-row-matched")
        elif category == "rule_only":
            classes.append("atk-row-rule-only")
        elif category == "selection_only":
            classes.append("atk-row-selection-only")
        elif category == "other":
            classes.append("atk-row-other")
        if is_archetype:
            classes.append("atk-row-archetype")
        return " ".join(classes)

    def _on_data_table_filter_changed(self, widget, event, data):
        """Callback quand le filtre de catégorie change."""
        self._data_table_category_filter = data or "all"
        self._refresh_data_table()

    def _refresh_data_table(self):
        """Met à jour le tableau Data selected. Toujours peupler pour refléter la sélection VS/ES."""
        if self.data_store.X is None:
            self.data_table.items = []
            return
        # Sync filter from select (au cas où)
        self._data_table_category_filter = (
            getattr(self.data_table_category_select, "v_model", None) or "all"
        )
        if self.X_rounded is None:
            self.X_rounded = self.data_store.X.apply(format_data)
        # Masque de base : tous les points (pour avoir les 4 catégories)
        mask_all = self.data_store.display_mask.reindex(
            self.X_rounded.index, fill_value=False
        ).astype(bool)
        # Catégories : matched, rule_only, selection_only, other
        rules_mask = self.data_store.rules_mask.reindex(
            self.X_rounded.index, fill_value=False
        ).astype(bool)
        selection_mask = self.data_store.selection_mask.reindex(
            self.X_rounded.index, fill_value=False
        ).astype(bool)
        archetype_idx = self.data_store.get_archetype_idx()

        # Catégories depuis les masks (aligné sur rule_selection_color)
        category_series = pd.Series(index=self.X_rounded.index, dtype=str)
        category_series[selection_mask & rules_mask] = "matched"
        category_series[~selection_mask & rules_mask] = "rule_only"
        category_series[selection_mask & ~rules_mask] = "selection_only"
        category_series[~selection_mask & ~rules_mask] = "other"

        # Points à afficher : selection | rules (ou tous si filtre "other")
        show_mask = mask_all & (selection_mask | rules_mask)
        if self._data_table_category_filter != "all":
            cat_mask = category_series == self._data_table_category_filter
            show_mask = mask_all & cat_mask
        else:
            show_mask = mask_all

        df = self.X_rounded.loc[show_mask].copy()
        df["__category__"] = category_series.loc[show_mask]
        df["__index__"] = df.index
        df["__archetype__"] = df.index == archetype_idx if archetype_idx is not None else False
        df["__row_class__"] = df.apply(
            lambda r: self._data_table_row_class_str(
                r["__category__"], r["__archetype__"]
            ),
            axis=1,
        )

        records = df.to_dict("records")
        self.data_table.items = records

    @timeit
    def refresh_X_exp(self):
        self.es_rules_wgt.change_underlying_dataframe(self.data_store.X_exp)
        self.es_rules_wgt.refresh()
        self._refresh_beeswarm_content()
        self._refresh_feature_dual_content()

    def _on_view_mode_changed(self, widget, event, data):
        """Switch between global view (rules + data) and feature view (dual ES|VS)."""
        if data == "feature":
            self._content_container.children = self._feature_content
            if self.feature_dual_widget is not None:
                self.feature_dual_view.refresh()
        else:
            self._content_container.children = self._global_content

    def _refresh_buttons(self):
        empty_rule_set = len(self.vs_rules_wgt.current_rules_set) == 0
        empty_history = self.vs_rules_wgt.history_size <= 1

        # data table
        self.data_table.disabled = not self._valid_selection
        # self.widget[2].children[0].disabled = not self.valid_selection
        self.find_rules_btn.disabled = not self._valid_selection
        self.undo_btn.disabled = empty_history
        self.cancel_btn.disabled = empty_rule_set and empty_history

        has_modif = (self.vs_rules_wgt.history_size > 1) or (
            self.es_rules_wgt.history_size == 1
            and self._region.num < 0  # do not validate a empty modif
        )
        self.validate_btn.disabled = not has_modif or empty_rule_set

    @timeit
    def refresh(self):
        self.vs_rules_wgt.refresh()
        self.es_rules_wgt.refresh()
        self._refresh_title_txt()
        self._refresh_data_table()
        self._refresh_selection_stat_card()
        self._refresh_buttons()
        # Refresh beeswarm SHAP view (rebuild if X_exp just became available)
        self._refresh_beeswarm_content()
        # Refresh feature dual view when in feature mode
        if self.view_mode_toggle.v_model == "feature":
            self._refresh_feature_dual_content()
            if self.feature_dual_widget is not None:
                self.feature_dual_view.refresh()

        # Règles en temps réel (debounced) - ne pas déclencher si on est en train de calculer
        if (
            self.auto_rules_checkbox.v_model
            and self._valid_selection
            and not getattr(self, "_computing_rules", False)
        ):
            self._realtime_rules_debouncer.trigger()

    @timeit
    def reset(self):
        self._region = Region(self.data_store.X)
        self.X_rounded = None  # Recompute after data change (e.g. outlier removal)
        self.data_store.reset_rules_mask()
        self.vs_rules_wgt.change_rules(RuleSet(), True)
        self.es_rules_wgt.change_rules(RuleSet(), True)
        self.refresh()

    @timeit
    def update_region(self, region: Region):
        self._region = region
        self.data_store._rules_mask = region.mask.copy()  # type: ignore
        self.data_store.selection_mask = region.mask.copy()

        self._refresh_title_txt()
        self.vs_rules_wgt.change_rules(region.rules, True)
        self.es_rules_wgt.change_rules(RuleSet(), True)
        self.refresh()

    # ----------- interactions -----------------#
    
    def _do_compute_skope_rules(self):
        """Called by realtime rules debouncer."""
        self.compute_skope_rules()
    
    @log_errors
    @timeit
    def compute_skope_rules(self, *args):
        with Log("compute_skope_rules", 2):
            self._computing_rules = True
            self.find_rule_progress.indeterminate = True
            self.find_rules_btn.disabled = True
            try:
                # compute es rules for info only
                es_skr_rules_set, _ = skope_rules(
                    self.data_store.selection_mask,
                    self.data_store.X_exp,
                    variables=self.data_store.variables,
                )
                self.es_rules_wgt.change_rules(es_skr_rules_set, False)
                # compute rules on vs space
                skr_rules_set, skr_score_dict = skope_rules(
                    self.data_store.selection_mask,
                    self.data_store.X,
                    variables=self.data_store.variables,
                )
                self.data_store.rules_mask = skr_rules_set.get_matching_mask(self.data_store.X)
                skr_score_dict["target_avg"] = self.data_store.y[self.data_store.selection_mask].mean()
                self.vs_rules_wgt.change_rules(skr_rules_set, False)
                self.refresh()
                self.update_callback()
                stats_logger.log("find_rules", skr_score_dict)
            finally:
                self._computing_rules = False
                self.find_rules_btn.disabled = False
                self.find_rule_progress.indeterminate = False

    @log_errors
    @timeit
    def undo_rules(self, *args):
        with Log("undo_rules", 2):
            if self.vs_rules_wgt.history_size > 0:
                self.vs_rules_wgt.undo()
            self._refresh_buttons()

    @log_errors
    @timeit
    def cancel_edit(self, *args):
        with Log("cancel_edit", 2):
            self.reset()
            self.update_callback()

    @timeit
    @log_errors
    def refresh_callback(self, caller, event: str):
        """
        Called by a RulesWidget Skope rule creation or when the user wants new rules to be plotted
        The function asks the HDEs to display the rules result
        """
        stats_logger.log("rule_changed")
        self.refresh()
        # We sent to the proper HDE the rules_indexes to render :
        self.update_callback()

    @log_errors
    @timeit
    def validate_rules(self, *args):
        with Log("validate_rules", 2):
            stats_logger.log("validate_rules")
            # get rule set and check validity
            rules_set = self.vs_rules_wgt.current_rules_set
            if len(rules_set) == 0:
                stats_logger.log("validate_rules", info={"error": "invalid rules"})
                self.vs_rules_wgt.show_msg(
                    "No rules found on Value space cannot validate region", "red--text"
                )
                return

            # we persist the rule set in the region
            self._region.update_rule_set(rules_set)
            # we ship the region to GUI to synchronize other tab
            self.validate_rules_callback(self._region)
            # we reset the tab
            self.reset()

    @log_errors
    def _detect_outliers_clicked(self, widget, event, data):
        """Detect outliers and enable the Remove button."""
        X = self.data_store.X
        y = self.data_store.y
        if y is None or len(y) == 0:
            self.outlier_status.children = ["y missing"]
            return
        method = getattr(self.outlier_method_select, "v_model", "iqr") or "iqr"
        self._outlier_mask = self._detect_outliers(y, X, method)
        n_out = int(self._outlier_mask.sum())
        if n_out == 0:
            self.outlier_status.children = ["No outliers detected"]
            self.retire_outliers_btn.disabled = True
        else:
            pct = 100 * n_out / len(y)
            self.outlier_status.children = [f"{n_out} outliers ({pct:.1f}%)"]
            self.retire_outliers_btn.disabled = False

    def _detect_outliers(self, y: pd.Series, X: pd.DataFrame, method: str) -> pd.Series:
        """Return a boolean mask where True = outlier."""
        if method == "iqr":
            Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            return (y < lower) | (y > upper)
        elif method == "zscore":
            z = np.abs((y - y.mean()) / y.std())
            return z > 3
        elif method == "isolation_forest":
            data = X.copy()
            data["_target"] = y
            iso = IsolationForest(contamination="auto", random_state=42, n_estimators=100)
            pred = iso.fit_predict(data)
            return pd.Series(pred == -1, index=y.index)
        raise ValueError(f"Unknown method: {method}")

    @log_errors
    def _retire_outliers_clicked(self, widget, event, data):
        """Remove outliers and regenerate VS/ES graphs."""
        if self._outlier_mask is None or self._outlier_mask.sum() == 0:
            return
        if self.retire_outliers_callback is None:
            self.outlier_status.children = ["Callback not configured"]
            return
        self.retire_outliers_callback(self._outlier_mask)
        self._outlier_mask = None
        self.retire_outliers_btn.disabled = True
        self.outlier_status.children = ["Outliers removed, graphs regenerated"]

    @timeit
    def data_panel_changed(self, *args):
        self._data_panel_expanded = not self._data_panel_expanded
        self._refresh_data_table()
