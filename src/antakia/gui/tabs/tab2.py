from functools import partial
from typing import Callable

import ipyvuetify as v
import numpy as np
import pandas as pd
from antakia_core.compute.skope_rule.skope_rule import skope_rules
from antakia_core.data_handler import RegionSet
from auto_cluster import AutoCluster
from sklearn.ensemble import IsolationForest

from antakia.config import AppConfig
from antakia.gui.graphical_elements.color_table import ColorTable
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
                {"text": "IQR (Interquartile Range)", "value": "iqr"},
                {"text": "Z-Score (>3σ)", "value": "zscore"},
                {"text": "Isolation Forest", "value": "isolation_forest"},
            ],
            label="Method",
            dense=True,
            style_="max-width: 200px",
            class_="ml-3 mt-1",
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
        self.auto_cluster_progress = v.ProgressLinear(  # 440212
            style_="width: 100%",
            class_="px-3",
            v_model=0,
            color="primary",
            height="15",
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
                                children=["Regions :"],
                            ),
                            self.region_table_wgt,
                            self.stats_wgt,
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
                                    self.auto_cluster_progress,
                                ],
                            ),
                        ],  # End v.Sheet Col 3 children
                    ),  # End v.Sheet Col 3
                ],  # End v.Sheet children
            ),  # End v.Sheet
        ]

        self.wire()
        self.update_region_table()

    def wire(self):
        self.region_table_wgt.set_callback(self.region_selected)
        self.substitute_btn.on_event("click", self.substitute_clicked)
        self.substitute_all_btn.on_event("click", self.substitute_all_clicked)
        self.edit_btn.on_event("click", self.edit_region_clicked)
        self.divide_btn.on_event("click", self.divide_region_clicked)
        self.merge_btn.on_event("click", self.merge_region_clicked)
        self.detect_outliers_btn.on_event("click", self.detect_outliers_clicked)
        self.delete_btn.on_event("click", self.delete_region_clicked)
        self.auto_cluster_btn.on_event("click", self.auto_cluster_clicked)
        self.auto_cluster_checkbox.v_model = True
        self.auto_cluster_checkbox.on_event("change", self.checkbox_auto_cluster_clicked)
        self.cluster_num_wgt.on_event("change", self.num_cluster_changed)

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

    def update_region_table(self):
        """
        Called to empty / fill the RegionDataTable and refresh plots
        """
        self.region_set.sort(by="size", ascending=False)
        temp_items = self.region_set.to_dict()

        # We populate the ColorTable :
        self.region_table_wgt.items = temp_items

        self.update_stats()
        self.update_btns()

    @log_errors
    def checkbox_auto_cluster_clicked(self, widget, event, data):
        """
        Called when the user clicks on the 'auto-cluster' checkbox
        """
        self.update_btns()

    @log_errors
    def auto_cluster_clicked(self, *args):
        """
        Called when the user clicks on the 'auto-cluster' button
        """
        # We disable the AC button. Il will be re-enabled when the AC progress is 100%
        with Log("auto_cluster", 2):
            self.auto_cluster_running = True
            # Feedback visuel immédiat : désactiver le bouton et activer la barre de progression
            self.auto_cluster_btn.disabled = True
            self.auto_cluster_btn.loading = True
            self.auto_cluster_btn.children = [
                v.Icon(class_="mr-2", children=["mdi-loading mdi-spin"]),
                "Computing...",
            ]
            self.auto_cluster_progress.indeterminate = True
            self.auto_cluster_progress.color = "primary"

            if self.region_set.stats()["coverage"] > 80:
                # UI rules :
                # region_set coverage is > 80% : we need to clear it to do another auto-cluster
                self.region_set.clear_unvalidated()

            # We assemble indices ot all existing regions :
            region_set_mask = self.region_set.mask
            not_rules_indexes_list = ~region_set_mask
            # We call the auto_cluster with remaining X and explained(X) :
            cluster_num = self._get_cluster_num()
            stats_logger.log(
                "auto_cluster",
                {
                    "cluster_num": cluster_num,
                    "vs_proj": str(self.vs_pvs.current_proj),
                    "es_proj": str(self.es_pvs.current_proj),
                },
            )

            self._compute_auto_cluster(not_rules_indexes_list, cluster_num)
            self.update_region_table()
            self.update_callback()
            # Restaurer l'état du bouton
            self.auto_cluster_btn.loading = False
            self.auto_cluster_btn.children = [
                v.Icon(class_="mr-2", children=["mdi-auto-fix"]),
                "Auto-clustering",
            ]
            self.auto_cluster_progress.indeterminate = False
            self.auto_cluster_progress.v_model = 0
            self.auto_cluster_progress.color = "grey"
            # We re-enable the button
            self.auto_cluster_running = False
            self.update_btns()

    def num_cluster_changed(self, *args):
        """
        Called when the user changes the number of clusters
        """
        with Log("num_cluster_changed", 2):
            self.update_btns()

    def _get_cluster_num(self):
        if self.auto_cluster_checkbox.v_model:
            cluster_num = "auto"
        else:
            cluster_num = self.cluster_num_wgt.v_model - len(self.region_set)
            if cluster_num <= 2:
                cluster_num = 2
        return cluster_num

    def _compute_auto_cluster(self, not_rules_indexes_list, cluster_num="auto"):
        if len(not_rules_indexes_list) > AppConfig.ATK_MIN_POINTS_NUMBER:
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
                # assertion in antakia ac
                found_regions = ac.compute(
                    vs_proj_3d_df.loc[not_rules_indexes_list],
                    es_proj_3d_df.loc[not_rules_indexes_list],
                    # We send 'auto' or we read the number of clusters from the Slider
                    cluster_num,
                )  # type: ignore
            else:
                found_regions = RegionSet(self.data_store.X)
                found_regions.add_region(
                    mask=pd.Series(
                        [True] * n_points, index=vs_proj_3d_df.loc[not_rules_indexes_list].index
                    ),
                    auto_cluster=True,
                )
            self.region_set.extend(found_regions)
            progress_bar(100)
        else:
            print("not enough points to cluster")

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

        # substitute all - multiple regions (at least 1)
        self.substitute_all_btn.disabled = num_selected_regions < 1

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
                # Then we delete the region in self.region_set
                self.region_set.remove(region.num)
                # we compute the subregions and add them to the region set
                cluster_num = self._get_cluster_num()
                self._compute_auto_cluster(region.mask, cluster_num)
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

            # compute skope rules
            skr_rules_list, _ = skope_rules(mask, self.data_store.X, self.data_store.variables)

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
        with Log("substitute_region", 2):
            stats_logger.log("substitute_region")
            region = self.region_set.get(self.selected_regions[0]["Region"])
            self.substitute_callback(region)

    @log_errors
    def substitute_all_clicked(self, widget, event, data):
        """Called when user clicks 'Substitute All' for batch substitution."""
        with Log("substitute_all_regions", 2):
            regions = [self.region_set.get(r["Region"]) for r in self.selected_regions]
            stats_logger.log("substitute_all_regions", {"num_regions": len(regions)})
            # Pass all selected regions to the callback
            self.substitute_callback(regions)

    @log_errors
    def detect_outliers_clicked(self, widget, event, data):
        """
        Detect outliers and create a special 'Outliers' region.

        Uses the selected method (IQR, Z-Score, or Isolation Forest) to detect
        outliers based on the target variable y and creates a mask-only region
        (no rules, since outliers don't fit clean interval rules).
        """
        with Log("detect_outliers", 1):
            method = self.outlier_method_select.v_model
            y = self.data_store.y
            X = self.data_store.X

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

            # Create region with mask only (no rules)
            # This allows excluding scattered outliers that don't follow interval patterns
            region = self.region_set.add_region(
                mask=outlier_mask,
                auto_cluster=True,  # Use auto_cluster flag so name shows "auto-cluster"
            )
            # Override the auto_cluster name with outliers info
            region._outlier_method = method  # Store method for display

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
