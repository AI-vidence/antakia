from functools import partial

import ipyvuetify as v
import pandas as pd
from auto_cluster import AutoCluster
from antakia_core.compute.skope_rule.skope_rule import skope_rules
from antakia_core.data_handler.region import RegionSet

from antakia import config
from antakia.gui.graphical_elements.color_table import ColorTable
from antakia.gui.high_dim_exp.projected_values_selector import ProjectedValuesSelector
from antakia.gui.helpers.progress_bar import MultiStepProgressBar
from antakia.utils.stats import stats_logger, log_errors


class Tab2:
    region_headers = [
        {
            "text": column,
            "sortable": False,
            "value": column,
        }
        for column in ['Region', 'Rules', 'Average', 'Points', '% dataset', 'Sub-model']
    ]

    def __init__(
        self,
        variables,
        X: pd.DataFrame,
        vs_pvs: ProjectedValuesSelector,
        es_pvs: ProjectedValuesSelector,
        region_set: RegionSet,
        edit_callback: callable,
        update_callback: callable,
        substitute_callback: callable
    ):
        self.X = X
        self.vs_pvs = vs_pvs
        self.es_pvs = es_pvs
        self.variables = variables
        self.region_set = region_set
        self.edit_callback = partial(edit_callback, self)
        self.update_callback = partial(update_callback, self, self.region_set)
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
            v_on='tooltip.on',
            class_="ml-3 mt-8 green white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-swap-horizontal-circle-outline"
                    ],
                ),
                "Substitute",
            ],
        )
        self.divide_btn = v.Btn(  # 4401100
            v_on='tooltip.on',
            class_="ml-3 mt-8",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-content-cut"
                    ],
                ),
                "Divide",
            ],
        )
        self.edit_btn = v.Btn(  # 4401100
            v_on='tooltip.on',
            class_="ml-3 mt-3",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-pencil"
                    ],
                ),
                "Edit",
            ],
        )
        self.merge_btn = v.Btn(  # 4401200
            v_on='tooltip.on',
            class_="ml-3 mt-3",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-table-merge-cells"
                    ],
                ),
                "Merge",
            ],
        )
        self.delete_btn = v.Btn(  # 4401300
            v_on='tooltip.on',
            class_="ml-3 mt-3",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-delete"
                    ],
                ),
                "Delete",
            ],
        )
        self.auto_cluster_btn = v.Btn(  # 4402000
            v_on='tooltip.on',
            class_="mt-8 primary",
            children=[
                v.Icon(  # 44020000
                    class_="mr-2",
                    children=[
                        "mdi-auto-fix"
                    ],
                ),
                "Auto-clustering",
            ],
        )
        self.cluster_num_wgt = v.Slider(  # 4402100
            v_on='tooltip.on',
            class_="mt-10 mx-2",
            v_model=6,
            min=2,
            max=12,
            thumb_color='blue',  # marker color
            step=1,
            thumb_label="always"
        )
        self.auto_cluster_checkbox = v.Checkbox(  # 440211
            class_="px-3",
            v_model=True,
            label="Automatic number of clusters"
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
                        ]
                    ),  # End Col 1
                    v.Col(  # v.Sheet Col 2 = buttons #4401
                        class_="col-2",
                        style_="size: 50%",
                        children=[
                            v.Row(  # 44010
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440100
                                        bottom=True,
                                        v_slots=[
                                            {
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.substitute_btn
                                            }
                                        ],
                                        children=[
                                            'Find an explicable surrogate model on this region']
                                    )
                                ]
                            ),
                            v.Row(  # 44011
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440110
                                        bottom=True,
                                        v_slots=[
                                            {
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.divide_btn
                                            }
                                        ],
                                        children=[
                                            'Divide a region into sub-regions']
                                    )
                                ]
                            ),
                            v.Row(  # 44011
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440110
                                        bottom=True,
                                        v_slots=[
                                            {
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.edit_btn
                                            }
                                        ],
                                        children=[
                                            'Edit region\'s rules']
                                    )
                                ]
                            ),
                            v.Row(  # 44012
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440120
                                        bottom=True,
                                        v_slots=[
                                            {
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.merge_btn
                                            }
                                        ],
                                        children=[
                                            'Merge regions']
                                    )
                                ]
                            ),
                            v.Row(  # 44013
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440130
                                        bottom=True,
                                        v_slots=[
                                            {
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.delete_btn
                                            }
                                        ],
                                        children=[
                                            'Delete region']
                                    )
                                ]
                            ),
                        ]  # End v.Sheet Col 2 children
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
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.auto_cluster_btn
                                            }
                                        ],
                                        children=[
                                            'Find homogeneous regions in both spaces']
                                    )
                                ]
                            ),
                            v.Row(  # 44021
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 440210
                                        bottom=True,
                                        v_slots=[
                                            {
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.cluster_num_wgt,
                                            }
                                        ],
                                        children=['Number of clusters you expect to find']
                                    ),
                                    self.auto_cluster_checkbox,
                                    self.auto_cluster_progress
                                ]
                            ),
                        ]  # End v.Sheet Col 3 children
                    )  # End v.Sheet Col 3
                ]  # End v.Sheet children
            ),  # End v.Sheet
        ]

        self.wire()
        self.update_region_table()

    def wire(self):
        self.region_table_wgt.set_callback(self.region_selected)
        self.substitute_btn.on_event("click", self.substitute_clicked)
        self.edit_btn.on_event("click", self.edit_region_clicked)
        self.divide_btn.on_event("click", self.divide_region_clicked)
        self.merge_btn.on_event("click", self.merge_region_clicked)
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

    def update_stats(self):
        region_stats = self.region_set.stats()
        str_stats = [
            f"{region_stats['regions']} {'regions' if region_stats['regions'] > 1 else 'region'}",
            f"{region_stats['points']} points",
            f"{region_stats['coverage']}% of the dataset",
            f"{region_stats['delta_score']:.2f} subst score"
        ]
        self.stats_wgt.children = [', '.join(str_stats)]

    def update_region_table(self):
        """
        Called to empty / fill the RegionDataTable and refresh plots
        """
        self.region_set.sort(by='size', ascending=False)
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
        self.auto_cluster_running = True
        if self.region_set.stats()["coverage"] > 80:
            # UI rules :
            # region_set coverage is > 80% : we need to clear it to do another auto-cluster
            self.region_set.clear_unvalidated()

        # We assemble indices ot all existing regions :
        region_set_mask = self.region_set.mask
        not_rules_indexes_list = ~region_set_mask
        # We call the auto_cluster with remaining X and explained(X) :
        cluster_num = self._get_cluster_num()
        stats_logger.log('auto_cluster', {
            'cluster_num': cluster_num,
            'vs_proj': str(self.vs_pvs.current_proj),
            'es_proj': str(self.es_pvs.current_proj)
        })

        self._compute_auto_cluster(not_rules_indexes_list, cluster_num)
        self.update_region_table()
        self.update_callback()
        # We re-enable the button
        self.auto_cluster_running = False
        self.update_btns()

    def num_cluster_changed(self, *args):
        """
        Called when the user changes the number of clusters
        """
        self.update_btns()

    def _get_cluster_num(self):
        if self.auto_cluster_checkbox.v_model:
            cluster_num = "auto"
        else:
            cluster_num = self.cluster_num_wgt.v_model - len(self.region_set)
            if cluster_num <= 2:
                cluster_num = 2
        return cluster_num

    def _compute_auto_cluster(self, not_rules_indexes_list, cluster_num='auto'):
        if len(not_rules_indexes_list) > config.ATK_MIN_POINTS_NUMBER:
            vs_compute = int(not self.vs_pvs.is_computed(dim=3))
            es_compute = int(not self.es_pvs.is_computed(dim=3))
            steps = 1 + vs_compute + es_compute

            progress_bar = MultiStepProgressBar(self.auto_cluster_progress, steps=steps)
            step = 1
            vs_proj_3d_df = self.vs_pvs.get_current_X_proj(
                3,
                progress_callback=progress_bar.get_update(step)
            )

            step += vs_compute
            es_proj_3d_df = self.es_pvs.get_current_X_proj(
                3,
                progress_callback=progress_bar.get_update(step)
            )

            step += es_compute
            ac = AutoCluster(self.X, progress_bar.get_update(step))

            found_regions = ac.compute(
                vs_proj_3d_df.loc[not_rules_indexes_list],
                es_proj_3d_df.loc[not_rules_indexes_list],
                # We send 'auto' or we read the number of clusters from the Slider
                cluster_num,
            )  # type: ignore
            self.region_set.extend(found_regions)
            progress_bar.set_progress(100)
        else:
            print('not enough points to cluster')

    def update_btns(self, current_operation=None):
        selected_region_nums = [x['Region'] for x in self.selected_regions]
        if current_operation:
            if current_operation['type'] == 'select':
                selected_region_nums.append(current_operation['region_num'])
            elif current_operation['type'] == 'unselect':
                selected_region_nums.remove(current_operation['region_num'])
        num_selected_regions = len(selected_region_nums)
        if num_selected_regions:
            first_region = self.region_set.get(selected_region_nums[0])
            enable_div = (num_selected_regions == 1) and bool(first_region.num_points() >= config.ATK_MIN_POINTS_NUMBER)
        else:
            enable_div = False

        # substitute
        self.substitute_btn.disabled = num_selected_regions != 1

        # edit
        self.edit_btn.disabled = num_selected_regions != 1

        # divide
        self.divide_btn.disabled = not enable_div

        # merge
        enable_merge = (num_selected_regions > 1)
        self.merge_btn.disabled = not enable_merge

        # delete
        self.delete_btn.disabled = num_selected_regions == 0

        # auto_cluster
        self.auto_cluster_btn.disabled = self.auto_cluster_running
        self.cluster_num_wgt.disabled = bool(self.auto_cluster_checkbox.v_model)

    def region_selected(self, data):
        operation = {
            'type': 'select' if data['value'] else 'unselect',
            'region_num': data['item']['Region']
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
        stats_logger.log('edit_region')
        # we recover the region to sudivide
        region = self.region_set.get(self.selected_regions[0]['Region'])
        self.edit_callback(region)

    @log_errors
    def divide_region_clicked(self, *args):
        """
        Called when the user clicks on the 'divide' (region) button
        """
        stats_logger.log('divide_region')
        # we recover the region to sudivide
        region = self.region_set.get(self.selected_regions[0]['Region'])
        if region.num_points() > config.ATK_MIN_POINTS_NUMBER:
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

        selected_regions = [self.region_set.get(r['Region']) for r in self.selected_regions]
        stats_logger.log('merge_region', {'num_regions': len(selected_regions)})
        mask = None
        for region in selected_regions:
            if mask is None:
                mask = region.mask
            else:
                mask |= region.mask

        # compute skope rules
        skr_rules_list, _ = skope_rules(mask, self.X, self.variables)

        # delete regions
        for region in selected_regions:
            self.region_set.remove(region.num)
        # add new region
        if len(skr_rules_list) > 0:
            r = self.region_set.add_region(rules=skr_rules_list)
        else:
            r = self.region_set.add_region(mask=mask)
        self.selected_regions = [{'Region': r.num}]
        self.update_region_table()
        self.update_callback()

    @log_errors
    def delete_region_clicked(self, *args):
        """
        Called when the user clicks on the 'delete' (region) button
        """
        stats_logger.log('merge_region', {'num_regions': len(self.selected_regions)})
        for selected_region in self.selected_regions:
            region = self.region_set.get(selected_region['Region'])
            # Then we delete the regions in self.region_set
            self.region_set.remove(region.num)

        # There is no more selected region
        self.clear_selected_regions()
        self.update_region_table()
        self.update_callback()

    @log_errors
    def substitute_clicked(self, widget, event, data):
        stats_logger.log('substitute_region')
        region = self.region_set.get(self.selected_regions[0]['Region'])
        self.substitute_callback(region)
