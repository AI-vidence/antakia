from typing import Callable

import ipyvuetify as v
from antakia_core.utils import colors
from traitlets.traitlets import List, Unicode


class ColorTable(v.VuetifyTemplate):
    """
    table to display regions with delta color coding
    """

    headers: List = List([]).tag(sync=True, allow_null=True)
    items: List = List([]).tag(sync=True, allow_null=True)
    selected: List = List([]).tag(sync=True, allow_null=True)
    colors: List = List(colors).tag(sync=True)
    template: Unicode = Unicode(
        """
        <template>
            <v-data-table
                v-model="selected"
                :headers="headers"
                :items="items"
                item-key="Region"
                show-select
                :hide-default-footer="false"
                @item-selected="tableselect"
                dense
            >
            <template #header.data-table-select></template>
            
            <!-- Region column with color chip -->
            <template v-slot:item.Region="{ item }">
              <v-chip :color="item.color" small>
                {{ item.Region }}
              </v-chip>
            </template>
            
            <!-- Sub-model column with delta color coding (vert=gain, rouge=perte, jaune=neutre) -->
            <template v-slot:item.Sub-model="{ item }">
              <div v-if="item['Sub-model']" style="display: flex; align-items: center;">
                <span>{{ item['Sub-model'] }}</span>
                <v-chip 
                  v-if="item.delta !== null && item.delta !== undefined"
                  x-small
                  :style="'background-color: ' + (item.delta_color || '#fff3a8') + '; color: #333; font-size: 10px;'"
                  class="ml-1"
                >
                  {{ item.delta > 0 ? '+' : '' }}{{ item.delta ? item.delta.toFixed(3) : '' }}
                </v-chip>
                <span 
                  v-if="item.overfit_risk"
                  class="ml-1"
                  :title="item.overfit_risk === '✓' ? 'Risque de sur-apprentissage faible (écart train/test &lt; 5%)' : item.overfit_risk === '⚠' ? 'Risque modéré (écart train/test 5-15%)' : 'Risque élevé (écart train/test &gt; 15%)'"
                  style="font-size: 10px; cursor: help;"
                >
                  {{ item.overfit_risk }}
                </span>
              </div>
              <span v-else style="color: #999;">-</span>
            </template>
            
            </v-data-table>
        </template>
        """
    ).tag(
        sync=True
    )  # type: ignore
    disable_sort = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.callback = None

    @staticmethod
    def get_color(item):
        return item.color

    # @click:row="tableclick"
    # def vue_tableclick(self, data):
    #     raise ValueError(f"click event data = {data}")

    def set_callback(self, callback: Callable):  # type: ignore
        self.callback = callback

    def vue_tableselect(self, data):
        self.callback(data)
