from traitlets.traitlets import List, Unicode
import ipyvuetify as v
from antakia_core.utils import colors
from typing import Callable


class ColorTable(v.VuetifyTemplate):
    """
    table to display regions
    """
    headers: List = List([]).tag(sync=True, allow_null=True)
    items: List = List([]).tag(sync=True, allow_null=True)
    selected: List = List([]).tag(sync=True, allow_null=True)
    colors: List = List(colors).tag(sync=True)
    template: Unicode = Unicode('''
        <template>
            <v-data-table
                v-model="selected"
                :headers="headers"
                :items="items"
                item-key="Region"
                show-select
                :hide-default-footer="false"
                @item-selected="tableselect"
            >
            <template #header.data-table-select></template>
            <template v-slot:item.Region="{ item }">
              <v-chip :color="item.color" >
                {{ item.Region }}
              </v-chip>
            </template>
            </v-data-table>
        </template>
        ''').tag(sync=True)  # type: ignore
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
