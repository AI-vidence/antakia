from typing import Callable
from traitlets.traitlets import List, Unicode
import ipyvuetify as v


class SubModelTable(v.VuetifyTemplate):
    headers: List = List([]).tag(sync=True, allow_null=True)
    items: List = List([]).tag(sync=True, allow_null=True)
    selected: List = List([]).tag(sync=True, allow_null=True)
    template: Unicode = Unicode('''
        <template>
            <v-data-table
                v-model="selected"
                :headers="headers"
                :items="items"
                item-key="Sub-model"
                show-select
                single-select
                :hide-default-footer="false"
                @item-selected="tableselect"
            >

            <template v-slot:item.delta="{ item }">
              <v-chip :color="item.delta_color" label size="small">
                    {{ item.delta }}
              </v-chip>
            </template>
            </v-data-table>
        </template>
        ''').tag(sync=True)  # type: ignore
    disable_sort = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.callback = None

    # @click:row="tableclick"
    # def vue_tableclick(self, data):
    #     raise ValueError(f"click event data = {data}")

    def set_callback(self, callback: Callable):  # type: ignore
        self.callback = callback

    def vue_tableselect(self, data):
        self.callback(data)
