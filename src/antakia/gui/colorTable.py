from traitlets import traitlets
import ipyvuetify as v
from antakia_core.utils.utils import colors


class ColorTable(v.VuetifyTemplate):
    """
    table to display regions
    """
    headers = traitlets.List([]).tag(sync=True, allow_null=True)
    items = traitlets.List([]).tag(sync=True, allow_null=True)
    selected = traitlets.List([]).tag(sync=True, allow_null=True)
    colors = traitlets.List(colors).tag(sync=True)
    template = traitlets.Unicode('''
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

    def set_callback(self, callback: callable):  # type: ignore
        self.callback = callback

    def vue_tableselect(self, data):
        self.callback(data)
