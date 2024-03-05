from traitlets import traitlets
import ipyvuetify as v


class SubModelTable(v.VuetifyTemplate):
    headers = traitlets.List([]).tag(sync=True, allow_null=True)
    items = traitlets.List([]).tag(sync=True, allow_null=True)
    selected = traitlets.List([]).tag(sync=True, allow_null=True)
    template = traitlets.Unicode('''
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

    def set_callback(self, callback: callable):  # type: ignore
        self.callback = callback

    def vue_tableselect(self, data):
        self.callback(data)
