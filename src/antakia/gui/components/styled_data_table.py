"""
DataTable avec support fiable des classes CSS par ligne (fonds colorés, gras).
Utilise VuetifyTemplate car item-class du v.DataTable standard ne s'applique pas
correctement dans ipyvuetify.
"""

import ipyvuetify as v
from traitlets import Bool, List, Unicode


class StyledDataTable(v.VuetifyTemplate):
    """
    v-data-table avec :item-class fonctionnel pour styliser les lignes
    (fonds colorés par catégorie, gras pour le point typique).
    Chaque item doit avoir une clé __row_class__ avec les classes CSS.
    """

    headers = List([]).tag(sync=True, allow_null=True)
    items = List([]).tag(sync=True, allow_null=True)
    disabled = Bool(False).tag(sync=True)
    template = Unicode(
        """
        <template>
            <v-data-table
                :headers="headers"
                :items="items"
                :disabled="disabled"
                item-key="__index__"
                :item-class="(item) => (item && item['__row_class__']) ? item['__row_class__'] : ''"
                :show-select="false"
                :hide-default-footer="false"
                dense
            >
            </v-data-table>
        </template>
        """
    ).tag(sync=True)
