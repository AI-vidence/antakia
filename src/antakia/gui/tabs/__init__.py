"""
Tabs de l'interface AntakIA.

Modules disponibles :
- tab1 : Création de règles
- tab2 : Régions
- tab3 : Modèles
- tessellation_tab : Exploration des Tesselles
"""

from antakia.gui.tabs.tab1 import Tab1
from antakia.gui.tabs.tab2 import Tab2
from antakia.gui.tabs.tab3 import Tab3
from antakia.gui.tabs.tessellation_tab import (
    TessellationTab,
    TessellationTabConfig,
    create_hierarchy_figure,
    create_tessellation_figure,
)

__all__ = [
    "Tab1",
    "Tab2",
    "Tab3",
    "TessellationTab",
    "TessellationTabConfig",
    "create_tessellation_figure",
    "create_hierarchy_figure",
]
