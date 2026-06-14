"""
Reusable GUI components for AntakIA.
"""

from antakia.gui.components.action_history import ActionHistory, action_history
from antakia.gui.components.notifications import NotificationManager, notifications
from antakia.gui.components.color_manager import (
    ColorManager, 
    color_manager, 
    ALL_PALETTES,
    generate_shades,
)
from antakia.gui.components.beeswarm_plot import BeeswarmPlot, create_beeswarm_shap_plot
from antakia.gui.components.realtime_rules import RealtimeRulesDebouncer

__all__ = [
    "NotificationManager",
    "notifications",
    "ActionHistory",
    "action_history",
    "ColorManager",
    "color_manager",
    "ALL_PALETTES",
    "generate_shades",
    "BeeswarmPlot",
    "create_beeswarm_shap_plot",
    "RealtimeRulesDebouncer",
    "FeatureDualView",
    "create_feature_dual_figure",
]
