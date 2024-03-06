from functools import partial

import ipyvuetify as v

from antakia import config
from antakia.utils.stats import log_errors, stats_logger


class DimSwitch:
    def __init__(self, update_callback):
        self.update_callback = partial(update_callback,self)
        self._build_widget()

    def _build_widget(self):
        self.widget = v.Switch(  # 100 # Dimension switch
            v_on='tooltip.on',
            class_="ml-6 mr-2",
            v_model=config.ATK_DEFAULT_DIMENSION == 3,
            label="2D/3D",
        )

        self.widget.on_event("change", self.switch_dimension)

    @log_errors
    def switch_dimension(self, widget=None, event=None, data=None):
        dim = 3 if data else 2
        stats_logger.log('dim_changed', {'dim':dim})
        self.update_callback(dim)
