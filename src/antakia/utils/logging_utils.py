from logging import Handler, Logger, DEBUG, Formatter, INFO

import ipywidgets as widgets
from IPython.core.display_functions import display
from ipywidgets import Widget

from antakia import config as config


class OutputWidgetHandler(Handler):
    """Custom logging handler sending logs to an output widget"""

    def __init__(self, height: int, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {"width": "100%", "height": str(height) + "px", "border": "1px solid black", "overflow_y": "auto"}
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """Overload of logging.Handler method"""
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def show_logs(self):
        """Show the logs"""
        display(self.out)

    def clear_logs(self):
        """Clear the current logs"""
        self.out.clear_output()


def conf_logger(logger: Logger, height: int = 160) -> Handler:
    if config.ATK_SHOW_LOG_MODULE_WIDGET:
        logger.setLevel(DEBUG)
        handler = OutputWidgetHandler(height)
        handler.setFormatter(Formatter('%(asctime)s-%(levelname)s:%(module)s|%(lineno)s:: %(message)s'))
        logger.addHandler(handler)
        handler.clear_logs()
        handler.show_logs()
    else:
        logger.setLevel(INFO)


def wrap_repr(widget: Widget, size: int = 200) -> str:
    text = widget.__repr__()
    if widget.layout is None:
        text += " Layout is None !"
    else:
        text += " Visibility : " + widget.layout.visibility
    s_wrap_list = textwrap.wrap(text, size)
    return '\n'.join(s_wrap_list)
