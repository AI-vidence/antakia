"""
Utils module for the antakia package.
"""

import numpy as np
import pandas as pd

import os
from logging import Logger, Handler, Formatter, DEBUG, INFO


import ipywidgets as widgets
from ipywidgets.widgets.widget import Widget
import ipyvuetify as v
from IPython.display import display

import antakia.config as config
import logging as logging

# ---------------------------------------------------------------------

class OutputWidgetHandler(Handler):
    """Custom logging handler sending logs to an output widget"""

    def __init__(self, height:int, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {"width": "100%", "height": str(height)+"px", "border": "1px solid black", "overflow_y" : "auto"}
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

def conf_logger(logger : Logger, height:int =160) -> Handler:
    if config.SHOW_LOG_MODULE_WIDGET:
        logger.setLevel(DEBUG)
        handler = OutputWidgetHandler(height)
        handler.setFormatter(Formatter('%(asctime)s-%(levelname)s:%(module)s|%(lineno)s:: %(message)s'))
        logger.addHandler(handler)
        handler.clear_logs()
        handler.show_logs()
    else:
        logger.setLevel(INFO)

logger = logging.getLogger(__name__)
conf_logger(logger)
    
def wrap_repr(widget : Widget, size : int = 200) -> str:
    text = widget.__repr__()
    if widget.layout is None :
        text += " Layout is None !"
    else :
        text += " Visibility : "+ widget.layout.visibility
    s_wrap_list = textwrap.wrap(text, size)
    return  '\n'.join(s_wrap_list)

def overlap_handler(ens_potatoes, liste):
    # function that allows you to manage conflicts in the list of regions.
    # indeed, as soon as a region is added to the list of regions, the points it contains are removed from the other regions
    gliste = [x.indexes for x in ens_potatoes]
    for i in range(len(gliste)):
        a = 0
        for j in range(len(gliste[i])):
            if gliste[i][j - a] in liste:
                gliste[i].pop(j - a)
                a += 1
    for i in range(len(ens_potatoes)):
        ens_potatoes[i].setIndexes(gliste[i])
    return ens_potatoes

