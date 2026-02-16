import time
from logging import Handler, Logger, DEBUG, Formatter, INFO

import ipywidgets as widgets
from IPython.core.display_functions import display

from antakia.config import AppConfig
from antakia.utils.stats import stats_logger


class Log:

    def __init__(self, msg: str, level, iter=-1):
        self.msg = msg
        self.level = level
        self._iter = iter
        self._start_time = time.time()
        self.ended = False

    def _print(self, msg, end):
        process_time = time.time() - self._start_time
        if self.level <= AppConfig.verbose and not self.ended:
            if AppConfig.log_with_time:
                msg = msg + f' {process_time:.2f} sec'
            print(f'{msg:<250}', end=end)

    def start(self):
        self._start_time = time.time()
        self._print(self.msg + ' ...', end='')
        return self

    def end(self):
        self._end_time = time.time()
        self._print('\r' + self.msg + f' done', end='\n')
        process_time = time.time() - self._start_time
        stats_logger.log('\r' + self.msg + f' done {process_time:.2f} sec')
        self.ended = True

    def iter(self, i):
        self._print('\r' + self.msg + f' ...[{i}/{self._iter}]', end='')

    def percent(self, percentage):
        self._print('\r' + self.msg + f' ...{percentage:.0f}%', end='')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.end()


class OutputWidgetHandler(Handler):
    """Custom logging handler sending logs to an output widget"""

    def __init__(self, height: int, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            "width": "100%",
            "height": str(height) + "px",
            "border": "1px solid black",
            "overflow_y": "auto"
        }
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """Overload of logging.Handler method"""
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = (new_output, ) + self.out.outputs

    def show_logs(self):
        """Show the logs"""
        display(self.out)

    def clear_logs(self):
        """Clear the current logs"""
        self.out.clear_output()


def conf_logger(logger: Logger, height: int = 160):
    if AppConfig.ATK_SHOW_LOG_MODULE_WIDGET:
        logger.setLevel(DEBUG)
        handler = OutputWidgetHandler(height)
        handler.setFormatter(
            Formatter(
                '%(asctime)s-%(levelname)s:%(module)s|%(lineno)s:: %(message)s'
            ))
        logger.addHandler(handler)
        handler.clear_logs()
        handler.show_logs()
    else:
        logger.setLevel(INFO)
