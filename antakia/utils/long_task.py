import time
from abc import ABC, abstractmethod

import pandas as pd


class LongTask(ABC):
    """
    Abstract class to compute long tasks, often in a separate thread.

    Attributes
    ----------
    X : dataframe
    progress_updated : an optional callback function to call when progress is updated
    start_time : float
    progress:int
    """

    def __init__(self, X: pd.DataFrame=None, progress_updated: callable = None):
        if X is None:
            raise ValueError("You must provide a dataframe for a LongTask")
        self.X = X
        self.progress_updated = progress_updated
        self.start_time = time.time()

    def publish_progress(self, progress: int):
        if self.progress_updated:
            self.progress_updated(progress, time.time() - self.start_time)

    @abstractmethod
    def compute(self, **kwargs) -> pd.DataFrame:
        """
        Method to compute the long task and update listener with the progress.
        """
        pass
