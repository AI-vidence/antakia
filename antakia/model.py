import pandas as pd
from abc import ABC, abstractmethod


class Model() :

    @abstractmethod
    def predict(self, x:pd.DataFrame ) -> pd.Series:
        pass