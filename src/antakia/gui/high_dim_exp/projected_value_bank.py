import pandas as pd
from antakia_core.data_handler.projected_values import ProjectedValues


class ProjectedValueBank:
    def __init__(self, y: pd.Series):
        self.projected_values = {}
        self.y = y

    def get_projected_values(self, X: pd.DataFrame) -> ProjectedValues:
        if id(X) not in self.projected_values:
            self.projected_values[id(X)] = ProjectedValues(X, self.y)
        return self.projected_values[id(X)]
