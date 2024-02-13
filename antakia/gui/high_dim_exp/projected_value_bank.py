from antakia_core.data_handler.projected_values import ProjectedValues, Proj


class ProjectedValueBank:
    def __init__(self, y):
        self.projected_values = {}
        self.y = y

    def get_projected_values(self, X):
        if id(X) not in self.projected_values:
            self.projected_values[id(X)] = ProjectedValues(X, self.y)
        return self.projected_values[id(X)]
