from antakia.compute.model_subtitution.model_class import MLModel, LinearMLModel, AvgClassificationBaseline
from sklearn import linear_model, tree



####################################
# Baseline Models                  #
####################################

class AvgClassificationBaselineModel(MLModel):
    def __init__(self):
        super().__init__(AvgClassificationBaseline(), 'average baseline')


####################################
# Linear Models                    #
####################################

class LogisticRegression(LinearMLModel):
    def __init__(self):
        super().__init__(linear_model.LogisticRegression(), 'linear regression')


class DecisionTreeClassifier(MLModel):
    def __init__(self):
        super().__init__(tree.DecisionTreeClassifier(), 'Decision Tree')
