import unittest
import sklearn.linear_model as lm
from antakia.antakia import *
from antakia.dataset import *

# Define class to test the program
class TestAntakIA(unittest.TestCase):
	X = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
	y = pd.Series([1, 2, 3])
	model = lm.LinearRegression()
	dataset = Dataset(X = X, model = model, y = y)
	atk = AntakIA(dataset)
	
	def test_computing(self):
		self.atk.computeSHAP(verbose = False)
		self.atk.computeLIME(verbose = False)
		self.assertIsNotNone(self.atk.explain['SHAP'])
		self.assertIsNotNone(self.atk.explain['LIME'])


if __name__ == '__main__':
	unittest.main()
