<<<<<<< HEAD
# User guide

Example of usage (find more examples in the <a href="https://code.ai-vidence.com/laurent/antakia/">examples</a> folder)
=======
# :eyes: Usage

Example of usage (find more example in the <a href="https://code.ai-vidence.com/laurent/antakia/">example</a> folder)
>>>>>>> d52101bce3abd5085f3ea47a4d5e1b95209494f0

In a notebook :

```python
import pandas as pd
df = pd.read_csv('data/california_housing.csv')
X = df.iloc[:,0:8]
Y = df.iloc[:,9]
```

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state = 9)
model.fit(X, Y)
```

```python
import antakia
explain = antakia.Xplainer(X = X, Y = Y, model = model)
display(explain.interface(explanation = SHAP, default_projection = "PaCMAP"))
```