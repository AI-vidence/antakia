# User guide

Here is a simple use case of the AntaKIA package.

_(find more examples in the <a href="https://github.com/AI-vidence/antakia/tree/main/examples">this folder</a>)_

## :rocket: Launch the GUI

After installing the package (see [here](getting-started.md)), you can use the package in a notebook:

```python
import pandas as pd
df = pd.read_csv('data/california_housing.csv')
X = df.iloc[:,0:8]
Y = df.iloc[:,9]
SHAP = pd.read_csv('data/pre_computed_SHAP_values.csv')
```

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state = 9)
model.fit(X, Y)
```

```python
import antakia

dataset = antakia.Dataset(X, model = model, y=Y)
atk = antakia.AntakIA(dataset, explain = SHAP)
atk.startGUI()
```

![Screenshot 1](img/screenshot1.png)

## :mag: Create our first Potato

## :straight_ruler: Apply Skope Rules

## :control_knobs: Apply sub-model

## :white_check_mark: Validate the region

## :magic_wand: Have everything done for you : the magic button !