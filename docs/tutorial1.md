# Tutorial

This is the part 1 of our tutorial. It explains how to prepare data and launch AntakIA. Those steps are common to most of AntakIA uses. If you feel familiar enough, you can directly jump to second part.

## The California housing dataset

We'll use the [California housing](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html) dataset, very famous in the datascience ecosystem.

This dataset describes 20 640 block groups (ie. groups of houses) in California using 8 variables :
* MedInc        median income in block group
* HouseAge      median house age in block group
* AveRooms      average number of rooms per household
* AveBedrms     average number of bedrooms per household
* Population    block group population
* AveOccup      average number of household members
* Latitude      block group latitude
* Longitude     block group longitude

The dataset also gives for each block group the average price of a house. This data comes from real real estate transactions.

In our noteboox, this dataset is stored in a Pandas Dataframe named `X`.
If you type `X.head()` you'll get :

![](img/head_X.png)

The "medium house values" are stored in a Pandas Series named `y`.
A `y.head()` will give you somethin like :

<div><img src="img/y.png" height="120"></div>

## The use case

We can imagine several use cases where AntakIA could be very useful. For instance :
* Let's say you're a real estate agent in California. A datascientist in your team has trained a wonderful ML model that is capable to predict the market value of any house in the state, as long as you provide sufficent data. You're amazed and want to unnderstand how this model works in order to gain insights of your market : what drives the price ? any segmentation ? So you decided to use AntakIA.
* Or, you don't have such model. But you still want to have an accurate understanding of your market. Then you ask a datascientist to train a model. ANd then you use AntakIA on it.

It's quite the same story : you have dataset `X`, you do a supervised training (`X`,`y`) to get a fitted model M. AntakIA will help you to understand how and why M can predict house values.

## Preparing the data

Open the file `california_housing.ipynb` (in the `examples` folder of the code, or wherever you downloaded it) 

Let's analyze the first cells :

```
import pandas as pd
df = pd.read_csv('../data/california_housing.csv').drop(['Unnamed: 0'], axis=1)
````
We start creating a dataframe from a local CSV file. You could have imported this dataset from the Scikit-learn package [here](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html). As you'll see, AntakIA needs to compute other values calues (eg. SHAP values for the data and the model). So make this tutorial quicker and more pleaseant, our CSV file includes these values pre-computed.

```
# Remove outliers:
df = df.loc[df['Population']<10000] 
df = df.loc[df['AveOccup']<6]
df = df.loc[df['AveBedrms']<1.5]
df = df.loc[df['HouseAge']<50]

# # Only San Francisco :
df = df.loc[(df['Latitude']<38.07)&(df['Latitude']>37.2)]
df = df.loc[(df['Longitude']>-122.5)&(df['Longitude']<-121.75)]
```
In the same way, the previous lines are not compulsory. But it appears that the dataset for the sole city of San Francisco is better to get rapidly a good intuition of how AntakIA works.

```
X = df.iloc[:,0:8] # the dataset
y = df.iloc[:,9] # the target variable
shap_values = df.iloc[:,[10,11,12,13,14,15,16,17]] # the SHAP values
```
Here we have extracted from our big CSV dataset : the `X` values, the `y` Series and the `shap_values` (we'll explain those values further).

## 