# Regional explaination for trustworthy AI by design

## Context

Most business needs and most famous ML algorithms themselves rely on ‘segmentation’. Classical ML model explaination tools like SHAP, LIME, do address the local (decision level) or global scales (model level), but none of them approach the ‘regional’ scale (segment). Understanding by both the data scientist (DS) and business owner (BO) often relies on that specific scale (be it Operation Regime, Customer segment) still not dealt with.

## Keywords

Regional explaination , segmentation, clustering, SHAP, LIME, decision trees, data visualisation, Banzhaf , Shapley Shubick , TreeRank , ranking forest

## Our dyadic approach

AntakIA methodology by AI-vidence aims at gaining a common understanding between the DS (data scientist) and firstly the BO (Business Owner) by building explainations of the models at a regional level. The main steps leading to Regional Explaination rely on ‘dyadic’ steps, implying simultaneously the Values Space (VS) and Explainations Space (ES) computed e.g. through Shapley values or other indexes. The ES is constructed by computing a local explanation for each data point of the data set. Quite a lot of explanation values can be used, depending on the goal and the situation. For example, LIME values can be more understandable when the phenomenom is rather linear. Shapley values have become a standard, and give and additive approach of the estimated importance of each predictor to account for the gap between the prediction and the mean. Correlations with other predictors can be a way to constitute such a space when the dataset is a set of windowed time series, for instance, etc.

## Dyadic visualisation

Visualize the VS and ES datasets at the same time through dimensions reduction approaches (e.g. PCA, t-SNE, UMAP, PACMAP,...) Especially xMAP approaches help at getting a sense of how close observations (in the VS) and explanations (in the ES) are. At this point, some obvious clusters can visually appear in both spaces.

## Dyadic exploration

Explore the simultaneously consistent zones of both spaces with DS and BO. Through manual selection of points, the DS or BO suggests to explore the dataset, starting either in the VS (classical clustering) or in the ES (explainations clustering : points close in the ES share similar explanations). Here, we aim at identifying the first regions, namely selection of points clustering simultaneously in the ES and the VS Some ML tools are useful at that step :

* data visualisation
* Skope Rules, to provide the simplest description of a selection of points through rules on values of predictors. The result mainly helps to understand the model behaviour when used in the ES. In the VS, those rules help get a first natural definition of a classical segment.
* DiCE, to exhibit an archetype of the segment, or a counterfactual (soon implemented !)
* causal inference and DAG (soon implemented !)
* ...

When one succeeds to identify such a doubly homogeneous selection, then a raw region is defined.

## Dyadic segmentation

Define precisely a region, describe as simply as possible each region in both spaces At this step, density graphs are used in both the VS and the ES to define precisely the segment, predictor-wise. For instance, the BO or the DS could consider sub-segmenting, given the distribution of the data points along the predictor(s) whole range of values. To remain esaily usable, the segment should be described through ranges of values in the VS.

## Dyadic understanding

Make sure it makes sense ! Through mutual explaination and understanding between the DS and BO, with complementary feature wise analyses. This ultimate step is essential, not to have spurious selection of points. This paves the way for the next important step of the methodology towards and explainable by design Model : defining a surrogate explianable model on each region.

These steps are to be iterated until all the VS is addressed.

## Result on toys dataset

On a simple simulated datasets with an explicitly 5 segment biased model (e.g. age below 25, or over 40 and man vs woman , etc.), we have been able through this dyadic approach to reconstruct the relevant segments learnt by a standard black box Model ( XGBoost )), considering simultaneously the original values and model explainations.

## Result on anomaly detection use case

We have been using this dyadic approach for anomaly detection on time series - defining a more understandable VS (with signal procesing and ad hoc aggregated features) - using unsupervised detection anomaly algorithms then SHAP to construct an ES

Do have a look to the ever evolving /demo folder for recent illustrative contributions.

## Prospects and future Work

AntakIA methodology also encompasses the building of surrogate models so as to construct Explainable , then hopefully Certifiable AI by design. We will be analysing the gain of using :

* other decision decision power indexes e.g. from games theory Banzhaf , Shapley Shubick , …),
* other specific explainations methods e.g. tree based extracted from surrogate logical models [P. Marquis],
* top performance surrogate models generic or more specific, such as TreeRank algorithms for ranking [S. Clemençon ].