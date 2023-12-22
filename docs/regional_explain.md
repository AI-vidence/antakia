# Regional Explaination for Trustworthy AI by design

## Context
Most business needs and most used algorithms themselves rely on ‘segmentation’. Classical ML model explaination tools like SHAP, LIME, do address the local (decision level) or global
scales (model level), but none of them approach the ‘regional’ scale (segment).
Understanding by both the Data Scientist (DS) and Business Owner (BO) often relies on that specific scale (be it Operation Regime, Customer segment) still not dealt with.

## Keywords
Regional Explaination , segmentation, clustering, SHAP, LIME, Decision Trees , Data
Visualisation, Banzhaf , Shapley Shubick , TreeRank , Ranking Forest

## Method
*AntakIA* methodology by Aividence aims at gaining a common understanding between the DS and the BO by building explainations of the models at a regional level. The main steps leading
to Regional Explaination rely on ‘dyadic’ steps, implying simultaneously the Values Space (VS) and Explainations Space (ES ) computed e.g. through Shapley values or other indexes.
### DYADIC VISUALISATION : 
Visualize the VS and ES datasets at the same time through dimensions reduction approaches (t SNE, UMAP, PCA)
### DYADIC EXPLORATION : 
Explore the simultaneously consistent zones of both spaces with DS and BO
### DYADIC SEGMENTATION : 
Define precisely a region, describe as simply as possible each region in both spaces
### DYADIC UNDERSTANDING : 
Make sure it makes sense ! Through mutual explaination and understanding between the DS and BO, with complementary feature wise analyses.
These steps are to be iterated until all the VS is addressed.

## Result on toys dataset
On a simple simulated datasets with an explicitly 5 segment biased model (e.g. age below 25, or over 40 and man vs woman , etc.), we have been able through this dyadic approach to reconstruct the relevant segments learnt by a standard black box Model ( XGBoost )), considering simultaneously the original values and model explainations.

## Result on anomaly detection use case
We have been using this dyadic approach for anomaly detection on time series
- *defining a more understandable VS* (with signal procesing and ad hoc aggregated features)
- *using unsupervised detection anomaly algorithms then SHAP* to construct an ES

Do have a look to the ever evolving */demo* folder for recent illustrative contributions.

## Prospects and future Work
AntakIA methodology also encompasses the building of surrogate models so as to construct
Explainable , then hopefully Certifiable AI by design.
We will be analysing the gain of using :
- *other decision decision power indexes* e.g. from games theory Banzhaf , Shapley Shubick , …),
- *other specific explainations methods* e.g. tree based extracted from surrogate logical models [P. Marquis],
- *top performance surrogate models* generic or more specific, such as TreeRank algorithms for ranking [S. Clemençon ].
