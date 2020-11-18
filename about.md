# Project Description
In this project, the dataset contains house sale prices for King County, which includes Seattle, from Washington state of USA. The data is taken from kaggle competition [House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction). It includes homes sold between May 2014 and May 2015. There are 19 house features and one dependent feature `price`. The aim of the project is to estimate the house price. The features that determines the house price are given below:

```
id # index feature
price # target feature

date    yr_built yr_renovated
zipcode lat      long

bedrooms bathrooms condition
grade    floors    waterfront view

sqft_living   sqft_lot       sqft_above
sqft_basement sqft_living15  sqft_lot15
```
These are the original features in the dataset. But during feature engineering we create lots of other features and also one hot encode the categorical variables `bedrooms bathrooms condition grade floors waterfront view` there by making many more features. So, its not easy to type all the features manually and we may want to upload the test data to this website and get the prediction. The "Main" part of the app has the option for sample data and we can edit it to fit our purpose of uploading the new data.

# Usage
This app has multiple drop down menus. The first menu is the Homepage, where we can upload the test data and get the house price prediction. We may first download the sample test data and change it so that we can get prediction on new data. Then this page shows the prediction and other model evaluation metric.

The other tab is "Pandas Profile", in this page we can see the various statisical analysis of raw data. This raw data does not include synthesized features during feature engineering.

The other tab is "Sweetviz Profile", in this page we see the report similar to pandas profile. It also gives statistical analysis of raw data.