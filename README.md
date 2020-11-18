# House Price Prediction using Heroku
This is the storage repo for my heroku app [house price prediction](https://deploy-house-price.herokuapp.com/).
The heroku website takes the data from this github repo and deploys the machine learning model. In this app I have used the house price data from [Kaggle competition:  House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction). I splitted the data into train and test and trained catboost model in the training data with comprehensive feature engineering. The dataset is from May 2014 to May 2015 and includes the prices of house in King County, Wahshigton State, USA. To estimate the house price we can upload a csv file (sample provied in the app) and run the app.

# About Author
```
Author: Bhishan Poudel, Ph.D. Physics
Website: https://bhishanpdl.github.io/
```

# About Heroku
For free accounts Heroku provides 550 dyno hours. One time visits counts to 30 minute (1/2 hours) of dyno hours. If we made some changes in github repo, we need to go to heroku website and deploy the branch again. If we no longer need to run an app we can do this `Dashboard => Your App Name => Resources => Pencil icon=> Flip the switch => Confirm`.
