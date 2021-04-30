# Energy-Consumption-Prediction

### Description
This project used a Kaggle dataset of AEP’s eastern grid’s hourly energy consumption from December 2004 – December 2018. This was combined with a dataset containing hourly temperature information. The algorithm is able to predict the energy consumption of any given hour with an R^2 of 0.77.

### Methods
After examining a map of AEP’s eastern grid, and seeing which cities were available in the temperature dataset, Indianapolis was elected to be used as the baseline temperature. The temperature dataset only contained info from 2013 – 2017, so this was the time frame used.
The dataset was sliced into 5 different features:  <br /> <br />
  ⦁	  Hour of the Day  <br />
  ⦁ 	Day of the Week  <br />
  ⦁	  Month  <br />
  ⦁	  Year  <br />
  ⦁	  Temperature  <br /> <br />
Initially a linear regression model was attempted, but it then became clear most of these features had “bends” and couldn’t be accurately modeled linearly. I found a python library pyGAM that allows one to build custom models based on different qualities of the features. In this case a spline regression general additive model was used for our features which does an excellent job of accounting for small twists in the data. This essentially applies a different function to each feature, rather than a linear or polynomial slope. A random 80% of the data was used to train the model, and the other 20% was used to test the model’s predictive power <br /> <br />

### Sources

The initial data was pulled from Rob Mulla’s Kaggle (kaggle.com/robikscube) as well as David Beniaguev’s page (www.kaggle.com/selfishgene). Also, pyGAM has done extensive work with generalized additive models, which you can read more about here (https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html). 




