import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from statistics import mean
import datetime
from dateutil.parser import parse
from pygam import LinearGAM, s, f


## import the two datasources, seperate Indianapolis data, and merge the dataframes
## both were downloaded from kaggle dot come

consumption = pd.read_csv('AEP_hourly.csv')

weather = pd.read_csv('Temperature.csv')

weather_indiana = weather[['Date_time_str', 'Indianapolis']]

print(weather_indiana.head(10))

df = pd.merge(consumption, weather_indiana, on='Date_time_str')

## convert degrees from Kelvin to Farenheit

df['Farenheit'] = df.Indianapolis.apply(lambda x: (x * 1.8) - 459.67)

## assigning variables based on the timestamp

df['Date_time'] = df.Date_time_str.apply(lambda x: parse(x))

df['Year'] = df.Date_time.apply(lambda x: x.year)

df['Month'] = df.Date_time.apply(lambda x: x.month)

df['Day_of_month'] = df.Date_time.apply(lambda x: x.day)

df['Day_of_week'] = df.Date_time.apply(lambda x: x.isoweekday())

df['Hour'] = df.Date_time.apply(lambda x: x.hour)

print(df.head(30))

# splitting the data into the 2013 - 2018 span that includes weather data

df2 = df[(df.Year == 2013) | (df.Year == 2014) | (df.Year == 2015) | (df.Year == 2016) | (df.Year == 2017)]

df3 = df2.dropna()

x = df3[['Farenheit', 'Year', 'Month', 'Day_of_week', 'Hour']]

y = df3['AEP_MW']

# splitting the data into the 80% used to train the model and the 20% used to test the model

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)

# passing the data into the model
gam = LinearGAM()
gam.fit(x_train, y_train)

y_predicted = gam.predict(x_test)

gam.summary()
# building out the axis labels for the graphs of the different features
months = range(13)
month_names = [" ", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep","Oct", "Nov", "Dec"]

days = range(8)
day_names = ['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

hours = [0, 0, 3, 6, 9, 12, 15, 18, 21, 24]
hour_times = ['', '12AM', '3AM', '6AM', '9AM', '12PM', '3PM', '6PM', '9PM', '12AM']

plt.figure();
fig, axs = plt.subplots(1,1)

titles = ['Temperature', 'Year', 'Month', 'Day of the Week', 'Hour of the Day']

#plotting the predictive results 

plt.scatter(y_test, y_predicted, alpha = .15)
plt.xlabel('Actual MW consumed')
plt.ylabel('Predicted MW consumed')
plt.title('Actual vs Predicted Hourly Energy Consumption')
plt.show()







