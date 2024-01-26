#!/usr/bin/env python
# coding: utf-8

# In[4]:
import warnings
warnings.filterwarnings("ignore")
# Import libraries
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from getpass import getpass



# In[15]:

#Security - access control to data
print("\nEnter your credentials")
username = input('Username: ')
password = getpass('Password: ')
if username != 'name' or password != 'pass':
    raise Exception("Invalid credentials")


# In[14]:

# Load the gold price data
df = pd.read_csv("monthly_csv.csv")
ad = pd.read_csv("annual_csv.csv")


# In[6]:

# Descriptive method - calculate summary statistics
print("\nSummary")
print(df.describe())


# In[9]:

# Non-descriptive method - Linear regression model
# Set date as datetime 
df['Date'] = pd.to_datetime(df['Date'])  

# Create features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month  

# Set target variable
df['Target'] = df['Price']

# Set column names
df.columns = ['Date', 'Price', 'Year', 'Month', 'Target']  

# Create X data
X = df[['Year', 'Month']]

# Split data into train and test
cutoff_date = '2020-06'

X_train = X[df['Date'] < cutoff_date]  
y_train = df[df['Date'] < cutoff_date]['Target'] 

X_test = X[df['Date'] >= cutoff_date]  
y_test = df[df['Date'] >= cutoff_date]['Target']

# Train model
model = LinearRegression() 
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Evaluate predictions
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE: {rmse}')
print()

# In[16]:


# Data visualization 1 - Line plot
plt.plot(df['Date'], df['Price'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Gold Price Over Time')
plt.savefig('LinePlot.png')
plt.clf() # Clear figure


# In[17]:


# Data visualization 2 - Histogram
plt.hist(df['Price'])
plt.xlabel('Price')
plt.ylabel('Frequency') 
plt.title('Distribution of Gold Prices')
plt.savefig('Histogram.png')
plt.clf() # Clear figure


# In[18]:


# Data visualization 3 - Heatmap 
# Pivot with months as index  
prices = df.pivot(index='Month', columns='Year', values='Price')

fig, ax = plt.subplots(figsize=(20, 10)) 
im = ax.imshow(prices, cmap='viridis')

# Month ticks on y-axis 
months = list(range(1, 13))  
ax.set_yticks(np.arange(len(months)))
ax.set_yticklabels(months)

# Year ticks along x-axis
years = list(range(min(df['Year']), max(df['Year'])+1))   
ax.set_xticks(np.arange(len(years)))
ax.set_xticklabels(years)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.set_xlabel('Year')
ax.set_ylabel('Month')
ax.set_title('Gold Prices')  

fig.tight_layout() 
plt.title('Gold Price Heatmap: Monthly Prices from 1950-2020')
plt.savefig('Heatmap.png') 
plt.clf() # Clear figure


# In[19]:

# Interactive queries
print("Lookup historical data:")
date = input("Enter a date to get gold price (YYYY-MM): ")

try:
    datetime.datetime.strptime(date, "%Y-%m")
except ValueError:
    print(f"Incorrect date format entered: {date}")  
else:
    try:
        price = df.loc[df['Date'] == date, 'Price'].values[0]
        print("Gold price at " + str(date) + " $" + str(price))
    except IndexError:
        print(f"No data for entered date: {date}")


# In[20]:


# Monitoring and maintenance
         
import sys
print("\nData Size Monitoring") 
data_size = sys.getsizeof(df)
print(f"Current dataframe size: {data_size} bytes") 
if sys.getsizeof(df) > 1e6:
    print('Dataframe size exceeded threshold, truncating')
    df = df.iloc[:500]


# In[21]:

# Evaluate model accuracy
print("\nModel Accuracy:")
actual_values = y_test[:2] 
predictions = model.predict(X_test)[:2]
error = metrics.mean_absolute_error(actual_values, predictions)
print(f"Average absolute error on test set: {error} \n")

# Make future gold price predictions  
print("Gold Price Forecasts:")

future_date = input("Enter future date (YYYY-MM): ")

year, month = future_date.split("-")
month = int(month)
year = int(year)  

future_month_year = [[year, month]]

prediction = model.predict(future_month_year)
print(f"Predicted gold price for {future_date}: ${prediction[0]:.2f}")
