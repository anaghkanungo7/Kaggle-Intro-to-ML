# Basic data exploration
# Author: Anagh Kanungo

import pandas as pd

# Read in file
file = './melb_data.csv'
data = pd.read_csv(file)

# Describe standard statistics of the dataset
# print(data.describe())

# Show columns of dataset
# print(data.columns)

# Drop rows which have missing values
data = data.dropna(axis=0)

# Get prediction target (column we want to predict)
y = data.Price

# Features - columns inputted into our model and used to make predictions
# Column names must match those of dataset
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]
# print(X.describe())
# print(X.head())
