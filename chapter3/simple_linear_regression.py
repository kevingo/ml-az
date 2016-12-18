# data pre-processing

# Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# import data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # independent variable
Y = dataset.iloc[:, 1].values   # dependent variable

# Split data into training / testing dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print X_train"""

# Fitting simple linear regression model to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the test set results
Y_predict = regressor.predict(X_test)

# Visulization the Train Set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title(' Salary V.S. Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visulization the Test Set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title(' Salary V.S. Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


