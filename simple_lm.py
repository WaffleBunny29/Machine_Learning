#!/usr/bin/env python3

# Import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Read
df = pd.read_csv('dental_websites.csv')

# Split
df_features= df[['Speed','Responsiveness']]
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]

# Train
lm = LinearRegression()
train_x = np.asanyarray(train[['Speed']])
train_y =np.asanyarray(train[['Responsiveness']])
train_line = lm.fit(train_x,train_y)

# Predict
test_x = np.asanyarray(test[['Speed']])
test_y = np.asanyarray(test[['Responsiveness']])
y_pred = lm.predict(test_x)

# Plot
plt.scatter(train_x, train_y, color='b')
plt.scatter(test_x, test_y,color='g')
plt.plot(test_x, y_pred,color='r')
plt.xlabel('Website Speed')
plt.ylabel('Responsiveness')
plt.savefig('predicted_model.png')

# Regression Metrics
print('Coefficient: ', lm.coef_,  '\nIntercept: ', lm.intercept_)
print("MSE: %.2f" % mean_squared_error(test_y, y_pred))
print("R-squared: %.2f" % r2_score(test_y, y_pred))
