#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read 
df = pd.read_csv('dental_websites.csv')
#print(df.describe())

# Select Features
df_features = df[['Speed','Responsiveness','Age']]
# print(df_features.head())

# Split
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]

# Model
lm = LinearRegression()
train_x = np.asanyarray(train[['Speed', 'Age']])
train_y = np.asanyarray(train[['Responsiveness']])
results_formula=lm.fit(train_x,train_y)

# Predict
test_x = np.asanyarray(test[['Speed','Age']])
test_y = np.asanyarray(test[['Responsiveness']])
y_pred = lm.predict(test_x)
#y_pred = lm.predict([[1,11]])
#print(y_pred)

# Plot
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(121,projection='3d')
ax.scatter(df[['Speed']],df[['Age']],df[['Responsiveness']],edgecolor='k', s=150, c=df[['Responsiveness']])
ax.set_xlabel('Speed')
ax.set_zlabel('Responsiveness')
ax.set_ylabel('Age')
ax.title.set_text('Original Data')

ax = fig.add_subplot(122, projection='3d')
ax.scatter(test[['Speed']],test[['Age']],test_y, color='red', s=150)
ax.scatter(test[['Speed']],test[['Age']],y_pred, color='green', s=150)

x_plane, y_plane = np.meshgrid(np.linspace(df.Speed.min(), df.Speed.max(), 100),np.linspace(df.Age.min(), df.Age.max(), 100))
only_x = pd.DataFrame({'Speed': x_plane.ravel(), 'Age': y_plane.ravel()})
fitted_y = results_formula.predict(only_x)
fitted_y = np.array(fitted_y)
ax.plot_surface(x_plane,y_plane,fitted_y.reshape(x_plane.shape), color='pink', alpha=0.5)
ax.set_xlabel('Speed')
ax.set_zlabel('Responsiveness')
ax.set_ylabel('Age')
ax.title.set_text('Test & Prediction')

plt.savefig('3var_prediction.png')

# Regression Metrics
print('Coefficient: ', lm.coef_,  '\nIntercept: ', lm.intercept_)
print("MSE: %.2f" % mean_squared_error(test_y, y_pred))
print("R-squared: %.2f" % r2_score(test_y, y_pred))
