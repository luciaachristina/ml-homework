import os
import pandas as pd
import numpy as np

#BASIC PART


from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
columns_names = diabetes.feature_names
x = diabetes.data
y = diabetes.target

print(f'sklearn dataset x shape: {x.shape}')
print(f'sklearn dataset y shape: {y.shape}')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, columns_names)}")


predicted_values = lm.predict(x_test)


for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="seismic")

# Plotting differenct between real and predicted values
sns.scatterplot(y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

# Plotting the residuals: the error between the real and predicted values
residuals = y_test - predicted_values
sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=30, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution')
plt.show()

from sklearn import metrics
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")