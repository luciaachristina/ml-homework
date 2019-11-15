import os
import pandas as pd
import numpy as np


#ADVANCED PART


from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=1)


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(x_train, y_train)

#for the purpose of this assignment, n_neighbors is =5, however, tested it out with others :)

predicted_values = neigh.predict(x_test)


from sklearn import metrics
print(f"Printing MAE error: {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")

#print(neigh.predict(x))

