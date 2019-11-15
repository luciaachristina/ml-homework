import os
import pandas as pd
import numpy as np


#AREACH PART


from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)


from sklearn.ensemble import GradientBoostingRegressor
gradient = GradientBoostingRegressor()
gradient.fit(x_train, y_train)


predicted_values = gradient.predict(x_test)

from sklearn import metrics
print(f"Printing RMSE error for GradiebtBoostingRegressor: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")



#why GradientBoost?
    #From my understanding, it takes on more decision trees (I'm assuming trees means decision trees) compared to the AdaBoost
    #which means it goes deeper to predict the 'proper' values. + given that it is a 'boost' regressor, it's given a 'boost' to
    #try and properly predict outcomes.
    #the diabetes dataset needs the boost, in my opinion, as it is not crystal clear which physical or physiological factors
    #come into play to predict diabetes progression. A boost certainly helps in isolating a little closer.