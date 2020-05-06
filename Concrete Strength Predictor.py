#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
Dataset = pd.read_csv("concrete.csv")

X = Dataset.iloc[:,:-1]
y = Dataset.iloc[: ,-1]

#So here introduce the polynomial features of degree 3.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X = poly_reg.fit_transform(X)

#Feature scaling is not needed here. But its always a good practice to feature scale.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train , X_test, y_train , y_test = train_test_split(X, y, test_size = 0.2 , random_state = 15)

#Importing the model(using SGD)
from sklearn.linear_model import SGDRegressor
regressor = SGDRegressor(max_iter=1000, tol=1e-3, alpha =0.001, random_state = 0, learning_rate = 'constant' , eta0 = 0.001)
regressor.fit(X_train, y_train)

#Predicting our results with the test set
y_pred = regressor.predict(X_test)

#Now lets calculate the Coefficient of Determination and the RMSE for our training set
from sklearn.metrics import r2_score , mean_squared_error

rmse_train = (np.sqrt(mean_squared_error(y_train, regressor.predict(X_train) )))
r_squared_train = r2_score(y_train , regressor.predict(X_train))
print("R squared for the training set")
print("---------------------------------")
print(r_squared_train)
print("---------------------------------")
print("RMSEfor the training set")
print("---------------------------------")
print(rmse_train)
print()
#Now lets calculate the Coefficient of Determination and the RMSE for our training set
rmse_test = (np.sqrt(mean_squared_error(y_test, regressor.predict(X_test) )))
r_squared_test = r2_score(y_test , regressor.predict(X_test))
print("R squared for the testing set")
print("---------------------------------")
print(r_squared_test)
print("---------------------------------")
print("RMSEfor the testing set")
print("---------------------------------")
print(rmse_test)
