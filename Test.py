from LinearRegression import RegressionLineaire
import numpy as np


X = np.random.rand(50,5)
E = np.random.rand(50,1)
T = np.random.rand(5,1)

Y = X.dot(T) + E*0.1

print(T)

model1 = RegressionLineaire()
model1.fit(X, Y)
model1.evaluate(X, Y)

model2 = RegressionLineaire(gradient_descent=True)
model2.fit(X, Y)
model2.evaluate(X, Y)

model3 = RegressionLineaire(gradient_descent=True, stochastic =True)
model3.fit(X, Y)
model3.evaluate(X, Y)

from sklearn.model_selection import train_test_split
X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model3 = RegressionLineaire(gradient_descent=True, stochastic =True)
model3.fit(X, Y, X_val = X_val, Y_val = Y_val)
model3.evaluate(X, Y)