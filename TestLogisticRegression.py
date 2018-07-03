from LogisticRegression import LogisticRegression
import numpy as np

u = np.array([[0.5,0,-0.3,-0.7, 0.5]])
v = np.array([[0.7,-0.4,0.9,-0.5,0.6]])

Y = np.random.randint(0,2,(100,1))
R = np.tile(Y,(1,5))
E = np.random.rand(100,1)

X = u*R + (np.ones((100,5)) - R)*v + E

model1 = LogisticRegression()
model1.fit(X, Y)
print(model1.evaluate(X, Y))

model2 = LogisticRegression(stochastic =True, batch_size = 50)
model2.fit(X, Y)
print(model2.evaluate(X, Y))

from sklearn.model_selection import train_test_split
X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model3 = LogisticRegression(stochastic =True, batch_size = 50, epochs = 10000)
model3.fit(X, Y, X_val = X_val, Y_val = Y_val)
print(model3.evaluate(X, Y))