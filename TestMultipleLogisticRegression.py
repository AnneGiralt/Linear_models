from MultipleLogisticRegression import SoftmaxRegression
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
X =iris.data
y = iris.target

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
Y =lb.fit_transform(y)

shuffled_index = np.random.permutation(range(0,X.shape[0]))
X = X[shuffled_index]
Y = Y[shuffled_index]

model1 = SoftmaxRegression(learning_step = 0.01,epochs = 1000)
model1.fit(X, Y)
print(model1.evaluate(X, Y))

model2 = SoftmaxRegression(stochastic = True, batch_size = 50)
model2.fit(X, Y)
print(model2.evaluate(X, Y))

from sklearn.model_selection import train_test_split
X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model3 = SoftmaxRegression(stochastic =True, batch_size = 50, epochs = 10000)
model3.fit(X, Y, X_val = X_val, Y_val = Y_val)
print(model3.evaluate(X, Y))