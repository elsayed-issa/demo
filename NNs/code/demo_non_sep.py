import numpy as np
from sklearn.linear_model import Perceptron

x = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([0,1,1,0])
p = Perceptron(tol=1e-3)
p.fit(x,y)

print(f"actual output: {p.predict(x)}")

#Linearly non-separated output: [0 1 1 0]