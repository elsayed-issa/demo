import numpy as np
from perc import Perceptron

### Applying Some function ###
# inputs
x = np.array([3,2])
# weights
w = np.array([0.5, -1.0])
# bias
b = 3

p = Perceptron(x,w,b)
p.fit_some_function()
# 10.5
