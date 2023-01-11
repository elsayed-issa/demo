import numpy as np
from sklearn.linear_model import LinearRegression

# X.shape (2,4)
X = np.array([[1,1],[1,0],[0,1],[0,0]])
# Y.shape (4,1)
Y = np.array([[0],[1],[1],[0]])
# Create a LinearRegression model
model = LinearRegression()
# Fit the model to the data
model.fit(X, Y)
# Get the slope and intercept of the fitted model
slope = model.coef_[0]
# 0.0
intercept = model.intercept_
# 0.5