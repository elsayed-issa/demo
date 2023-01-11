import numpy as np

class Perceptron:
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b
    def some_function(self, i):
        return 2 * i
    def linear_function(self, i):
        return 1 if i > 0 else 0
    def perc(self):
        return (self.b + sum(self.x*self.w))
    def fit_some_function(self):
        return self.some_function(self.perc())
    def fit_linear_function(self):
        return self.linear_function(self.perc())