import numpy as np 

class BaseHyperboxRegressor: 
    def __init__(self, theta = 0.5, V = None, W = None, d1 = None, d2 = None,r = None):
        self.theta = theta
        if V is not None:
            self.V = V
        else:
            self.V = np.array([])
        if W is not None:
            self.W = W
        else:
            self.W = np.array([])
        if d1 is not None:
            self.d1 = d1
        else:
            self.d1 = np.array([])
        if d2 is not None:
            self.d2 = d2
        else:
            self.d2 = np.array([])
        if r is not None:
            self.r = r
        else:
            self.r = np.array([])

    def _init_hyperboxes(self):
        self.V = np.array([])
        self.W = np.array([])
    
    def fit (self, X,y):
        return self
    
    def get_n_hyperboxes(self):
        return self.V.shape[0]
        