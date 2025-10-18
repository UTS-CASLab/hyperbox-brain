import numpy as np 
from hbbrain.utils.unbounded_membership_calc import unbounded_gfmm_membership_func1, unbounded_gfmm_membership_func2, unbounded_gfmm_membership_func4, unbounded_gfmm_membership_func3, impute_missing_val_x
from hbbrain.base.base_regressor import BaseHyperboxRegressor

def is_contain_missing_value(X):
    if np.isnan(X).sum() > 0:
        return True
    else:
        return False

class BaseUHMRegressor (BaseHyperboxRegressor):
    def __init__(self, theta = 0.5, alpha = 1, V = None, W = None, d1 = None, d2 = None,r = None, membership_func = unbounded_gfmm_membership_func1):
        self.alpha = alpha 
        self.membership_func = membership_func
        super().__init__( theta, V, W, d1, d2, r)

    def predict(self, X):
        X = np.array(X)
        return self._predict(X,X)

    def _predict(self, Xl, Xu):
        Xl = np.array(Xl) # (n_samples, n_features)
        Xu = np.array(Xu)
        is_certain = not np.any(Xl != Xu)

        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            Xl, Xu = impute_missing_val_x(Xl, Xu)

        is_exist_missing_value = (self.V > self.W).any()

        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
            Xu = Xu.reshape(1, -1)

        n_samples = Xl.shape[0]
        y_pred = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # Task 1: hyperbox Min max clustering -> membership value
            if is_exist_missing_value:
                membership_val = self.membership_func(Xl[i,:], Xu[i,:], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.alpha) # (n_hyperboxes)
            else:
                membership_val = self.membership_func(Xl[i,:], Xu[i,:], self.V, self.W, self.alpha) # (n_hyperboxes)

            # Task 2: Normalization:
            membership_val = membership_val / (np.sum(membership_val) + 1e-15)

            # Task 3: Local Regressor:
            if is_certain:
                X = Xl[i,:]
                regressor_val = np.dot(Xl[i,:].reshape(1,-1), self.d1.T) + self.r.reshape(1,-1)
                y_pred[i,0] = y_pred[i,1] = np.dot(membership_val, regressor_val.ravel())
            else:
                X = np.hstack((Xl[i, :], Xu[i, :])) # (n_features x 2)
                regressor_val_lb = np.dot(X.reshape(1,-1), self.d1.T)  + self.r[:,0].reshape(1,-1)
                regressor_val_ub = np.dot (X.reshape(1,-1), self.d2.T) + self.r[:,1].reshape(1,-1) # (1, n_hyperboxes)
                y_pred_1 = np.dot(membership_val, np.minimum(regressor_val_lb, regressor_val_ub).ravel())
                y_pred_2 = np.dot (membership_val, np.maximum(regressor_val_lb, regressor_val_ub).ravel())
                y_pred[i,0] = y_pred_1
                y_pred[i,1] = y_pred_2

        if is_certain:
            return y_pred.mean(axis = 1)
        else:
            return y_pred
