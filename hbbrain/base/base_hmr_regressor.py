import numpy as np 
from hbbrain.utils.membership_calc import membership_func_gfmm
from hbbrain.constants import EPSILON_MISSING_VAL
from hbbrain.base.base_regressor import BaseHyperboxRegressor

def is_contain_missing_value(X):
    if np.isnan(X).sum() > 0:
        return True
    else:
        return False
    
def convert_format_missing_input_zero_one(Xl, Xu, yl=None, yu=None, yl_rep=None, yu_rep=None):
    Xl_out = np.where(np.isnan(Xl), 1 + EPSILON_MISSING_VAL, Xl)
    Xu_out = np.where(np.isnan(Xu), -EPSILON_MISSING_VAL, Xu)
    
    if yl is not None and yu is not None:
        yl_out = np.where(np.isnan(yl), yl_rep, yl)
        yu_out = np.where(np.isnan(yu), yu_rep, yu)
    else:
        yl_out = None
        yu_out = None
        
    return Xl_out, Xu_out, yl_out, yu_out

class BaseHMRegressor (BaseHyperboxRegressor):
    def __init__(self, theta = 0.5, _lambda = 1, V = None, W = None, d1 = None, d2 = None,r = None):
        self._lambda = _lambda 
        super().__init__( theta, V, W, d1, d2, r)

    def predict(self, X):
        X = np.array(X)
        return self._predict(X,X)

    def _predict(self, Xl, Xu):
        Xl = np.array(Xl) # (n_samples, n_features)
        Xu = np.array(Xu)
        is_certain = not np.any(Xl != Xu)

        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)
        
        is_exist_missing_value = (self.V > self.W).any()

        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
            Xu = Xu.reshape(1, -1)

        n_samples = Xl.shape[0]
        y_pred = np.zeros((n_samples, 2))

        for i in range(n_samples):
            # Task 1: hyperbox Min max clustering -> membership value
            if is_exist_missing_value:
                membership_val = membership_func_gfmm(Xl[i,:], Xu[i,:], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self._lambda) # (n_hyperboxes)
            else:
                membership_val = membership_func_gfmm(Xl[i,:], Xu[i,:], self.V, self.W, self._lambda) # (n_hyperboxes)

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
