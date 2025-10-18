# Import necessary library
import numpy as np
import pandas as pd
import time
import random
import itertools
from hbbrain.base.base_uhmr_regressor import BaseUHMRegressor, is_contain_missing_value
from hbbrain.utils.unbounded_membership_calc import unbounded_gfmm_membership_func1, unbounded_gfmm_membership_func2, unbounded_gfmm_membership_func4, unbounded_gfmm_membership_func3, impute_missing_val_x

# UHMR Learning process
class UHMR_learner(BaseUHMRegressor):
    def __init__ (self, theta = 0.5, alpha = 1, V = None, W= None, d1 = None , d2 = None, r = None,  N_samples=None, membership_func = unbounded_gfmm_membership_func1, lambda_reg = 1e-4):
        super().__init__( theta, alpha, V, W, d1, d2, r, membership_func)
        self.N_samples = N_samples if N_samples is not None else np.array([])
        self.lambda_reg = lambda_reg

    def fit (self, X, y):
        n_samples = X.shape[0]
        if X.shape[0] > n_samples:
            Xl = X[:n_samples, :]
            Xu = X[n_samples:, :]
            yl = y[:n_samples]
            yu = y[n_samples:]
            return self._fit(Xl, Xu, yl, yu)
        return self._fit(X,X,y,y)

    def _fit (self, Xl, Xu, yl, yu):
        # Start
        self._init_hyperboxes()
        is_certain = not np.any(Xl != Xu)

        if is_contain_missing_value(yl) == True:
            yl = np.where(np.isnan(yl), np.nanmean(yl), yl)
        
        if is_contain_missing_value(yu) == True:
            yu = np.where(np.isnan(yu), np.nanmean(yu), yu)

        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            is_exist_missing_value = True
            Xl, Xu = impute_missing_val_x(Xl, Xu)
        else:
            is_exist_missing_value = False

        time_start = time.perf_counter()
        # Step 1: Hyperbox clustering
        n_samples, n_features = Xl.shape
        for i in range (n_samples):
            if (self.W.size == 0 or self.V.size ==0):
                self.W = np.array([Xu[i,:]])
                self.V = np.array([Xl[i,:]])
                self.N_samples = np.array([1])
                continue

            # Check top-k member ship function ( all existing hyperbox )
            if is_exist_missing_value:
                membership_val = self.membership_func(Xl[i,:], Xu[i,:],np.minimum(self.V, self.W),np.maximum(self.W, self.V), self.alpha)
            else:
                membership_val = self.membership_func(Xl[i,:], Xu[i,:],self.V,self.W, self.alpha)

            best = np.argmax(membership_val)
            if membership_val[best] >= self.theta:
                # expand the best hyperbox
                self.V[best] = np.minimum(self.V[best], Xl[i, :])
                self.W[best] = np.maximum(self.W[best], Xu[i, :])
                self.N_samples[best] += 1
            else:
                # create new hyperbox
                self.V = np.concatenate((self.V, Xl[i, :].reshape(1, -1)), axis=0)
                self.W = np.concatenate((self.W, Xu[i, :].reshape(1, -1)), axis=0)
                self.N_samples = np.append(self.N_samples, 1)

        # Step 2: Least Square Optimization with OLS
        A = []
        n_hyperboxes = self.V.shape[0]
        if not is_certain:
            dim_w = 2* n_features + 1
            self.r = np.zeros((n_hyperboxes, 2))
            self.d1 = np.zeros((n_hyperboxes, 2 * n_features))
            self.d2 = np.zeros((n_hyperboxes, 2* n_features))
        else:
            dim_w = n_features + 1
            self.r = np.zeros(n_hyperboxes)
            self.d1 = np.zeros((n_hyperboxes, n_features))
            self.d2 = np.zeros((n_hyperboxes, n_features))

        # Second pass for LSO
        for i in range(n_samples):
            if is_exist_missing_value:
                membership_val_new = self.membership_func(Xl[i,:], Xu[i,:],np.minimum(self.V, self.W),np.maximum(self.W, self.V), self.alpha)
            else:
                membership_val_new = self.membership_func(Xl[i,:], Xu[i,:], self.V, self.W, self.alpha)
            # Normalize
            membership_val_norm = membership_val_new / (membership_val_new.sum() + 1e-15)

            # Form X
            if not is_certain:
                X = np.hstack((Xl[i,:], Xu[i,:]))
            else:
                X = Xu[i,:]

            # Turn 3D -> 2D arr for matrix calculation
            A_sample = membership_val_norm.reshape(-1,1) * np.append(X, 1).reshape (1,-1)
            A.append(A_sample.flatten())

        A = np.array(A).reshape(n_samples, -1)

        # Regularization
        reg_mask = np.ones(A.shape[1], dtype=bool)
        for i in range(n_hyperboxes):
            reg_mask[i * dim_w + dim_w - 1] = False  # skip bias term of each hyperbox
        I = np.diag(reg_mask.astype(float))

        Yl = yl.reshape(-1, 1)
        Yu = yu.reshape(-1, 1)

        if is_certain:
            D = np.linalg.solve(A.T @ A + self.lambda_reg * I , A.T @ Yl).ravel()
            D = D.flatten()
            # Extract tunned paremeters for regressor
            for i in range(n_hyperboxes):
                base = i * dim_w
                self.d1[i] = self.d2[i] = D[base: base + n_features]
                self.r[i]= D[base + n_features]
        else:
            # Handling uncertaincy utilsising 2 regressors / hyperbox
            A_pinv = np.linalg.solve(A.T @ A + self.lambda_reg * I, A.T)
            D1 = (A_pinv @ Yl).ravel()
            D2 = (A_pinv @ Yu).ravel()
            for i in range(n_hyperboxes):
                base = i * dim_w
                self.d1[i] = D1[base: base + dim_w - 1]
                self.d2[i] = D2[base: base + dim_w - 1]
                self.r[i,0]= D1[base + dim_w - 1]
                self.r[i,1]= D2[base + dim_w - 1]

        time_end = time.perf_counter()
        self.elapsed_time = time_end - time_start
        return self 