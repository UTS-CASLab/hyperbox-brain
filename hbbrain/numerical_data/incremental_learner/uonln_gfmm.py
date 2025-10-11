import argparse
import os
import pandas as pd
import numpy as np
import time
import itertools
from sklearn.metrics import accuracy_score

from hbbrain.base.base_gfmm_estimator import (
    BaseGFMMClassifier,
    convert_format_missing_input_zero_one,
    is_contain_missing_value,
    predict_with_manhattan,
)
from hbbrain.utils.unbounded_membership_calc import unbounded_gfmm_membership_func1, unbounded_gfmm_membership_func2, unbounded_gfmm_membership_func4, unbounded_gfmm_membership_func3, impute_missing_val_x
from hbbrain.utils.adjust_hyperbox import overlap_resolving_num_data, is_two_hyperboxes_overlap_num_data_general
from hbbrain.utils.membership_calc import membership_func_gfmm, get_membership_gfmm_all_classes
from hbbrain.utils.dist_metrics import manhattan_distance, manhattan_distance_with_missing_val
from hbbrain.constants import UNLABELED_CLASS, MARKER_LIST

class UOnlineGFMM(BaseGFMMClassifier):
    def __init__ (self, theta = 0.5, alpha = 1, gamma = 1, V = None, W = None, C = None, N_samples=None, membership_func = unbounded_gfmm_membership_func1):
        BaseGFMMClassifier.__init__(self, theta=theta, gamma=gamma, is_draw= False, V=V, W=W, C=C)
        self.alpha = alpha
        self.N_samples = N_samples if N_samples is not None else np.array([])
        self.membership_func = membership_func

    def fit (self, X,y):
        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        y = y.astype('int')
        n_samples = len(y)
        
        if X.shape[0] > n_samples:
            Xl = X[:n_samples, :]
            Xu = X[n_samples:, :]
            return self._fit(Xl, Xu, y)
        else:
            return self._fit(X, X, y)


    def _fit (self, Xl, Xu, y):
        self._init_hyperboxes()
        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)
        y = y.astype('int')

        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            is_exist_missing_value = True
            Xl, Xu = impute_missing_val_x(Xl, Xu)
        else:
            is_exist_missing_value = False

        time_start = time.perf_counter()
        n_samples, n_features = Xl.shape

        for i in range (n_samples):
            id_of_winner_hyperbox = None
            is_expanded = False

            if (self.W.size == 0 or self.V.size ==0):
                self.V = np.array([Xl[i,:]])
                self.W = np.array([Xu[i,:]])
                self.C = np.array([y[i]])
                self.N_samples = np.array([1])
                continue

            else: 
                if y[i] == UNLABELED_CLASS:
                    id_same_input_label_group = np.ones(len(self.C), dtype=bool)
                else:
                    id_same_input_label_group = (self.C == y[i]) | (
                        self.C == UNLABELED_CLASS)
                # Only consider same labelled or unlabelled hyperboxes for hyperbox expansion 
                if id_same_input_label_group.any() == True:
                    V_sameX = self.V[id_same_input_label_group]
                    W_sameX = self.W[id_same_input_label_group]
                    lb_sameX = self.C[id_same_input_label_group]
                    id_range = np.arange(len(self.C))
                    id_processing = id_range[id_same_input_label_group]

                    if is_exist_missing_value:
                        membership_val = self.membership_func(Xl[i,:], Xu[i,:], np.minimum(V_sameX, W_sameX), np.maximum(V_sameX, W_sameX), alpha = self.alpha)
                    else:
                        membership_val = self.membership_func(Xl[i,:], Xu[i,:], V_sameX, W_sameX, alpha = self.alpha)
                        
                    # Only get the highest membership hyperbox 
                    highest_local_idx = np.argsort(membership_val)[-1]
                    highest_idx = id_processing[highest_local_idx]


                    if membership_val[highest_local_idx] == 1:
                        if self.C[highest_idx] == UNLABELED_CLASS and y[i] != UNLABELED_CLASS:
                            self.C[highest_idx] = y[i]
                        self.N_samples[highest_idx] += 1
                        id_of_winner_hyperbox = highest_idx
                        is_expanded = True
                        continue

                    if membership_val[highest_local_idx] >= self.theta:
                        self.W[highest_idx] = np.maximum(self.W[highest_idx], Xu[i,:])
                        self.V[highest_idx] = np.minimum (self.V[highest_idx], Xl[i,:])
                        if self.C[highest_idx] == UNLABELED_CLASS and y[i] != UNLABELED_CLASS:
                            self.C[highest_idx] = y[i]
                        id_of_winner_hyperbox = highest_idx
                        is_expanded = True
                        self.N_samples[highest_idx] += 1

                        # Overlap check and contraction
                        n_existed_hyperboxes = self.V.shape[0]
                        id_diff_input_label_group = np.logical_not(id_same_input_label_group)
    
                        if id_diff_input_label_group.any() == True:
                            diff_indices = np.where(id_diff_input_label_group)[0]
                            for ii in diff_indices:
                                # overlap test
                                is_overlap = is_two_hyperboxes_overlap_num_data_general(
                                    self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii])

                                # Contraction
                                if is_overlap == True:
                                    self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii] = overlap_resolving_num_data(
                                        self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.C[id_of_winner_hyperbox], self.V[ii], self.W[ii], self.C[ii])

                # Create new hyperbox
                if not is_expanded:
                    self.V = np.concatenate((self.V, Xl[i,:].reshape(1,-1)), axis = 0)
                    self.W = np.concatenate((self.W, Xu[i,:].reshape(1,-1)), axis = 0)
                    self.C = np.concatenate((self.C, [y[i]]))
                    self.N_samples = np.append(self.N_samples, 1)


        time_end = time.perf_counter()
        self.elapsed_time = time_end - time_start
        return self 

    def predict (self, X):
        X = np.array(X)
        return self._predict(X, X)

    def _predict (self, Xl, Xu):
        Xl = np.array(Xl)
        Xu = np.array(Xu)
        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
            Xu = Xu.reshape(1, -1)

        n_samples = Xl.shape[0]
        y_pred = np.zeros((n_samples))

        # classifications
        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            is_exist_missing_value = True
            Xl, Xu = impute_missing_val_x(Xl, Xu)
        else:
            is_exist_missing_value = False

        for i in range(n_samples):
            if is_exist_missing_value:
                membership_val = self.membership_func(Xl[i,:], Xu[i,:], np.minimum(self.V, self.W), np.maximum(self.V,self.W) , alpha = self.alpha)
            else:
                membership_val = self.membership_func(Xl[i,:], Xu[i,:], self.V, self.W, alpha = self.alpha)

            bmax = membership_val.max() # get the maximum membership value
            max_mem_V_id = np.nonzero(membership_val == bmax)[0] # get indices of all hyperboxes with the maximum membership values
            if len(np.unique(self.C[max_mem_V_id])) > 1:
                if ((Xl[i] > Xu[i]).any() == True) or ((self.V[max_mem_V_id] > self.W[max_mem_V_id]).any() == True):
                    maht_dist = manhattan_distance_with_missing_val(Xl[i], Xu[i], self.V[max_mem_V_id], self.W[max_mem_V_id])
                else:
                    if (Xl[i] == Xu[i]).all() == False:
                        Xl_mat = np.ones((len(max_mem_V_id), 1)) * Xl[i]
                        Xu_mat = np.ones((len(max_mem_V_id), 1)) * Xu[i]
                        Xg_mat = (Xl_mat + Xu_mat) / 2
                    else:
                        Xg_mat = np.ones((len(max_mem_V_id), 1)) * Xl[i]
                    # Find all average points of all hyperboxes with the same membership value
                    avg_point_mat = (self.V[max_mem_V_id] + self.W[max_mem_V_id]) / 2
                    # compute the Manhattan distance from Xg_mat to all average points of all hyperboxes with the same membership value
                    maht_dist = manhattan_distance(avg_point_mat, Xg_mat)

                id_min_dist = maht_dist.argmin()
                y_pred[i] = self.C[max_mem_V_id[id_min_dist]]
            else:
                y_pred[i] = self.C[max_mem_V_id[0]]

        return y_pred