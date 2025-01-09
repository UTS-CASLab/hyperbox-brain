"""
General fuzzy min-max neural network trained by the batch incremental
learning algorithm for mixed attribute data, in which categorical features are
encoded using one-hot encoding.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from hbbrain.base.base_estimator import BaseHyperboxClassifier
from hbbrain.base.base_gfmm_estimator import (
    convert_format_missing_input_zero_one,
    is_contain_missing_value,
)
from hbbrain.utils.membership_calc import (
    membership_func_onehot_gfmm,
    get_membership_onehot_gfmm_all_classes,
)
from hbbrain.utils.adjust_hyperbox import (
    overlap_resolving_num_data,
    is_two_hyperboxes_overlap_num_data_general
)
from hbbrain.utils.dist_metrics import (
    manhattan_distance,
    manhattan_distance_with_missing_val,
)
from hbbrain.constants import UNLABELED_CLASS, CAT_MISSING_FEATURE


def one_hot_encoding_cat_feature(X, categorical_features, encodings=None):
    """
    Encode categorical features by the one-hot encoding method.

    Note
    ----

    Each categorical feature is transformed into a list of boolean values, each
    contains only one value of True and the rest elements show False values.

    For examples:
            X = array([[4, 2, 'red', 4],\n
                       [5, 6, 'green', 6],\n
                       [5, 1, 'red', 10],\n
                       [6, 4, 'yellows', 2],\n
                       [5, 5, 'green', 7],\n
                       [9, 5, 'red', 12]], dtype=object)
            X_transformed = array([[4, 2, array([ True, False, False]), 4],\n
                                   [5, 6, array([False,  True, False]), 6],\n
                                   [5, 1, array([ True, False, False]), 10],\n
                                   [6, 4, array([False, False,  True]), 2],\n
                                   [5, 5, array([False,  True, False]), 7],\n
                                   [9, 5, array([ True, False, False]), 12]], dtype=object)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input patterns.
    categorical_features : array-like of shape (n_cat_features, )
        Indices of categorical features.
    encodings : a list of objects, optional, default=None
        Storing a list of one-hot encoders each for a categorical feature.

    Returns
    -------
    X_out : array-like of shape (n_samples, n_features)
        An input data matrix with the encoded categorical features.
    encodings_out : TYPE
        An one-hot encoder was used to encode categorical features.

    """
    X_out = X.copy()
    encodings_out = []
    for i, val in enumerate(categorical_features):
        if encodings is None:
            encoding = OneHotEncoder(handle_unknown='ignore')
            encoding.fit(X[:, [val]])
            encodings_out.append(encoding)
        else:
            encoding = encodings[i]

        oh_transformed = encoding.transform(X[:, [val]]).toarray()
        oh_transformed_reshape = [np.array(j, dtype=bool)
                                  for j in oh_transformed]
        X_out[:, val] = oh_transformed_reshape

    return X_out, encodings_out


def predict_onehot_cat_feature_manhanttan(V, W, D, C, Xl, Xu, Xd, g=1):
    """
    Predict class labels for mixed-class samples in `X` represented in the
    form of invervals `[Xl, Xu, Xd]`. This is a common function to determine
    the right class labels for `X` wrt a trained hyperbox-based classifier
    represented by `[V, W, D, C]`. It uses the winner-takes-all principle
    to predict class labels for each sample in `X` by assigning the class
    label of the sample to the class label of the hyperbox with the maximum
    membership value to that sample. It will use a Manhattan distance for
    continous features in the case of many hyperboxes with different classes
    having the same maximum membership value. If there is no continuous feature
    the random selection will be used for the case of many winner hyperboxes.

    Parameters
    ----------
    Xl : array-like of shape (n_samples, n_continuous_features)
        Lower bounds of continuous features of all input samples.
        If None, there are no continous features.
    Xu : array-like of shape (n_samples, n_continuous_features)
        Lower bounds of continuous features of all input samples.
        If None, there are no continous features.
    Xd : array-like of shape (n_samples, n_cat_features)
        Bounds of categorical features of all input patterns.
        If None, there are no categorical features.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        Minimum points of all continuous features of the existing hyperboxes
        in the trained model. If None, there are no continous features.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        Maximum points of all continuous features of the existing hyperboxes
        in the trained model. If None, there are no continous features.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        Bounds of all categorical features of the existing hyperboxes in the
        trained model. If None, there are no categorical features.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes corresponding to the values
        stored in `V`, `W`, and `D`.
    g : float or ndarray of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continous dimension.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        A vector contains the predictions. In binary and multiclass problems,
        this is a vector containing `n_samples`.

    """
    if Xl is not None:
        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
        if Xu.ndim == 1:
            Xu = Xu.reshape(1, -1)

        if is_contain_missing_value(Xl) == True or is_contain_missing_value(Xu) == True:
            Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

    if Xd is not None:
        if Xd.ndim == 1:
            if Xd.shape[0] > 1:
                Xd = Xd.reshape(1, -1)
            else:
                Xd_tmp = np.zeros((1, 1), dtype=np.object)
                Xd_tmp.fill(Xd)
                Xd = Xd_tmp

    if Xl is not None:
        n_samples = Xl.shape[0]
    else:
        n_samples = Xd.shape[0]

    if V is not None:
        is_exist_missing_value = (V > W).any()
    else:
        is_exist_missing_value = False

    y_pred = np.full(n_samples, 0)
    sample_id = 0
    np.random.seed(0)
    for i in range(n_samples):
        sample_id += 1

        if (Xl is not None) and (Xd is not None):
            # calculate memberships for all hyperboxes
            if not is_exist_missing_value:
                mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], Xd[i], V, W, D, g)
            else:
                mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], Xd[i], np.minimum(V, W), np.maximum(W, V), D, g)
        else:
            if Xl is not None:
                if not is_exist_missing_value:
                    mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], None, V, W, D, g)
                else:
                    mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], None, np.minimum(V, W), np.maximum(W, V), D, g)
            else:
                mem_vals = membership_func_onehot_gfmm(None, None, Xd[i], V, W, D, g)

        # get max membership value
        bmax = mem_vals.max()
        if (Xl is not None) and (((Xl[i] < 0).any() == True) or ((Xu[i] > 1).any() == True)):
            print(">>> The testing sample %d with the coordinate %s is outside the range [0, 1]. Membership value = %f. The prediction is more likely incorrect." %(sample_id, Xl[i], bmax))

        # get indices of all hyperboxes with max membership
        max_mem_box_ids = np.nonzero(mem_vals == bmax)[0]
        winner_cls = np.unique(C[max_mem_box_ids])

        if len(winner_cls) > 1:
            if Xl is None:
                y_pred[i] = np.random.choice(winner_cls, 1, False)[0]
            else:
                if ((Xl[i] > Xu[i]).any() == True) or ((V[max_mem_box_ids] > W[max_mem_box_ids]).any() == True):
                    maht_dist = manhattan_distance_with_missing_val(Xl[i], Xu[i], V[max_mem_box_ids], W[max_mem_box_ids])
                else:
                    if (Xl[i] == Xu[i]).all() == False:
                        Xl_mat = np.ones((len(max_mem_box_ids), 1)) * Xl[i]
                        Xu_mat = np.ones((len(max_mem_box_ids), 1)) * Xu[i]
                        Xg_mat = (Xl_mat + Xu_mat) / 2
                    else:
                        Xg_mat = np.ones((len(max_mem_box_ids), 1)) * Xl[i]
                    # Find all average points of all hyperboxes with the same
                    # membership value
                    avg_point_mat = (V[max_mem_box_ids] + W[max_mem_box_ids]) / 2
                    # compute the Manhattan distance from Xg_mat to all average
                    # points of all hyperboxes with the same membership value
                    maht_dist = manhattan_distance(avg_point_mat, Xg_mat)

                id_min_dist = maht_dist.argmin()
                y_pred[i] = C[max_mem_box_ids[id_min_dist]]
        else:
            y_pred[i] = C[max_mem_box_ids[0]]

    return y_pred


def impute_missing_value_cat_feature(Xd):
    """
    Impute missing values of categorical features in `Xd` by a constant value.

    Parameters
    ----------
    Xd : array-like of shape (n_samples, n_cat_features)
        Categorical features.

    Returns
    -------
    Xd : array-like of shape (n_samples, n_cat_features)
        Categorial features after doing data imputation.

    """
    Xd = np.where(pd.isna(Xd), CAT_MISSING_FEATURE, Xd)
    return Xd


class OneHotOnlineGFMM(BaseHyperboxClassifier):
    """Batch incremental learning algorithm with mixed-attribute data for a
    general fuzzy min-max neural network, in which categorical features are
    encoded using the one-hot encoding method and the similarity degrees among
    categorical values are computed using one-hot encoding values with logical
    operators. The final membership value is the average of membership values
    for continuous features and membership values for categorical features.

    See [1]_ for more detailed information regarding this batch incremental
    learning algorithm.

    Parameters
    ----------
    theta : float, optional, default=0.5
        Maximum hyperbox size for continuous features.
    theta_min : float, optional, default=1
        Minimum value of the maximum hyperbox size for continuous features so
        that the training loop is still performed. If the value of `theta_min`
        is larger than the value of `theta`, it will be automatically assigned
        a value equal to `theta`.
    gamma : float or ndarray of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous feature.
    min_percent_overlap_cat : float, optional, default=0.5
        The minimum number of categorical values in the categorical features of
        the input pattern that match the values in the categorical dimensions of
        the winner hyperbox to be expansion.
    alpha : float, optional, default=0.9
        Multiplier factor to reduce the value of maximum hyperbox size after
        each training loop.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all minimal points for continuous features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all maximal points for continuous features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all bounds for categorical features of all
        existing hyperboxes, in which each row is a lower bound of a hyperbox.
        Elements in this matrix are binary strings. 
    C : array-like of shape (n_hyperboxes,)
        A vector stores all class labels correponding to existing hyperboxes.

    Attributes
    ----------
    categorical_features_ : int array of shape (n_cat_features,)
        Indices of categorical features in the training data and hyperboxes.
    continuous_features_ : int array of shape (n_continuous_features,)
        Indices of continuous features in the training data and hyperboxes.
    encoder_ : sklearn.preprocessing.OneHotEncoder
        An one-hot encoder was used to encode categorical features.
    is_exist_continuous_missing_value : boolean
        Is there any missing values in continuous features in the training data.
    elapsed_training_time : float
        Training time in seconds.
    n_passes : int
        Number of training loops.

    References
    ----------
    .. [1] T. T. Khuat and B. Gabrys "An in-depth comparison of methods handling
           mixed-attribute data for general fuzzy min–max neural network",
           Neurocomputing, vol 464, pp. 175-202, 2021.

    Examples
    --------
    >>> from hbbrain.mixed_data.onehot_onln_gfmm import OneHotOnlineGFMM
    >>> from hbbrain.datasets import load_japanese_credit
    >>> X, y = load_japanese_credit()
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> numerical_features = [1, 2, 7, 10, 13, 14]
    >>> categorical_features = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    >>> scaler.fit(X[:, numerical_features])
    MinMaxScaler()
    >>> X[:, numerical_features] = scaler.transform(X[:, numerical_features])
    >>> clf = OneHotOnlineGFMM(theta=0.1, min_percent_overlap_cat=0.6)
    >>> clf.fit(X, y, categorical_features)
    >>> print("Number of hyperboxes = %d"%clf.get_n_hyperboxes())
    Number of hyperboxes = 236
    >>> clf.predict(X[[10, 100]])
    array([0, 0])

    """

    def __init__(self, theta=0.5, theta_min=1, min_percent_overlap_cat=0.5, gamma=1, alpha=0.9, V=None, W=None, D=None, C=None):
        BaseHyperboxClassifier.__init__(self, theta, False, V, W, C)
        if D is not None:
            self.D = D
        else:
            self.D = np.array([])
        self.gamma = gamma
        self.theta_min = theta_min
        self.alpha = alpha
        self.min_percent_overlap_cat = min_percent_overlap_cat

    def _validate_data(self):
        """
        Validate the initial values of parameters and initialise default values
        for parameters.

        Returns
        -------
        None.

        """
        if (self.theta_min > self.theta):
            self.theta_min = self.theta
        self._init_hyperboxes()
        if self.D is None:
            self.D = np.array([])

    def is_satisfied_cat_expansion_conds(self, xd, Dj, n_cat_features):
        """
        Check whether the expansion condition for categorical features `xd` of 
        an input pattern can be covered by categorical features of the hyperbox
        :math:`B_j` with the categorical features stored in `Dj`.

        Parameters
        ----------
        xd : array-like of shape (n_cat_features,)
            Categorical features of an input pattern.
        Dj : array-like of shape (n_cat_features,)
            Categorical features bounds of the hyperbox `Bj` which can be
            extended to cover the input pattern.
        n_cat_features : int
            Number of categorical features in the training set.

        Returns
        -------
        bool
            If True, the categorical features in `Dj` are satisfied with the
            expansion conditions for the categorical feature so that it can be
            expanded to cover the input pattern. Otherwise, the conditions for
            the categorical features are not met.

        """
        min_n_overlap_cat_features = max(int(self.min_percent_overlap_cat * n_cat_features), 1)
        and_res = np.bitwise_and(xd, Dj)
        count_bit_ones = [np.any(i) for i in and_res]

        if np.sum(count_bit_ones) >= min_n_overlap_cat_features:
            return True
        else:
            return False
        
    def fit(self, X, y, categorical_features=None):
        """
        Build a general fuzzy min-max neural network from the training set
        (X, y) using the original incremental learning algorithm for mixed
        attribute data, in which categorical features are encoded using one-hot
        encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (2*n_samples, n_features)
            The training input samples including both continuous and categorical
            features. If the number of rows in `X` is 2*n_samples, the first 
            n_samples rows contain lower bounds of input patterns and the rest
            n_samples rows contain upper bounds.
        y : array-like of shape (n_samples,)
            The class labels.
        categorical_features : a list of int, optional, default=None
            Indices of categorical features in the training set. If None, there
            is no categorical feature.

        Returns
        -------
        self : object.
            Fitted estimator.

        """
        self.categorical_features_ = categorical_features
        if X.ndim == 1:
            X = X.reshape(shape=(1, -1))

        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        y = y.astype('int')
        if categorical_features is not None:
            X[:, categorical_features] = impute_missing_value_cat_feature(X[:, categorical_features])
            X, self.encoders_ = one_hot_encoding_cat_feature(X, categorical_features)
            Xd = X[:, categorical_features]

        n_features = X.shape[1]
        if (categorical_features is None) or (len(categorical_features) < n_features):
            continuous_features = []
            for i in range(n_features):
                if i not in categorical_features:
                    continuous_features.append(i)
            self.continuous_features_ = continuous_features
            n_samples = len(y)
            X_con = X[:, continuous_features].astype(float)
            if X_con.shape[0] > n_samples:
                Xl = X_con[:n_samples, :]
                Xu = X_con[n_samples:, :]
                if categorical_features is None:
                    return self._fit(Xl, Xu, None, y)
                else:
                    Xd = Xd[:n_samples, :]
                    return self._fit(Xl, Xu, Xd, y)
            else:
                if categorical_features is None:
                    return self._fit(X_con, X_con, None, y)
                else:
                    return self._fit(X_con, X_con, Xd, y)
        else:
            self.continuous_features_ = None
            return self._fit(None, None, Xd, y)

    def _fit(self, Xl, Xu, Xd, y):
        """
        Build a general fuzzy min-max neural network from the training set
        (X, y) using the original incremental learning algorithm for mixed
        attribute data, in which categorical features are encoded using one-hot
        encoding. Input training data in this method were split into continuous
        features with lower and upper bounds and categorical features. Categorical
        values in `Xd` were encoded by an one-hot encoder.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_continuous_features)
            A matrix stores the lower bounds of training continuous features.
            If there is no continuous feature, this variable will get a None value.
        Xu : array-like of shape (n_samples, n_continuous_features)
            A matrix stores the upper bounds of training continuous features.
            If there is no continuous feature, this variable will get a None value.
        Xd : array-like of shape (n_samples, n_cat_features)
            Bounds of categorical features of all input patterns. Elements stored
            in this parameter need to be encoded by an one-hot encoder in
            the :func:one_hot_encoding_cat_feature. If None, there are no
            categorical features.
        y : array-like of shape (n_samples,)
            The class labels.

        Returns
        -------
        self : object
            The fitted estimator.

        """
        time_start = time.perf_counter()

        if Xl is not None:
            n_samples = Xl.shape[0]
            n_continuous_features = Xl.shape[1]
        else:
            n_samples = Xd.shape[0]
            n_continuous_features = 0
            
        if Xd is not None:
            if Xd.ndim == 1:
                Xd = Xd.reshape(-1, 1)
            n_cat_features = Xd.shape[1]
        else:
            n_cat_features = 0
        
        self._validate_data()

        self.is_exist_continuous_missing_value = False
        if Xl is not None:
            if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
                self.is_exist_continuous_missing_value = True
                Xl, Xu, y = convert_format_missing_input_zero_one(Xl, Xu, y)

        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        theta = self.theta
        training_acc = 0
        self.n_passes = 0

        while theta >= self.theta_min and training_acc < 1:
            self.n_passes += 1
            threshold_mem_val = 1 - np.max(self.gamma) * theta
            # Loop through each training input pattern
            for i in range(n_samples):
                if (n_continuous_features > 0 and self.V.size == 0) or (n_cat_features > 0 and self.D.size == 0):
                    # no model provided, start from scratch
                    if Xl is not None:
                        self.V = np.array([Xl[i]])
                        self.W = np.array([Xu[i]])
                    if Xd is not None:
                        self.D = Xd[i].reshape(1, -1)

                    self.C = np.array([y[i]])
                else:
                    if y[i] == UNLABELED_CLASS:
                        id_same_input_label_group = np.ones(len(self.C), dtype=bool)
                    else:
                        id_same_input_label_group = (self.C == y[i]) | (self.C == UNLABELED_CLASS)

                    if id_same_input_label_group.any() == True: 
                        if n_continuous_features > 0:
                            V_sameX = self.V[id_same_input_label_group]
                            W_sameX = self.W[id_same_input_label_group]
                        else:
                            V_sameX = None
                            W_sameX = None
                            
                        if n_cat_features > 0:
                            D_sameX = self.D[id_same_input_label_group]
                        else:
                            D_sameX = None
                            
                        lb_sameX = self.C[id_same_input_label_group]
                        id_range = np.arange(len(self.C))
                        id_processing = id_range[id_same_input_label_group]
                        
                        if n_continuous_features > 0 and n_cat_features > 0:
                            if self.is_exist_continuous_missing_value:
                                b = membership_func_onehot_gfmm(Xl[i], Xu[i], Xd[i], np.minimum(V_sameX, W_sameX), np.maximum(W_sameX, V_sameX), D_sameX, self.gamma)
                            else:
                                b = membership_func_onehot_gfmm(Xl[i], Xu[i], Xd[i], V_sameX, W_sameX, D_sameX, self.gamma)
                        else:
                            if n_continuous_features > 0:
                                if self.is_exist_continuous_missing_value:
                                    b = membership_func_onehot_gfmm(Xl[i], Xu[i], None, np.minimum(V_sameX, W_sameX), np.maximum(W_sameX, V_sameX), D_sameX, self.gamma)
                                else:
                                    b = membership_func_onehot_gfmm(Xl[i], Xu[i], None, V_sameX, W_sameX, D_sameX, self.gamma)
                            else:
                                b = membership_func_onehot_gfmm(None, None, Xd[i], V_sameX, W_sameX, D_sameX, self.gamma)

                        id_descending_mem_val = np.argsort(b)[::-1]
                    
                        if b[id_descending_mem_val[0]] != 1 or (y[i] != lb_sameX[id_descending_mem_val[0]] and y[i] != UNLABELED_CLASS):
                            adjust = False
                            count = 0
                            for j in id_processing[id_descending_mem_val]:
                                if n_cat_features == 0 and b[id_descending_mem_val[count]] < threshold_mem_val:
                                    break
                                
                                count += 1
                                # test violation of max hyperbox size and class labels
                                if (y[i] == self.C[j] or self.C[j] == UNLABELED_CLASS or y[i] == UNLABELED_CLASS):
                                    is_met_expansion = False
                                    if n_continuous_features > 0 and n_cat_features > 0:
                                        if (((np.maximum(self.W[j], Xu[i]) - np.minimum(self.V[j], Xl[i])) <= theta).all() == True) and (self.is_satisfied_cat_expansion_conds(Xd[i], self.D[j], n_cat_features) == True):
                                            is_met_expansion = True
                                    else:
                                        if (n_continuous_features > 0) and (((np.maximum(self.W[j], Xu[i]) - np.minimum(self.V[j], Xl[i])) <= theta).all() == True):
                                            is_met_expansion = True
                                        if (n_cat_features > 0) and (self.is_met_cat_expansion_conds(Xd[i], self.D[j], n_cat_features) == True):
                                            is_met_expansion = True
                                    
                                    if is_met_expansion == True:
                                        # adjust the j-th hyperbox
                                        if n_continuous_features > 0:
                                            self.V[j] = np.minimum(self.V[j], Xl[i])
                                            self.W[j] = np.maximum(self.W[j], Xu[i])
                                        if n_cat_features > 0:
                                            self.D[j] = np.bitwise_or(self.D[j], Xd[i])
                                        
                                        id_of_winner_hyperbox = j
                                        adjust = True
                                        if y[i] != UNLABELED_CLASS and self.C[j] == UNLABELED_CLASS:
                                            self.C[j] = y[i]
                                        # found out the winner hyperbox to adjust => break the loop
                                        break

                            # if i-th sample did not fit into any existing box, create a new one
                            if not adjust:
                                if n_continuous_features > 0:
                                    self.V = np.concatenate((self.V, Xl[i].reshape(1, -1)), axis = 0)
                                    self.W = np.concatenate((self.W, Xu[i].reshape(1, -1)), axis = 0)
                                if n_cat_features > 0:
                                    self.D = np.vstack((self.D, Xd[i]))
                                self.C = np.concatenate((self.C, [y[i]]))
                            else:
                                if (n_continuous_features > 0) and (self.V.shape[0] > 1):
                                    for ii in range(self.V.shape[0]):
                                        if (ii != id_of_winner_hyperbox) and (self.C[ii] != self.C[id_of_winner_hyperbox] or self.C[id_of_winner_hyperbox] == UNLABELED_CLASS):
                                            # overlap test
                                            is_overlap = is_two_hyperboxes_overlap_num_data_general(
                                                self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii])

                                            if is_overlap == True:
                                                self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii] = overlap_resolving_num_data(
                                                    self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.C[id_of_winner_hyperbox], self.V[ii], self.W[ii], self.C[ii])
                    else:
                        if n_continuous_features > 0:
                            self.V = np.concatenate((self.V, Xl[i].reshape(1, -1)), axis = 0)
                            self.W = np.concatenate((self.W, Xu[i].reshape(1, -1)), axis = 0)
                        if n_cat_features > 0:
                            self.D = np.vstack((self.D, Xd[i]))
                        self.C = np.concatenate((self.C, [y[i]]))
            
            if n_continuous_features > 0:
                theta = theta * self.alpha
                if theta >= self.theta_min:
                    y_pred = self._predict(Xl, Xu, Xd)
                    training_acc = accuracy_score(y, y_pred)
            else:
                training_acc = 2  # stop the loop

        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start

        return self

    def predict(self, X):
        """
        Predict class labels for samples in `X`.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i`, an additional criterion based on the
            minimum Manhattan distance between continous featurers of :math:`X_i`
            and the central points of continous features of winner hyperboxes
            are used to find the final winner hyperbox that its class label is
            used for predicting the class label of the input pattern :math:`X_i`.
            If there are only categorical features but many winner hyperboxes
            belonging to different classes, a random selection will be used to
            choose the final class label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.categorical_features_ is not None:
            X[:, self.categorical_features_] = impute_missing_value_cat_feature(X[:, self.categorical_features_])
            X, _ = one_hot_encoding_cat_feature(X, self.categorical_features_, self.encoders_)
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                X_con = X[:, self.continuous_features_].astype(float)
            else:
                X_con = None
            X_cat = X[:, self.categorical_features_]
            y_pred = self._predict(X_con, X_con, X_cat)
        else:
            y_pred = self._predict(X, X, None)

        return y_pred

    def _predict(self, Xl, Xu, Xd):
        """
        Predict class labels for samples in the form of hyperboxes represented
        by low bounds `Xl` and upper bounds `Xu`.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i` in the form of an hyperbox represented by
            a lower bound :math:`Xl_i` and an upper bound :math:`Xu_i` for
            continous features and a bound :math:`Xd_i` for categorical features,
            an additional criterion based on the minimum Manhattan distance
            between the central point of continous features in the input hyperbox
            :math:`X_i - [Xl_i, Xu_i]` and the central points of continous
            features in winner hyperboxes are used to find the final winner
            hyperbox that its class label is used for predicting the class
            label of the input hyperbox :math:`X_i`.

        .. warning::

            Another important point to pay attention is that the categorical
            features storing in `Xd` need to be encoded by using the function
            :func:`one_hot_encoding_cat_feature` before pushing the values
            to this method.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains the lower bounds of input patterns
            for which we want to predict the targets.
        Xu : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains the upper bounds of input patterns
            for which we want to predict the targets.
        Xd : array-like of shape (n_samples, n_cat_features)
            The data matrix contains  the bounds for categorical features
            of input patterns for which we want to predict the targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        y_pred = predict_onehot_cat_feature_manhanttan(self.V, self.W, self.D, self.C, Xl, Xu, Xd, self.gamma)

        return y_pred

    def predict_with_membership(self, X):
        """
        Predict class membership values of the input samples X including
        both categorical and continuous features.

        The predicted class membership value is the membership value
        of the representative hyperbox of that class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        mem_vals : ndarray of shape (n_samples, n_classes)
            The class membership values of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.categorical_features_ is not None:
            X[:, self.categorical_features_] = impute_missing_value_cat_feature(X[:, self.categorical_features_])
            X, _ = one_hot_encoding_cat_feature(X, self.categorical_features_, self.encoders_)
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                X_con = X[:, self.continuous_features_].astype(float)
            else:
                X_con = None
            Xd = X[:, self.categorical_features_]
            mem_vals = self._predict_with_membership(X_con, X_con, Xd)
        else:
            mem_vals = self._predict_with_membership(X, X, None)

        return mem_vals

    def _predict_with_membership(self, Xl, Xu, Xd):
        """
        Predict class membership values of the input hyperboxes represented by
        lower bounds Xl and upper bounds Xu for continuous features and
        categorical bounds X_cat for categorical features.

        The predicted class membership value is the membership value
        of the representative hyperbox of that class.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_continuous_features)
            The lower bounds for continous features of input hyperboxes.
        Xu : array-like of shape (n_samples, n_continuous_features)
            The upper bounds for continous features of input hyperboxes.
        Xd : array-like of shape (n_samples, n_cat_features)
            The bounds for categorical features of input hyperboxes.

        Returns
        -------
        mem_vals : ndarray of shape (n_samples, n_classes)
            The class membership values of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        if Xl is not None:
            if Xl.ndim == 1:
                Xl = Xl.reshape(1, -1)
            if Xu.ndim == 1:
                Xu = Xu.reshape(1, -1)

            if is_contain_missing_value(Xl) == True or is_contain_missing_value(Xu) == True:
                Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

        if Xd is not None:
            if Xd.ndim == 1:
                if Xd.shape[0] > 1:
                    Xd = Xd.reshape(1, -1)
                else:
                    Xd_tmp = np.zeros((1, 1), dtype=np.object)
                    Xd_tmp.fill(Xd)
                    Xd = Xd_tmp

        mem_vals, _ = get_membership_onehot_gfmm_all_classes(Xl, Xu, Xd, self.V, self.W, self.D, self.C, self.gamma)

        return mem_vals

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X including both
        continuous and categorical features.

        The predicted class probability is the fraction of the membership value
        of the representative hyperbox of that class and the sum of all
        membership values of all representative hyperboxes of all classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        mem_vals = self.predict_with_membership(X)
        normalizer = mem_vals.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = mem_vals / normalizer

        return proba

    def _predict_proba(self, Xl, Xu, Xd):
        """
        Predict class probabilities of the input hyperboxes represented by
        lower bounds Xl and upper bounds Xu for continuous features and
        categorical bounds X_cat for categorical features.

        The predicted class probability is the fraction of the membership value
        of the representative hyperbox of that class and the sum of all
        membership values of all representative hyperboxes of all classes.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_continuous_features)
            The lower bounds for continous features of input hyperboxes.
        Xu : array-like of shape (n_samples, n_continuous_features)
            The upper bounds for continous features of input hyperboxes.
        Xd : array-like of shape (n_samples, n_cat_features)
            The bounds for categorical features of input hyperboxes.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        mem_vals = self._predict_with_membership(Xl, Xu, Xd)
        normalizer = mem_vals.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = mem_vals / normalizer

        return proba

    def simple_pruning(self, X_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Simply prune low qualitied hyperboxes based on a pre-defined accuracy
        threshold for each hyperbox.

        Parameters
        ----------
        X_val : array-like of shape (n_samples, n_features)
            The data matrix contains both continous and categorical features of
            validation patterns.
        y_val : ndarray of shape (n_samples,)
            A vector contains the true class label corresponding to each
            validation pattern.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction
            process on the validation set. If True, keep them, otherwise the
            decision for keeping or removing based on the classification
            accuracy on the validation dataset.

        Returns
        -------
        self
            A hyperbox-based model with the low-qualitied hyperboxes pruned.

        """
        y_val = y_val.astype(int)
        n_val_samples = len(y_val)
        if self.categorical_features_ is not None:
            # Handle the case of existing categorical features
            X_val[:, self.categorical_features_] = impute_missing_value_cat_feature(X_val[:, self.categorical_features_])
            X_val, _ = one_hot_encoding_cat_feature(X_val, self.categorical_features_, self.encoders_)
            Xd_val = X_val[:, self.categorical_features_]

            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                Xval_con = X_val[:, self.continuous_features_].astype(float)
                if Xval_con.shape[0] > n_val_samples:
                    Xl_val = Xval_con[:n_val_samples, :]
                    Xu_val = Xval_con[n_val_samples:, :]
                    Xd_val = Xd_val[:n_val_samples, :]
                    self._simple_pruning(Xl_val, Xu_val, Xd_val, y_val, acc_threshold, keep_empty_boxes)
                else:
                    self._simple_pruning(Xval_con, Xval_con, Xd_val, y_val, acc_threshold, keep_empty_boxes)
            else:
                # No continous features
                self._simple_pruning(None, None, Xd_val, y_val, acc_threshold, keep_empty_boxes)
        else:
            # Handle the case of no categorical features
            if Xval_con.shape[0] > n_val_samples:
                Xl_val = Xval_con[:n_val_samples, :]
                Xu_val = Xval_con[n_val_samples:, :]
                self._simple_pruning(Xl_val, Xu_val, None, y_val, acc_threshold, keep_empty_boxes)
            else:
                self._simple_pruning(Xval_con, Xval_con, None, y_val, acc_threshold, keep_empty_boxes)

        return self

    def _simple_pruning(self, Xl_val, Xu_val, Xd_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Private function for simply pruning low qualitied hyperboxes based on
        a pre-defined accuracy threshold for each hyperbox.

        Parameters
        ----------
        Xl_val : array-like of shape (n_samples, n_con_features)
            The data matrix contains lower bounds of continous features in
            validation patterns.
        Xu_val : array-like of shape (n_samples, n_con_features)
            The data matrix contains upper bounds of continous features in
            validation patterns.
        Xd_val : array-like of shape (n_samples, n_cat_features)
            The data matrix contains the bounds of categorical features in
            validation patterns.
        y_val : ndarray of shape (n_samples,)
            A vector contains the true class label corresponding to each
            validation pattern.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction
            process on the validation set. If True, keep them, otherwise the
            decision for keeping or removing based on the classification
            accuracy on the validation dataset.

        Returns
        -------
        self
            A hyperbox-based model with the low-qualitied hyperboxes pruned.

        """
        if Xl_val is not None:
            n_samples = Xl_val.shape[0]
            if (is_contain_missing_value(Xl_val) == True) or (is_contain_missing_value(Xu_val) == True):
                Xl_val, Xu_val, y_val = convert_format_missing_input_zero_one(Xl_val, Xu_val, y_val)
        else:
            n_samples = Xd_val.shape[0]

        rnd = np.random
        rnd.seed(0)
        # Matrix stores the classification accuracy for each created hyperbox
        # in the trained model. The first column stores the number of corrected
        # classification samples and the second column stores the number of
        # wrong classification samples
        hyperboxes_performance = np.zeros((len(self.C), 2))

        for i in range(n_samples):
            if (Xl_val is not None) and (Xd_val is not None):
                # calculate memberships for all hyperboxes
                if not self.is_exist_continuous_missing_value:
                    mem_vals = membership_func_onehot_gfmm(Xl_val[i], Xu_val[i], Xd_val[i], self.V, self.W, self.D, self.gamma)
                else:
                    mem_vals = membership_func_onehot_gfmm(Xl_val[i], Xu_val[i], Xd_val[i], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.D, self.gamma)
            else:
                if Xl_val is not None:
                    if not self.is_exist_continuous_missing_value:
                        mem_vals = membership_func_onehot_gfmm(Xl_val[i], Xu_val[i], None, self.V, self.W, self.D, self.gamma)
                    else:
                        mem_vals = membership_func_onehot_gfmm(Xl_val[i], Xu_val[i], None, np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.D, self.gamma)
                else:
                    mem_vals = membership_func_onehot_gfmm(None, None, Xd_val[i], self.V, self.W, self.D, self.gamma)

            bmax = mem_vals.max() # get max membership value
            max_mem_V_id = np.nonzero(mem_vals == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(max_mem_V_id) == 1:
                # Only one hyperbox with the highest membership function
                if self.C[max_mem_V_id[0]] == y_val[i]:
                    hyperboxes_performance[max_mem_V_id[0], 0] = hyperboxes_performance[max_mem_V_id[0], 0] + 1                 
                else:
                    hyperboxes_performance[max_mem_V_id[0], 1] = hyperboxes_performance[max_mem_V_id[0], 1] + 1
            else:
                if Xl_val is not None:
                    # More than one hyperbox with highest membership,
                    # we use Manhattan distance for categorical data
                    if ((Xl_val[i] > Xu_val[i]).any() == True) or ((self.V[max_mem_V_id] > self.W[max_mem_V_id]).any() == True):
                        maht_dist = manhattan_distance_with_missing_val(Xl_val[i], Xu_val[i], self.V[max_mem_V_id], self.W[max_mem_V_id])
                    else:
                        if (Xl_val[i] == Xu_val[i]).all() == False:
                            XlT_mat = np.ones((len(max_mem_V_id), 1)) * Xl_val[i]
                            XuT_mat = np.ones((len(max_mem_V_id), 1)) * Xu_val[i]
                            XgT_mat = (XlT_mat + XuT_mat) / 2
                        else:
                            XgT_mat = np.ones((len(max_mem_V_id), 1)) * Xl_val[i]
    
                        # Find all average points of all hyperboxes with the same membership value
                        avg_point_mat = (self.V[max_mem_V_id] + self.W[max_mem_V_id]) / 2
                        # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                        maht_dist = manhattan_distance(avg_point_mat, XgT_mat)
                    
                    id_min_dist = maht_dist.argmin()
                    # the id of the selected hyperbox
                    id_min_hyperbox = max_mem_V_id[id_min_dist]
                       
                    if self.C[id_min_hyperbox] != y_val[i] and y_val[i] != UNLABELED_CLASS:
                        hyperboxes_performance[id_min_hyperbox, 1] = hyperboxes_performance[id_min_hyperbox, 1] + 1
                    else:
                        hyperboxes_performance[id_min_hyperbox, 0] = hyperboxes_performance[id_min_hyperbox, 0] + 1
                else:
                    # There are no continuous features, we choose randomly
                    final_winner_box_id = rnd.choice(max_mem_V_id, 1, False)[0]
                    if self.C[final_winner_box_id] != y_val[i] and y_val[i] != UNLABELED_CLASS:
                        hyperboxes_performance[final_winner_box_id, 1] = hyperboxes_performance[final_winner_box_id, 1] + 1
                    else:
                        hyperboxes_performance[final_winner_box_id, 0] = hyperboxes_performance[final_winner_box_id, 0] + 1
                    
        # pruning handling based on the validation results
        n_hyperboxes = hyperboxes_performance.shape[0]
        id_remained_excl_empty_boxes = np.zeros(n_hyperboxes).astype(bool)
        id_remained_incl_empty_boxes = np.zeros(n_hyperboxes).astype(bool)
        for i in range(n_hyperboxes):
            if (hyperboxes_performance[i, 0] + hyperboxes_performance[i, 1] != 0) and (hyperboxes_performance[i, 0] / (hyperboxes_performance[i, 0] + hyperboxes_performance[i, 1]) >= acc_threshold):
                id_remained_excl_empty_boxes[i] = True
                id_remained_incl_empty_boxes[i] = True
            if (hyperboxes_performance[i, 0] + hyperboxes_performance[i, 1] == 0):
                id_remained_incl_empty_boxes[i] = True

        if keep_empty_boxes == True:
            if Xl_val is not None:
                self.V = self.V[id_remained_incl_empty_boxes]
                self.W = self.W[id_remained_incl_empty_boxes]
            if Xd_val is not None:
                self.D = self.D[id_remained_incl_empty_boxes]
            self.C = self.C[id_remained_incl_empty_boxes]
        else:
            # keep one hyperbox for class that all of its hyperboxes are prunned
            current_classes = np.unique(self.C)
            class_tmp = self.C[id_remained_excl_empty_boxes]
            for c in current_classes:
                if c not in class_tmp:
                    pos = np.nonzero(self.C == c)[0]
                    id_kept = rnd.randint(len(pos))
                    id_remained_excl_empty_boxes[pos[id_kept]] = True

            if Xl_val is not None:
                V_pruned_excl_empty_boxes = self.V[id_remained_excl_empty_boxes]
                W_pruned_excl_empty_boxes = self.W[id_remained_excl_empty_boxes]
                W_pruned_incl_empty_boxes = self.W[id_remained_incl_empty_boxes]
                V_pruned_incl_empty_boxes = self.V[id_remained_incl_empty_boxes]
            else:
                V_pruned_excl_empty_boxes = None
                W_pruned_excl_empty_boxes = None
                W_pruned_incl_empty_boxes = None
                V_pruned_incl_empty_boxes = None
            
            if Xd_val is not None:
                D_pruned_excl_empty_boxes = self.D[id_remained_excl_empty_boxes]
                D_pruned_incl_empty_boxes = self.D[id_remained_incl_empty_boxes]
            else:
                D_pruned_excl_empty_boxes = None
                D_pruned_incl_empty_boxes = None

            C_pruned_excl_empty_boxes = self.C[id_remained_excl_empty_boxes]
            C_pruned_incl_empty_boxes = self.C[id_remained_incl_empty_boxes]

            y_val_pred_excl_empty_boxes = predict_onehot_cat_feature_manhanttan(
                V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes,
                D_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes,
                Xl_val, Xu_val, Xd_val, self.gamma)
            y_val_pred_incl_empty_boxes = predict_onehot_cat_feature_manhanttan(
                V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes,
                D_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes,
                Xl_val, Xu_val, Xd_val, self.gamma)

            if (accuracy_score(y_val, y_val_pred_excl_empty_boxes) >= accuracy_score(y_val, y_val_pred_incl_empty_boxes)):
                if Xl_val is not None:
                    self.V = V_pruned_excl_empty_boxes
                    self.W = W_pruned_excl_empty_boxes
                if Xd_val is not None:
                    self.D = D_pruned_excl_empty_boxes
                self.C = C_pruned_excl_empty_boxes
            else:
                if Xl_val is not None:
                    self.V = V_pruned_incl_empty_boxes
                    self.W = W_pruned_incl_empty_boxes
                if Xd_val is not None:
                    self.D = D_pruned_incl_empty_boxes
                self.C = C_pruned_incl_empty_boxes

        return self

    def get_sample_explanation(self, x):
        """
        Get useful information for explaining the reason behind the predicted
        result for the input pattern represented by upper and lower bounds for
        continous features together with the bound for categorical feature.

        Parameters
        ----------
        x : ndarray of shape (n_feature,)
            The input pattern which needs to be explained includes both
            continuous features and categorical features.

        Returns
        -------
        y_pred : int
            The predicted class of the input pattern
        dict_mem_val_classes : dictionary
            A dictionary stores all membership values for all classes. The key is 
            class label and the value is the corresponding membership value.
        dict_min_point_classes : dictionary
            A dictionary stores all mimimal points of hyperboxes having the maximum 
            membership value for each class. The key is the class label and the value 
            is the minimal points of the hyperbox corresponding to that class
        dict_max_point_classes : dictionary
            A dictionary stores all maximal points of hyperboxes having the maximum 
            membership value for each class. The key is the class label and the value 
            is the maximal points of the hyperbox corresponding to that class.
        dict_cat_point_classes: dictionary
            A dictionary stores all categorical features of hyperboxes having
            the maximum membership value for each class. The key is the class
            label and the value is the bound of categeorical features of the
            hyperbox corresponding to that class.

        """
        if self.categorical_features_ is not None:
            x[self.categorical_features_] = np.where(pd.isna(x[self.categorical_features_]), CAT_MISSING_FEATURE, x[self.categorical_features_])
            x = one_hot_encoding_cat_feature(x.reshape(1, -1), self.categorical_features_, self.encoders_)[0][0]
            x_cat = x[self.categorical_features_]
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                x_con = x[self.continuous_features_].astype(float)
                return self._get_sample_explanation(x_con, x_con, x_cat)
            else:
                return self._get_sample_explanation(None, None, x_cat)
        else:
            x_con = x[self.continuous_features_]
            return self._get_sample_explanation(x_con, x_con, None)

    def _get_sample_explanation(self, xl, xu, xd):
        """
        Get useful information for explaining the reason behind the predicted
        result for the input pattern represented by upper and lower bounds for
        continous features together with the bound for categorical feature.

        .. note::

            The categorical features storing in `xd` need to be encoded by
            using the function :func:`one_hot_encoding_cat_feature` before
            pushing the values to this method.

        Parameters
        ----------
        xl : ndarray of shape (n_continuous feature,)
            Lower bounds of continuous features of the input pattern which
            needs to be explained.
        xu : ndarray of shape (n_continuous feature,)
            Upper bounds of continuous features of the input pattern which
            needs to be explained.
        xd : ndarray of shape (n_cat feature,)
            Bounds of categorical features of the input pattern which needs
            to be explained.

        Returns
        -------
        y_pred : int
            The predicted class of the input pattern
        dict_mem_val_classes : dictionary
            A dictionary stores all membership values for all classes. The key is 
            class label and the value is the corresponding membership value.
        dict_min_point_classes : dictionary
            A dictionary stores all mimimal points of hyperboxes having the maximum 
            membership value for each class. The key is the class label and the value 
            is the minimal points of the hyperbox corresponding to that class
        dict_max_point_classes : dictionary
            A dictionary stores all maximal points of hyperboxes having the maximum 
            membership value for each class. The key is the class label and the value 
            is the maximal points of the hyperbox corresponding to that class.
        dict_cat_point_classes: dictionary
            A dictionary stores all categorical features of hyperboxes having
            the maximum membership value for each class. The key is the class
            label and the value is the bound of categeorical features of the
            hyperbox corresponding to that class.

        """
        mem_vals_for_classes, hyperbox_id_for_classes = get_membership_onehot_gfmm_all_classes(xl, xu, xd, self.V, self.W, self.D, self.C, self.gamma)
        class_values = np.unique(self.C)
        # get predicted class label for the input sample
        y_pred = self._predict(xl, xu, xd)[0]
        # create dictionaries with keys being class labels and values being membership values, maximum and minimum points
        dict_mem_val_classes = {}
        dict_min_point_classes = {}
        dict_max_point_classes = {}
        dict_cat_point_classes = {}
        for _id, c in enumerate(class_values):
            dict_mem_val_classes[c] = mem_vals_for_classes[0][_id]
            box_id = hyperbox_id_for_classes[0][_id]
            if xl is not None:
                dict_min_point_classes[c] = self.V[box_id]
                dict_max_point_classes[c] = self.W[box_id]
            if xd is not None:
                dict_cat_point_classes[c] = self.D[box_id]

        return (y_pred, dict_mem_val_classes, dict_min_point_classes, dict_max_point_classes, dict_cat_point_classes)

    def get_n_hyperboxes(self):
        """
        Get number of hyperboxes in the trained hyperbox-based model

        Returns
        -------
        int
            Number of hyperboxes in the trained hyperbox-based classifier.

        """
        if self.categorical_features_ is not None:
            return self.D.shape[0]
        else:
            return self.V.shape[0]


if __name__ == '__main__':
    import argparse
    import os

    def dir_path(path):
        if os.path.isfile(path) and os.path.exists(path):
            return path
        else:
            raise argparse.ArgumentTypeError(
                f"{path} is not a valid path or file does not exist")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Expect {v} is an boolean value")

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='The description of parameters')

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required positional arguments
    required.add_argument('-training_file', type=dir_path,
                          help='A required argument for the path to training data file (including file name)', required=True)
    required.add_argument('-testing_file', type=dir_path,
                          help='A required argument for the path to testing data file (including file name)', required=True)
    required.add_argument('-categorical_features', type=str,
                          help='Indices of categorical features', required=True)
    # Optional arguments
    optional.add_argument('--theta', type=float, default=0.5,
                          help='Maximum hyperbox size (in the range of (0, 1]) (default: 0.5)')
    optional.add_argument('--theta_min', type=float, default=0.5,
                          help='Mimimum value of the maximum hyperbox size to escape the training loop (in the range of (0, 1]) (default: 0.5)')
    optional.add_argument('--min_percent_overlap_cat', type=float, default=0.5,
                          help='Mimimum rate of numbers of categorical features overlapped for hyperbox expansion (default: 0.5)')
    optional.add_argument('--gamma', type=float, default=1,
                          help='A sensitivity parameter describing the speed of decreasing of the membership function in each continous dimension (larger than 0) (default: 1)')
    optional.add_argument('--alpha', type=float, default=0.9,
                          help='Multiplier showing the decrease of theta in each step (default: 0.9)')

    args = parser.parse_args()

    if args.theta <= 0 or args.theta > 1:
        parser.error("--theta has to be in the range of (0, 1]")

    if args.theta_min <= 0 or args.theta_min > 1:
        parser.error("--theta_min has to be in the range of (0, 1]")

    if args.min_percent_overlap_cat <= 0 or args.min_percent_overlap_cat > 1:
        parser.error("--min_percent_overlap_cat has to be in the range of (0, 1]")

    if args.alpha <= 0 or args.alpha >= 1:
        parser.error("--alpha has to be in the range of (0, 1)")

    if args.gamma <= 0:
        parser.error("--gamma has to be larger than 0")

    gamma = args.gamma
    theta = args.theta
    theta_min = args.theta_min
    alpha = args.alpha
    min_percent_overlap_cat = args.min_percent_overlap_cat
    training_file = args.training_file
    testing_file = args.testing_file

    import json

    categorical_features = json.loads(args.categorical_features)

    df_train = pd.read_csv(training_file, header=None, na_values=pd._libs.parsers.STR_NA_VALUES)
    df_test = pd.read_csv(testing_file, header=None, na_values=pd._libs.parsers.STR_NA_VALUES)

    Xy_train = df_train.to_numpy()
    Xy_test = df_test.to_numpy()

    Xtr = Xy_train[:, :-1]
    ytr = Xy_train[:, -1].astype(int)

    Xtest = Xy_test[:, :-1]
    ytest = Xy_test[:, -1].astype(int)

    onehot_onln_gfmm_clf = OneHotOnlineGFMM(
        theta=theta, theta_min=theta_min, min_percent_overlap_cat=min_percent_overlap_cat,
        gamma=gamma, alpha=alpha)
    onehot_onln_gfmm_clf.fit(Xtr, ytr, categorical_features)
    print('Number of hyperboxes = %d'%onehot_onln_gfmm_clf.get_n_hyperboxes())
    
    y_pred = onehot_onln_gfmm_clf.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # sample_need_explain = 10
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes, cat_poins_classes = onehot_onln_gfmm_clf.get_sample_explanation(Xtest[sample_need_explain])
    # print("Explain samples:")
    # print("Membership values for classes: ", mem_val_classes)
    # print("Predicted class = ", y_pred_input_0)
    # print("Minimum points of the selected hyperbox for each class: ", min_points_classes)
    # print("Maximum points of the selected hyperbox for each class: ", max_points_classes)
    # print("Categorical features of the selected hyperbox for each class: ", cat_poins_classes)
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/japanese_credit_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # onehot_onln_gfmm_clf.simple_pruning(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%onehot_onln_gfmm_clf.get_n_hyperboxes())
    
    # y_pred_2 = onehot_onln_gfmm_clf.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy (using a probability measure for samples on the boundary) = {acc_pruned * 100: .2f}%')
    