"""
General fuzzy min-max neural network trained by the extended improved
incremental learning algorithm for mixed attribute data.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import pandas as pd
import copy
import time
import random
import itertools
from sklearn.metrics import accuracy_score

from hbbrain.base.base_gfmm_estimator import (
    BaseGFMMClassifier,
    convert_format_missing_input_zero_one,
    is_contain_missing_value,
    predict_with_probability,
    predict_with_manhattan,
)
from hbbrain.utils.dist_metrics import manhattan_distance, manhattan_distance_with_missing_val
from hbbrain.utils.membership_calc import (
    membership_func_extended_iol_gfmm, 
    get_membership_extended_iol_gfmm_all_classes,
    membership_func_gfmm,
)
from hbbrain.utils.adjust_hyperbox import is_overlap_one_many_diff_label_hyperboxes_mixed_data_general
from hbbrain.constants import (
    UNLABELED_CLASS,
    PROBABILITY_MEASURE,
    MANHATTAN_DIS,
    DEFAULT_CATEGORICAL_VALUE,
)


def predict_with_manhattan_mixed_data(V, W, D, C, Xl, Xu, X_cat, g=1, alpha=0.5):
    """
    Predict class labels for samples in `X` represented in the form of invervals `[Xl, Xu]`
    for continuous features and `X_cat` for categorical features.
    This is a common function to determine the right class labels for X wrt. a trained hyperbox-based 
    classifier represented by `[V, W, D, C]`. It uses the winner-takes-all principle to predict 
    class labels for each sample in X by assigning the class label of the sample to the class 
    label of the hyperbox with the maximum membership value to that sample. It will use 
    a Manhattan distance for continous features in the case of many hyperboxes with different
    classes having the same maximum membership value.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all minimal points for all continuous features of all
        hyperboxes of a trained hyperbox-based model, in which each row is a
        minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all maximal points for all continuous features of all
        hyperboxes of a trained hyperbox-based model, in which each row is a
        maximal point of a hyperbox.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all categorical bounds for categorical features of all
        hyperboxes of a trained hyperbox-based model, in which each row is a
        categorical bound of a hyperbox.
    C : ndarray of shape (n_hyperboxes,)
        An array contains all class lables for all hyperboxes of a trained hyperbox-based model.
    Xl : array-like of shape (n_samples, n_continuous_features)
        The data matrix contains lower bounds for continuous features of input
        patterns for which we want to predict the targets.
    Xu : array-like of shape (n_samples, n_continuous_features)
        The data matrix contains upper bounds for continuous features of input
        patterns for which we want to predict the targets.
    X_cat : array-like of shape (n_samples, n_cat_features)
        The data matrix contains categorical bounds for categorical features
        of input patterns for which we want to predict the targets.
    g : float or array-like of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous dimension.
    alpha : float, optional, default=0.5
        The trade-off weighting factor between the impacts of categorical
        features and numerical features on the outputs of membership values.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        A vector contains the predictions. In binary and multiclass problems,
        this is a vector containing `n_samples`. 
    """
    if Xl is not None and Xl.ndim == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)
    if X_cat is not None and X_cat.ndim == 1:
        X_cat = X_cat.reshape(1, -1)

    if (Xl is not None) and ((is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True)):
        Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

    if X_cat is not None:
        X_cat = impute_missing_categorical_features(X_cat)

    if Xl is not None:
        n_samples = Xl.shape[0]
    else:
        n_samples = X_cat.shape[0]

    if V is not None:
        is_exist_missing_continous_value = (V > W).any()
    else:
        is_exist_missing_continous_value = False
    
    y_pred = np.full(n_samples, 0)
    sample_id = 0
    for i in range(n_samples):
        sample_id += 1
        if Xl is not None and X_cat is not None:
            if is_exist_missing_continous_value == False:
                mem_val = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], V, W, D, g, alpha)
            else:
                mem_val = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], np.minimum(V, W), np.maximum(W, V), D, g, alpha)
        else:
            if Xl is not None:
                if is_exist_missing_continous_value == False:
                    mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], V, W, g) # calculate memberships for all hyperboxes
                else:
                    mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], np.minimum(V, W), np.maximum(W, V), g) # calculate memberships for all hyperboxes
            else:
                mem_val = membership_func_extended_iol_gfmm(None, None, X_cat[i], V, W, D, g, alpha)

        bmax = mem_val.max() # get the maximum membership value
        
        if ((Xl[i] < 0).any() == True) or ((Xu[i] > 1).any() == True):
            print(">>> The testing sample %d with the coordinate %s is outside the range [0, 1]. Membership value = %f. The prediction is more likely incorrect." % (sample_id, Xl[i], bmax))

        # get indices of all hyperboxes with the maximum membership values
        max_mem_V_id = np.nonzero(mem_val == bmax)[0]
        winner_cls = np.unique(C[max_mem_V_id])
        if len(winner_cls) > 1:
            if Xl is None:
                y_pred[i] = np.random.choice(winner_cls, 1, False)[0]
            else:
                if ((Xl[i] > Xu[i]).any() == True) or ((V[max_mem_V_id] > W[max_mem_V_id]).any() == True):
                    maht_dist = manhattan_distance_with_missing_val(Xl[i], Xu[i], V[max_mem_V_id], W[max_mem_V_id])
                else:
                    if (Xl[i] == Xu[i]).all() == False:
                        Xl_mat = np.ones((len(max_mem_V_id), 1)) * Xl[i]
                        Xu_mat = np.ones((len(max_mem_V_id), 1)) * Xu[i]
                        Xg_mat = (Xl_mat + Xu_mat) / 2
                    else:
                        Xg_mat = np.ones((len(max_mem_V_id), 1)) * Xl[i]
                    # Find all average points of all hyperboxes with the same membership value
                    avg_point_mat = (V[max_mem_V_id] + W[max_mem_V_id]) / 2
                    # compute the Manhattan distance from Xg_mat to all average points of all hyperboxes with the same membership value
                    maht_dist = manhattan_distance(avg_point_mat, Xg_mat)
    
                id_min_dist = maht_dist.argmin()
                y_pred[i] = C[max_mem_V_id[id_min_dist]]
        else:
            y_pred[i] = C[max_mem_V_id[0]]
            
    return y_pred


def predict_with_probability_mixed_data(V, W, D, C, N_samples, Xl, Xu, X_cat, g=1, alpha=0.5):
    """
    Predict class labels for samples in `X` represented in the form of invervals `[Xl, Xu]`.
    This is a common function to determine the right class labels for X wrt. a trained hyperbox-based 
    classifier represented by `[V, W, C]`. It uses the winner-takes-all principle to predict 
    class labels for each sample in X by assigning the class label of the sample to the class 
    label of the hyperbox with the maximum membership value to that sample. It will use 
    a probability formula based on the number of samples included in each winner hyperbox 
    in the case of many hyperboxes with different classes having the same maximum membership value.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all minimal points of all hyperboxes of a trained
        hyperbox based model, each row is a minimal point for continuous
        features of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all maximal points of all hyperboxes of a trained
        hyperbox-based model, each row is a maximal point for continuous
        features of a hyperbox.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all maximal points of all hyperboxes of a trained
        hyperbox based model, each row is a categorical bound for a hyperbox.
    C : ndarray of shape (n_hyperboxes,)
        An array contains all class lables for all hyperboxes of a trained hyperbox-based model.
    N_samples : ndarray of shape (n_hyperboxes,)
        An array contains number of samples included in each hyperbox of a
        trained hyperbox-based model.
    Xl : array-like of shape (n_samples, n_continuous_features)
        The data matrix contains lower bounds of input patterns for which we
        want to predict the targets.
    Xu : array-like of shape (n_samples, n_continuous_features)
        The data matrix contains upper bounds of input patterns for which we
        want to predict the targets.
    X_cat : array-like of shape (n_samples, n_cat_features)
        The data matrix contains categorical bounds of input categorical
        patterns for which we want to predict the targets.
    g : float or array-like of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous dimension.
    alpha : float, optional, default=0.5
        The trade-off weighting factor between the impacts of categorical
        features and numerical features on the outputs of membership values.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        A vector contains the predictions. In binary and multiclass problems,
        this is a vector containing `n_samples`.

    """
    if Xl is not None:
        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
            Xu = Xu.reshape(1, -1)

        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

    if X_cat is not None and X_cat.ndim == 1:
        X_cat = X_cat.reshape(1, -1)

    if X_cat is not None:
        X_cat = impute_missing_categorical_features(X_cat)

    if Xl is not None:
        n_samples = Xl.shape[0]
    else:
        n_samples = X_cat.shape[0]

    if V is not None:
        is_exist_missing_continous_value = (V > W).any()
    else:
        is_exist_missing_continous_value = False

    y_pred = np.full(n_samples, 0)
    sample_id = 0
    # classifications
    for i in range(n_samples):
        sample_id += 1
        if Xl is not None and X_cat is not None:
            if is_exist_missing_continous_value == False:
                mem_val = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], V, W, D, g, alpha)
            else:
                mem_val = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], np.minimum(V, W), np.maximum(W, V), D, g, alpha)
        else:
            if Xl is not None:
                if is_exist_missing_continous_value == False:
                    mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], V, W, g) # calculate memberships for all hyperboxes
                else:
                    mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], np.minimum(V, W), np.maximum(W, V), g) # calculate memberships for all hyperboxes
            else:
                mem_val = membership_func_extended_iol_gfmm(None, None, X_cat[i], V, W, D, g, alpha)

        bmax = mem_val.max() # get the maximum membership value
        
        if ((Xl[i] < 0).any() == True) or ((Xu[i] > 1).any() == True):
            print(">>> The testing sample %d with the coordinate %s is outside the range [0, 1]. Membership value = %f. The prediction is more likely incorrect." % (sample_id, Xl[i], bmax))

        # get indices of all hyperboxes with the maximum membership values
        max_mem_V_id = np.nonzero(mem_val == bmax)[0]
        
        cls_same_mem = np.unique(C[max_mem_V_id])
        if len(cls_same_mem) > 1:
            cls_val = UNLABELED_CLASS
            
            is_find_prob_val = True
            if bmax == 1:
                id_box_with_one_sample = np.nonzero(N_samples[max_mem_V_id] == 1)[0]
                if len(id_box_with_one_sample) > 0:
                    is_find_prob_val = False
                    random.seed(0)
                    sel_id = random.choice(max_mem_V_id[id_box_with_one_sample])
                    cls_val = C[sel_id]
                    
            if is_find_prob_val == True:
                sum_prod_denum = (mem_val[max_mem_V_id] * N_samples[max_mem_V_id]).sum()
                max_prob = -1
                pre_id_cls = None
                for c in cls_same_mem:
                    id_cls = np.nonzero(C[max_mem_V_id] == c)[0]
                    sum_pro_num = (mem_val[max_mem_V_id[id_cls]] * N_samples[max_mem_V_id[id_cls]]).sum()
                    if sum_prod_denum != 0:
                        prob_val = sum_pro_num / sum_prod_denum
                    else:
                        prob_val = 0
                    
                    if prob_val > max_prob or ((prob_val == max_prob) and (pre_id_cls is not None) and (N_samples[max_mem_V_id[id_cls]].sum() > N_samples[max_mem_V_id[pre_id_cls]].sum())):
                        max_prob = prob_val
                        cls_val = c
                        pre_id_cls = id_cls
          
            y_pred[i] = cls_val
        else:
            y_pred[i] = C[max_mem_V_id[0]]
            
    return y_pred


def impute_missing_categorical_features(X_cat):
    """
    Impute missing values in categorical features by a default value.

    Parameters
    ----------
    X_cat : array-like of shape (n_samples, n_cat_features)
        Input matrix contains categorical features only.

    Returns
    -------
    X_cat : array-like of shape (n_samples, n_cat_features)
        The resulting matrix contains categorical features of which missing
        categorical values have been imputed by a default value.

    """
    id_missing_values = pd.isna(X_cat)
    if id_missing_values.any() == True:
        X_cat[id_missing_values] = DEFAULT_CATEGORICAL_VALUE
    return X_cat


class ExtendedImprovedOnlineGFMM(BaseGFMMClassifier):
    """Extended improved online learning algorithm for a general fuzzy min-max
    neural network with mixed-attribute data.
    
    This algorithm can handle the datasets with both continuous and categorical
    features. It uses the change in the entropy values of categorical features
    of the samples contained in a hyperbox to determine if the current hyperbox
    can be expanded to include the categorical values of a new training instance.
    An extended architecture of the original general fuzzy min-max neural network
    and its new membership function are also introduced for mixed-attribute data.

    See [1]_ for more detailed information regarding this extended improved
    online learning algorithm.
    
    Parameters
    ----------
    theta : float, optional, default=0.5
        Maximum hyperbox size for continuous features.
    gamma : float or ndarray of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous feature.
    delta : float, optional, default=0.5
        A maximum entropy changing threshold for categorical values after
        expansion of the existing hyperbox to cover an input pattern.
    alpha : float, optional, default=0.5
        A trade-off factor regulating the contribution level of continous
        features part and categorical features part to the membership score.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all minimal points for continuous features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all maximal points for continuous features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores a special structure for categorical features of all
        existing hyperboxes. Each element in `D` stores a set of symbolic
        values with their cardinalities for the j-th categorical dimension of
        a given hyperbox.
    C : array-like of shape (n_hyperboxes,)
        A vector stores all class labels correponding to existing hyperboxes.
    N_samples : array-like of shape (n_hyperboxes,)
        A vector stores the number of samples fully included in each existing
        hyperbox.

    Attributes
    ----------
    categorical_features_ : int array of shape (n_cat_features,)
        Indices of categorical features in the training data and hyperboxes.
    continuous_features_ : int array of shape (n_continuous_features,)
        Indices of continuous features in the training data and hyperboxes.
    is_exist_continuous_missing_value : boolean
        Is there any missing values in continuous features in the training data.
    elapsed_training_time : float
        Training time in seconds.

    References
    ----------
    .. [1] T. T. Khuat and B. Gabrys "An Online Learning Algorithm for a
           Neuro-Fuzzy Classifier with Mixed-Attribute Data", ArXiv preprint
           arXiv:2009.14670, 2020.

    Examples
    --------
    >>> from hbbrain.mixed_data.eiol_gfmm import ExtendedImprovedOnlineGFMM
    >>> from hbbrain.datasets import load_japanese_credit
    >>> X, y = load_japanese_credit()
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> numerical_features = [1, 2, 7, 10, 13, 14]
    >>> categorical_features = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    >>> scaler.fit(X[:, numerical_features])
    MinMaxScaler()
    >>> X[:, numerical_features] = scaler.transform(X[:, numerical_features])
    >>> clf = ExtendedImprovedOnlineGFMM(theta=0.1, delta=0.6)
    >>> clf.fit(X, y, categorical_features)
    >>> print("Number of hyperboxes = %d"%clf.get_n_hyperboxes())
    Number of hyperboxes = 613
    >>> clf.predict(X[[10, 100]])
    array([1, 0])
    """

    def __init__(self, theta=0.5, gamma=1, delta=0.5, alpha=0.5, V=None, W=None, D=None, C=None, N_samples=None):
        BaseGFMMClassifier.__init__(
            self, theta, gamma, False, V, W, C)
        self.alpha = alpha
        self.delta = delta
        if D is not None:
            self.D = D
        else:
            self.D = np.array([])
        if N_samples is not None:
            self.N_samples = N_samples
        else:
            self.N_samples = np.array([])

    def _init_data(self):
        """
        Initialise data for hyperboxes.

        Returns
        -------
        None.

        """
        self._init_hyperboxes()
        if self.D is None:
            self.D = np.array([])
        if self.N_samples is None:
            self.N_samples=np.array([])

    def compute_increasing_entropy(self, cat_extended_hyperbox, cat_cur_hyperbox):
        """
        Compute the increasing degree in the entropy for each categorical
        feature in both the current hyperbox and that hyperbox after extended.

        Parameters
        ----------
        cat_extended_hyperbox : array-like of shape (n_cat_features,)
            Categorical features in the current hyperbox after extended.
            Each dimension contains a dictionary with the key being categorical
            values and value being the number of samples in the hyperbox
            containing the given categorical value in that dimension.
        cat_cur_hyperbox : array-like of shape (n_cat_features,)
            Categorical features in the current hyperbox.
            Each dimension contains a dictionary with the key being categorical
            values and value being the number of samples in the hyperbox
            containing the given categorical value in that dimension.

        Returns
        -------
        increased_entropy : array-like of shape (n_cat_features,)
            The increased entropy value for each categorical dimension after
            extended.

        """
        n_cat_features = len(cat_extended_hyperbox)
        increased_entropy = np.zeros(n_cat_features)
        for j in range(n_cat_features):
            n_new = sum(cat_extended_hyperbox[j].values())
            n = sum(cat_cur_hyperbox[j].values())
            p_new = np.array(list(cat_extended_hyperbox[j].values())) / n_new
            p = np.array(list(cat_cur_hyperbox[j].values())) / n
            increased_entropy[j] = sum(-p_new * np.log2(p_new)) - n / n_new * sum(-p * np.log2(p))

        return increased_entropy
    
    def fit(self, X, y, categorical_features=None, N_incl_samples=None, type_cat_expansion=0):
        """
        Build a general fuzzy min-max neural network from the training set
        (X, y) using the extended improved online learning algorithm.

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
        N_incl_samples : array-like of shape (n_samples,), optional, default=None
            A vector stores numbers of samples fully contained in the input
            patterns in the case that input patterns form hyperboxes.
        type_cat_expansion : int, optional, default=0
            Type of the expansion condition for categorical features.
            If `type_cat_expansion` gets the value of 0, then the categorical
            feature expansion condition regarding the maximum entropy changing
            threshold will be applied for every categorical dimension.
            Otherwise, this expansion condition will be applied for the average
            entropy changing values of all categorical features.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        self.categorical_features_ = categorical_features
        if X.ndim == 1:
            X = X.reshape(shape=(1, -1))

        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        y = y.astype('int')

        if categorical_features is not None:
            X_cat = X[:, categorical_features]
            X_cat = impute_missing_categorical_features(X_cat)

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
                    return self._fit(Xl, Xu, None, y, N_incl_samples, type_cat_expansion)
                else:
                    X_cat = X_cat[:n_samples, :]
                    return self._fit(Xl, Xu, X_cat, y, N_incl_samples, type_cat_expansion)
            else:
                if categorical_features is None:
                    return self._fit(X_con, X_con, None, y, N_incl_samples)
                else:
                    return self._fit(X_con, X_con, X_cat, y, N_incl_samples, type_cat_expansion)
        else:
            self.continuous_features_ = None
            return self._fit(None, None, X_cat, y, N_incl_samples, type_cat_expansion)

    def _fit(self, Xl, Xu, X_cat, y, N_incl_samples=None, type_cat_expansion=0):
        """
        Build a general fuzzy min-max neural network from the training set
        using the extended improved online learning algorithm. Input training
        data in this method were split into continuous features with lower and
        upper bounds and categorical features.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_continuous_features)
            A matrix stores the lower bounds of training continuous features.
            If there is no continuous feature, this variable will get a None value.
        Xu : array-like of shape (n_samples, n_continuous_features)
            A matrix stores the upper bounds of training continuous features.
            If there is no continuous feature, this variable will get a None value.
        X_cat : array-like of shape (n_samples, n_cat_features)
            A matrix stores the training categorical features. If there is no
            categorical feature, this variable will get a None value.
        y : array-like of shape (n_samples,)
            The class labels.
        N_incl_samples : array-like of shape (n_samples,), optional, default=None
            A vector stores numbers of samples fully contained in the input
            hyperboxes.
        type_cat_expansion : int, optional, default=0
            Type of the expansion condition for categorical features.
            If `type_cat_expansion` gets the value of 0, then the categorical
            feature expansion condition regarding the maximum entropy changing
            threshold will be applied for every categorical dimension.
            Otherwise, this expansion condition will be applied for the average
            entropy changing values of all categorical features.

        Returns
        -------
        self : object
            The fitted estimator.

        """
        if Xl is not None:
            n_samples = Xl.shape[0]
            n_continuous_features = Xl.shape[1]
        else:
            n_samples = X_cat.shape[0]
            n_continuous_features = 0

        if X_cat is not None:
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(-1, 1)
            n_cat_features = X_cat.shape[1]
        else:
            n_cat_features = 0
        
        self._init_data()
        
        self.is_exist_continuous_missing_value = False
        if Xl is not None:
            if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
                self.is_exist_continuous_missing_value = True
                Xl, Xu, y = convert_format_missing_input_zero_one(Xl, Xu, y)
        
        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        time_start = time.perf_counter()

        is_firt_run = True
        for i in range(n_samples):
            if (n_continuous_features > 0 and self.V.size == 0) or (n_cat_features > 0 and self.D.size == 0):
                # no model provided, start from scratch
                if n_continuous_features > 0:
                    self.V = np.array([Xl[i]])
                    self.W = np.array([Xu[i]])
                if n_cat_features > 0:
                    self.D = np.empty((1, n_cat_features), dtype=object)
                    for j in range(n_cat_features):
                        dis = {}
                        if N_incl_samples is None:
                            dis[X_cat[i, j]] = 1
                        else:
                            dis[X_cat[i, j]] = N_incl_samples[i]
                        
                        self.D[0, j] = dis
                    
                self.C = np.array([y[i]])
                # save number of samples included in each hyperbox
                if N_incl_samples is None:
                    self.N_samples = np.array([1])
                else:
                    self.N_samples = np.array([N_incl_samples[i]])
            else:
                if y[i] == UNLABELED_CLASS:
                    id_same_input_label_group = np.arange(len(self.C))
                else:
                    id_same_input_label_group = np.nonzero(((self.C == y[i]) | (self.C == UNLABELED_CLASS)))[0]

                if len(id_same_input_label_group) > 0: 
                    if n_continuous_features > 0:
                        V_sameX = self.V[id_same_input_label_group]
                        W_sameX = self.W[id_same_input_label_group]
                    else:
                        V_sameX, W_sameX = None, None

                    if n_cat_features > 0:
                        D_sameX = self.D[id_same_input_label_group]
                    else:
                        D_sameX = None

                    lb_sameX = self.C[id_same_input_label_group]

                    if n_continuous_features > 0 and n_cat_features > 0:
                        if not self.is_exist_continuous_missing_value:
                            b = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], V_sameX, W_sameX, D_sameX, self.gamma, self.alpha)
                        else:
                            b = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], np.minimum(V_sameX, W_sameX), np.maximum(W_sameX, V_sameX), D_sameX, self.gamma, self.alpha)
                    else:
                        if n_continuous_features > 0:
                            if not self.is_exist_continuous_missing_value:
                                b = membership_func_extended_iol_gfmm(Xl[i], Xu[i], None, V_sameX, W_sameX, D_sameX, self.gamma, self.alpha)
                            else:
                                b = membership_func_extended_iol_gfmm(Xl[i], Xu[i], None, np.minimum(V_sameX, W_sameX), np.maximum(W_sameX, V_sameX), D_sameX, self.gamma, self.alpha)
                        else:
                            b = membership_func_extended_iol_gfmm(None, None, X_cat[i], V_sameX, W_sameX, D_sameX, self.gamma, self.alpha)

                    index = np.argsort(b)[::-1]
                    
                    if b[index[0]] != 1:
                        adjust = False
                        is_refind_diff_hyperbox = True
                        if y[i] != UNLABELED_CLASS:
                            id_lb_diff = ((self.C != y[i]) | (self.C == UNLABELED_CLASS))

                        for j in id_same_input_label_group[index]:
                            is_meet_continous_cond = True
                            is_meet_categorical_cond = True
                            if n_continuous_features > 0:
                                minV_new = np.minimum(self.V[j], Xl[i])
                                maxW_new = np.maximum(self.W[j], Xu[i])
                                is_meet_continous_cond = ((maxW_new - minV_new) <= self.theta).all()
                            else:
                                minV_new, maxW_new = None, None
                            
                            if y[i] == UNLABELED_CLASS:
                                id_lb_diff = ((self.C != self.C[j]) | (self.C == UNLABELED_CLASS))

                            if is_refind_diff_hyperbox == True:
                                if y[i] != UNLABELED_CLASS:
                                    is_refind_diff_hyperbox = False

                                no_check_overlap = False
                                
                                if len(id_lb_diff) == 0:
                                    # No hyperbox belongs to other class
                                    no_check_overlap = True
    
                                if n_cat_features > 0 and no_check_overlap == False:
                                    D_diff = self.D[id_lb_diff]
                                else:
                                    D_diff = None
                                
                                if no_check_overlap == False:
                                    N_sample_diff = self.N_samples[id_lb_diff]

                                if n_continuous_features > 0 and no_check_overlap == False:
                                    V_diff = self.V[id_lb_diff]
                                    W_diff = self.W[id_lb_diff]
                                    
                                    # examine only hyperboxes w/o missing dimensions,
                                    # meaning that in each dimension upper bound is
                                    # larger than lower bounds
                                    indcomp = np.nonzero((W_diff >= V_diff).all(axis = 1))[0]
                                    if len(indcomp) == 0:
                                        no_check_overlap = True
                                    else:
                                        V_diff = V_diff[indcomp]
                                        W_diff = W_diff[indcomp]
                                        N_sample_diff = N_sample_diff[indcomp]
                                        if n_cat_features > 0:
                                            D_diff = D_diff[indcomp]
                                else:
                                    V_diff, W_diff = None, None

                            if (n_cat_features > 0) and (is_meet_continous_cond == True):
                                D_new = np.empty(n_cat_features, dtype=object)
                                for fi in range(n_cat_features):
                                    dis = copy.deepcopy(self.D[j, fi])
                                    if X_cat[i, fi] in dis:
                                        if N_incl_samples is None:
                                            dis[X_cat[i, fi]] += 1
                                        else:
                                            dis[X_cat[i, fi]] += N_incl_samples[i]
                                    else:
                                        if N_incl_samples is None:
                                            dis[X_cat[i, fi]] = 1
                                        else:
                                            dis[X_cat[i, fi]] = N_incl_samples[i]

                                    D_new[fi] = dis

                                increased_entropy = self.compute_increasing_entropy(D_new, self.D[j])
                                if type_cat_expansion == 0:
                                    is_meet_categorical_cond = (increased_entropy <= self.delta).all()
                                else:
                                    is_meet_categorical_cond = np.average(increased_entropy) <= self.delta
                            else:
                                D_new = None
                                
                            if N_incl_samples is None:
                                N_sample_new = self.N_samples[j] + 1
                            else:
                                N_sample_new = self.N_samples[j] + N_incl_samples[i]

                            # test violation of max hyperbox size and class labels
                            if (is_meet_continous_cond == True) and (is_meet_categorical_cond == True):
                                if no_check_overlap == False and y[i] == UNLABELED_CLASS and self.C[j] == UNLABELED_CLASS:
                                    if n_continuous_features > 0:
                                        # remove hyperbox themself
                                        keep_id = (V_diff != self.V[j]).any(1)
                                        V_diff = V_diff[keep_id]
                                        W_diff = W_diff[keep_id]
                                        N_sample_diff = N_sample_diff[keep_id]
                                        if n_cat_features > 0:
                                            D_diff = D_diff[keep_id]
                                    else:
                                        # remove hyperbox themself
                                        keep_id = (D_diff != self.D[j]).any(1)
                                        D_diff = D_diff[keep_id]
                                        N_sample_diff = N_sample_diff[keep_id]

                                # Test overlap    
                                if no_check_overlap == True or is_overlap_one_many_diff_label_hyperboxes_mixed_data_general(V_diff, W_diff, D_diff, N_sample_diff, minV_new, maxW_new, D_new, N_sample_new) == False:
                                    # adjust the j-th hyperbox
                                    if n_continuous_features > 0:
                                        self.V[j] = minV_new
                                        self.W[j] = maxW_new
                                    
                                    if n_cat_features > 0:
                                        self.D[j] = D_new
                                    
                                    self.N_samples[j] = N_sample_new
                                    
                                    if y[i] != UNLABELED_CLASS and self.C[j] == UNLABELED_CLASS:
                                        self.C[j] = y[i]

                                    adjust = True
                                    break

                        # if i-th sample did not fit into any existing box, create a new one
                        if not adjust:
                            if n_continuous_features > 0:
                                self.V = np.concatenate((self.V, Xl[i].reshape(1, -1)), axis = 0)
                                self.W = np.concatenate((self.W, Xu[i].reshape(1, -1)), axis = 0)

                            if n_cat_features > 0:
                                tmp_D = np.empty((1, n_cat_features), dtype=object)
                                for ll in range(n_cat_features):
                                    dis = {}
                                    if N_incl_samples is None:
                                        dis[X_cat[i, ll]] = 1
                                    else:
                                        dis[X_cat[i, ll]] = N_incl_samples[i]
                                    
                                    tmp_D[0, ll] = dis
                                    
                                self.D = np.concatenate((self.D, tmp_D), axis = 0)

                            self.C = np.concatenate((self.C, [y[i]]))
                            if N_incl_samples is None:
                                self.N_samples = np.concatenate((self.N_samples, [1]))
                            else:
                                self.N_samples = np.concatenate((self.N_samples, [N_incl_samples[i]]))

                    else:
                        t = 0
                        # Find the first winner hyperbox with the same class with the input pattern
                        while (t + 1 < len(index)) and (b[index[t]] == 1) and (self.C[id_same_input_label_group[index[t]]] != y[i]) and (self.C[id_same_input_label_group[index[t]]] != UNLABELED_CLASS):
                            t = t + 1
                        if b[index[t]] == 1 and self.C[id_same_input_label_group[index[t]]] == y[i]:
                            # Update class label for the unlabelled hyperbox
                            if y[i] != UNLABELED_CLASS and self.C[id_same_input_label_group[index[t]]] == UNLABELED_CLASS:
                                self.C[id_same_input_label_group[index[t]]] = y[i]
                            # Update categorical values
                            for fi in range(n_cat_features):
                                if X_cat[i, fi] in self.D[id_same_input_label_group[index[t]], fi]:
                                    if N_incl_samples is None:
                                        self.D[id_same_input_label_group[index[t]], fi][X_cat[i, fi]] += 1
                                    else:
                                        self.D[id_same_input_label_group[index[t]], fi][X_cat[i, fi]] += N_incl_samples[i]
                                else:
                                    if N_incl_samples is None:
                                        self.D[id_same_input_label_group[index[t]], fi][X_cat[i, fi]] = 1
                                    else:
                                        self.D[id_same_input_label_group[index[t]], fi][X_cat[i, fi]] = N_incl_samples[i]
                            
                            if N_incl_samples is None:
                                self.N_samples[id_same_input_label_group[index[t]]] = self.N_samples[id_same_input_label_group[index[t]]] + 1
                            else:
                                self.N_samples[id_same_input_label_group[index[t]]] = self.N_samples[id_same_input_label_group[index[t]]] + N_incl_samples[i]
                else:
                    if n_continuous_features > 0:
                        self.V = np.concatenate((self.V, Xl[i].reshape(1, -1)), axis = 0)
                        self.W = np.concatenate((self.W, Xu[i].reshape(1, -1)), axis = 0)

                    if n_cat_features > 0:
                        tmp_D = np.empty((1, n_cat_features), dtype=object)
                        for ll in range(n_cat_features):
                            dis = {}
                            if N_incl_samples is None:
                                dis[X_cat[i, ll]] = 1
                            else:
                                dis[X_cat[i, ll]] = N_incl_samples[i]
                            
                            tmp_D[0, ll] = dis

                        self.D = np.concatenate((self.D, tmp_D), axis = 0)

                    self.C = np.concatenate((self.C, [y[i]]))

                    if N_incl_samples is None:
                        self.N_samples = np.concatenate((self.N_samples, [1]))
                    else:
                        self.N_samples = np.concatenate((self.N_samples, [N_incl_samples[i]]))       

        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start

        return self

    def predict(self, X, type_boundary_handling=PROBABILITY_MEASURE):
        """
        Predict class labels for samples in `X`.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i`, an additional criterion based on the
            probability generated by number of samples included in winner
            hyperboxes and membership values or the Manhattan distance between
            the central point of winner hyperboxes and the input sample is used
            to find the final winner hyperbox that its class label is used for
            predicting the class label of the input pattern :math:`X_i`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.

        type_boundary_handling : int, optional, default=PROBABILITY_MEASURE (aka 1)
            The way of handling many winner hyperboxes, i.e., PROBABILITY_MEASURE or MANHATTAN_DIS

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
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                X_con = X[:, self.continuous_features_].astype(float)
            else:
                X_con = None
            X_cat = X[:, self.categorical_features_]
            X_cat = impute_missing_categorical_features(X_cat)
            y_pred = self._predict(X_con, X_con, X_cat, type_boundary_handling)
        else:
            y_pred = self._predict(X, X, None, type_boundary_handling)

        return y_pred

    def _predict(self, Xl, Xu, X_cat, type_boundary_handling=PROBABILITY_MEASURE):
        """
        Predict class labels for samples in the form of hyperboxes represented
        by low bounds `Xl` and upper bounds `Xu`.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i` in the form of an hyperbox represented by
            a lower bound :math:`Xl_i` and an upper bound :math:`Xu_i` for
            continous features and a bound :math:`Xcat_i` for categorical
            features, an additional criterion based on a probability measure
            using the number of samples included in the hyperbox or the minimum
            Manhattan distance between the central point of continous features
            in the input hyperbox :math:`X_i = [Xl_i, Xu_i]` and the central
            points of continous features in winner hyperboxes are used to find
            the final winner hyperbox that its class label is used for predicting
            the class label of the input hyperbox :math:`X_i`.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains the lower bounds of input patterns
            for which we want to predict the targets.
        Xu : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains the upper bounds of input patterns
            for which we want to predict the targets.
        X_cat : array-like of shape (n_samples, n_cat_features)
            The data matrix contains the bounds for categorical features
            of input patterns for which we want to predict the targets.
        type_boundary_handling : int, optional, default=PROBABILITY_MEASURE (aka 1)
            The way of handling many winner hyperboxes, i.e., PROBABILITY_MEASURE or MANHATTAN_DIS

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        if type_boundary_handling == PROBABILITY_MEASURE:
            y_pred = predict_with_probability_mixed_data(self.V, self.W, self.D, self.C, self.N_samples, Xl, Xu, X_cat, self.gamma, self.alpha)
        else:
            y_pred = predict_with_manhattan_mixed_data(self.V, self.W, self.D, self.C, Xl, Xu, X_cat, self.gamma, self.delta)
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
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                X_con = X[:, self.continuous_features_].astype(float)
            else:
                X_con = None
            X_cat = X[:, self.categorical_features_]
            X_cat = impute_missing_categorical_features(X_cat)
            mem_vals = self._predict_with_membership(X_con, X_con, X_cat)
        else:
            mem_vals = self._predict_with_membership(X, X, None)

        return mem_vals

    def _predict_with_membership(self, Xl, Xu, X_cat):
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
        X_cat : array-like of shape (n_samples, n_cat_features)
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
                Xu = Xu.reshape(1, -1)

            if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
                Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

        if X_cat is not None:
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(1, -1)
            X_cat = impute_missing_categorical_features(X_cat)

        mem_vals, _ = get_membership_extended_iol_gfmm_all_classes(Xl, Xu, X_cat, self.V, self.W, self.D, self.C, self.gamma, self.alpha)

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
    
    def _predict_proba(self, Xl, Xu, X_cat):
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
        X_cat : array-like of shape (n_samples, n_cat_features)
            The bounds for categorical features of input hyperboxes.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        mem_vals = self._predict_with_membership(Xl, Xu, X_cat)
        normalizer = mem_vals.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = mem_vals / normalizer

        return proba

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

    def get_sample_explanation(self, x):
        """
        Get useful information for explaining the reason behind the predicted
        result for the input pattern represented by upper and lower bounds for
        continous features together with the categorical bounds for the
        categorical features.

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
            A dictionary stores all membership values for all classes. The key
            is class label and the value is the corresponding membership value.
        dict_min_point_classes : dictionary
            A dictionary stores all mimimal points of hyperboxes having the
            maximum membership value for each class. The key is the class label
            and the value is the minimal points of the hyperbox corresponding
            to that class.
        dict_max_point_classes : dictionary
            A dictionary stores all maximal points of hyperboxes having the
            maximum membership value for each class. The key is the class label
            and the value is the maximal points of the hyperbox corresponding
            to that class.
        dict_cat_bound_classes: dictionary
            A dictionary stores all categorical bounds of categorical features 
            for the hyperboxes having the maximum membership value for each class.
            The key is the class label and the value is the categorical bound of
            categorical features for the hyperboxes corresponding to each class.
        """
        if self.categorical_features_ is not None:
            x_cat = x[self.categorical_features_]
            x_cat = impute_missing_categorical_features(x_cat)
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                x_con = x[self.continuous_features_].astype(float)
                return self._get_sample_explanation(x_con, x_con, x_cat)
            else:
                return self._get_sample_explanation(None, None, x_cat)
        else:
            x_con = x[self.continuous_features_]
            return self._get_sample_explanation(x_con, x_con, None)

    def _get_sample_explanation(self, xl, xu, x_cat):
        """
        Get useful information for explaining the reason behind the predicted
        result for the input pattern represented by upper and lower bounds for
        continous features together with the categorical bounds for
        categorical features.

        Parameters
        ----------
        xl : ndarray of shape (n_continuous feature,)
            Lower bounds of continuous features of the input pattern which
            needs to be explained.
        xu : ndarray of shape (n_continuous feature,)
            Upper bounds of continuous features of the input pattern which
            needs to be explained.
        x_cat : ndarray of shape (n_cat feature,)
            Bounds of categorical features of the input pattern which needs
            to be explained.

        Returns
        -------
        y_pred : int
            The predicted class of the input pattern
        dict_mem_val_classes : dictionary
            A dictionary stores all membership values for all classes. The key
            is the class label and the value is the corresponding membership
            value.
        dict_min_point_classes : dictionary
            A dictionary stores all mimimal points of hyperboxes having the
            maximum membership value for each class. The key is the class label
            and the value is the minimal points of the hyperbox corresponding
            to that class.
        dict_max_point_classes : dictionary
            A dictionary stores all maximal points of hyperboxes having the
            maximum membership value for each class. The key is the class label
            and the value is the maximal points of the hyperbox corresponding
            to that class.
        dict_cat_bound_classes: dictionary
            A dictionary stores all categorical bounds of categorical features 
            for the hyperboxes having the maximum membership value for each class.
            The key is the class label and the value is the categorical bound of
            categorical features for the hyperboxes corresponding to each class.
        """
        mem_vals_for_classes, hyperbox_id_for_classes = get_membership_extended_iol_gfmm_all_classes(xl, xu, x_cat, self.V, self.W, self.D, self.C, self.gamma, self.alpha)
        class_values = np.unique(self.C)
        # get predicted class label for the input sample
        y_pred = self._predict(xl, xu, x_cat)[0]
        # create dictionaries with keys being class labels and values being membership values, maximum and minimum points
        dict_mem_val_classes = {}
        dict_min_point_classes = {}
        dict_max_point_classes = {}
        dict_cat_bound_classes = {}

        for _id, c in enumerate(class_values):
            dict_mem_val_classes[c] = mem_vals_for_classes[0][_id]
            box_id = hyperbox_id_for_classes[0][_id]
            if xl is not None:
                dict_min_point_classes[c] = self.V[box_id]
                dict_max_point_classes[c] = self.W[box_id]
            if x_cat is not None:
                dict_cat_bound_classes[c] = self.D[box_id]

        return (y_pred, dict_mem_val_classes, dict_min_point_classes, dict_max_point_classes, dict_cat_bound_classes)

    def simple_pruning(self, X_val, y_val, acc_threshold=0.5, keep_empty_boxes=False, type_boundary_handling=PROBABILITY_MEASURE):
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
        type_boundary_handling : int, optional, default=PROBABILITY_MEASURE (aka 1)
            The way of handling samples located on the boundary.

        Returns
        -------
        self
            A hyperbox-based model with the low-qualitied hyperboxes pruned.

        """
        y_val = y_val.astype(int)
        n_val_samples = len(y_val)
        if self.categorical_features_ is not None:
            # Handle the case of existing categorical features
            Xcat_val = X_val[:, self.categorical_features_]
            Xcat_val = impute_missing_categorical_features(Xcat_val)
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                Xval_con = X_val[:, self.continuous_features_].astype(float)
                if Xval_con.shape[0] > n_val_samples:
                    Xl_val = Xval_con[:n_val_samples, :]
                    Xu_val = Xval_con[n_val_samples:, :]
                    Xcat_val = Xcat_val[:n_val_samples, :]
                    self._simple_pruning(Xl_val, Xu_val, Xcat_val, y_val, acc_threshold, keep_empty_boxes, type_boundary_handling)
                else:
                    self._simple_pruning(Xval_con, Xval_con, Xcat_val, y_val, acc_threshold, keep_empty_boxes, type_boundary_handling)
            else:
                # No continous features
                self._simple_pruning(None, None, Xcat_val, y_val, acc_threshold, keep_empty_boxes, type_boundary_handling)
        else:
            # Handle the case of no categorical features
            if Xval_con.shape[0] > n_val_samples:
                Xl_val = Xval_con[:n_val_samples, :]
                Xu_val = Xval_con[n_val_samples:, :]
                self._simple_pruning(Xl_val, Xu_val, None, y_val, acc_threshold, keep_empty_boxes, type_boundary_handling)
            else:
                self._simple_pruning(Xval_con, Xval_con, None, y_val, acc_threshold, keep_empty_boxes, type_boundary_handling)

        return self

    def _simple_pruning(self, Xl_val, Xu_val, Xcat_val, y_val, acc_threshold=0.5, keep_empty_boxes=False, type_boundary_handling=PROBABILITY_MEASURE):
        """
        Simply prune low qualitied hyperboxes based on a pre-defined accuracy
        threshold for each hyperbox. This method handles the case of continuous
        features under the form of hyperboxes.

        Parameters
        ----------
        Xl_val : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains lower bounds for continuous features of
            validation patterns.
        Xu_val : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains upper bounds for continuous features of
            validation patterns.
        Xcat_val : array-like of shape (n_samples, n_cat_features)
            The data matrix contains categorical bounds for categorical
            features of validation patterns.
        y_val : ndarray of shape (n_samples,)
            A vector contains the true class label corresponding to each
            validation pattern.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction
            process on the validation set. If True, keep them, otherwise the
            decision for keeping or removing based on the classification
            accuracy on the validation dataset
        type_boundary_handling : int, optional, default=PROBABILITY_MEASURE (aka 1)
            The way of handling samples located on the boundary.

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
            n_samples = Xcat_val.shape[0]

        rnd = np.random
        rnd.seed(0)
        random.seed(0)
        # Matrices storing the classification accuracy for each created hyperbox in the trained model
        # The first column stores the number of corrected classification samples and the second column stores the number of wrong classification samples
        hyperboxes_performance = np.zeros((len(self.C), 2))
        
        for i in range(n_samples):
            if (Xl_val is not None) and (Xcat_val is not None):
                if not self.is_exist_continuous_missing_value:
                    mem_vals = membership_func_extended_iol_gfmm(Xl_val[i], Xu_val[i], Xcat_val[i], self.V, self.W, self.D, self.gamma, self.alpha)
                else:
                    mem_vals = membership_func_extended_iol_gfmm(Xl_val[i], Xu_val[i], Xcat_val[i], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.D, self.gamma, self.alpha)
            else:
                if Xl_val is not None:
                    if not self.is_exist_continuous_missing_value:
                        mem_vals = membership_func_extended_iol_gfmm(Xl_val[i], Xu_val[i], None, self.V, self.W, self.D, self.gamma, self.alpha)
                    else:
                        mem_vals = membership_func_extended_iol_gfmm(Xl_val[i], Xu_val[i], None, np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.D, self.gamma, self.alpha)
                else:
                    mem_vals = membership_func_extended_iol_gfmm(None, None, Xcat_val[i], self.V, self.W, self.D, self.gamma, self.alpha)

            bmax = mem_vals.max() # get max membership value
            max_mem_V_id = np.nonzero(mem_vals == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(max_mem_V_id) == 1:
                # Only one hyperbox with the highest membership function
                if self.C[max_mem_V_id[0]] == y_val[i]:
                    hyperboxes_performance[max_mem_V_id[0], 0] = hyperboxes_performance[max_mem_V_id[0], 0] + 1                 
                else:
                    hyperboxes_performance[max_mem_V_id[0], 1] = hyperboxes_performance[max_mem_V_id[0], 1] + 1
            else:
                # More than one hyperbox with highest membership
                if type_boundary_handling == PROBABILITY_MEASURE:
                    # Using a probability measure based on the number of samples included in each winner hyperbox and membership value
                    is_find_prob_val = True
                    if bmax == 1:
                        id_box_with_one_sample = np.nonzero(self.N_samples[max_mem_V_id] == 1)[0]
                        if len(id_box_with_one_sample) > 0:
                            is_find_prob_val = False
                            id_min_hyperbox = random.choice(max_mem_V_id[id_box_with_one_sample])
                            
                    if is_find_prob_val == True:
                        cls_same_mem = np.unique(self.C[max_mem_V_id])
                        sum_prod_denum = (mem_vals[max_mem_V_id] * self.N_samples[max_mem_V_id]).sum()
                        max_prob = -1
                        pre_id_cls = None
                        for c in cls_same_mem:
                            id_cls = np.nonzero(self.C[max_mem_V_id] == c)[0]
                            sum_pro_num = (mem_vals[max_mem_V_id[id_cls]] * self.N_samples[max_mem_V_id[id_cls]]).sum()
                            if sum_prod_denum != 0:
                                prob_val = sum_pro_num / sum_prod_denum
                            else:
                                prob_val = 0
                            
                            if prob_val > max_prob or ((prob_val == max_prob) and (pre_id_cls is not None) and (self.N_samples[max_mem_V_id[id_cls]].sum() > self.N_samples[max_mem_V_id[pre_id_cls]].sum())):
                                max_prob = prob_val
                                id_min_hyperbox = random.choice(max_mem_V_id[id_cls])
                                pre_id_cls = id_cls
                else:
                    # using Manhattan distance
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
            if Xcat_val is not None:
                self.D = self.D[id_remained_incl_empty_boxes]
            self.C = self.C[id_remained_incl_empty_boxes]
            self.N_samples = self.N_samples[id_remained_incl_empty_boxes]
        else:
            # keep one hyperbox for class that all of its hyperboxes are prunned
            current_classes = np.unique(self.C)
            class_tmp = self.C[id_remained_excl_empty_boxes]
            for c in current_classes:
                if c not in class_tmp:
                    pos = np.nonzero(self.C == c)[0]
                    id_kept = rnd.randint(len(pos))
                    id_remained_excl_empty_boxes[pos[id_kept]] = True
               
            V_pruned_excl_empty_boxes = self.V[id_remained_excl_empty_boxes]
            W_pruned_excl_empty_boxes = self.W[id_remained_excl_empty_boxes]
            D_pruned_excl_empty_boxes = self.D[id_remained_excl_empty_boxes]
            C_pruned_excl_empty_boxes = self.C[id_remained_excl_empty_boxes]
            N_samples_excl_empty_boxes = self.N_samples[id_remained_excl_empty_boxes]
            
            W_pruned_incl_empty_boxes = self.W[id_remained_incl_empty_boxes]
            V_pruned_incl_empty_boxes = self.V[id_remained_incl_empty_boxes]
            D_pruned_incl_empty_boxes = self.D[id_remained_incl_empty_boxes]
            C_pruned_incl_empty_boxes = self.C[id_remained_incl_empty_boxes]
            N_samples_incl_empty_boxes = self.N_samples[id_remained_incl_empty_boxes]
            
            if type_boundary_handling == PROBABILITY_MEASURE:
                y_val_pred_excl_empty_boxes = predict_with_probability_mixed_data(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, D_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, N_samples_excl_empty_boxes, Xl_val, Xu_val, Xcat_val, self.gamma, self.alpha)
                y_val_pred_incl_empty_boxes = predict_with_probability_mixed_data(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, D_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, N_samples_incl_empty_boxes, Xl_val, Xu_val, Xcat_val, self.gamma, self.alpha)
            else:
                y_val_pred_excl_empty_boxes = predict_with_manhattan_mixed_data(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, D_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, Xl_val, Xu_val, Xcat_val, self.gamma, self.alpha)
                y_val_pred_incl_empty_boxes = predict_with_manhattan_mixed_data(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, D_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, Xl_val, Xu_val, Xcat_val, self.gamma, self.alpha)
            
            if (accuracy_score(y_val, y_val_pred_excl_empty_boxes) >= accuracy_score(y_val, y_val_pred_incl_empty_boxes)):
                if Xl_val is not None:
                    self.V = V_pruned_excl_empty_boxes
                    self.W = W_pruned_excl_empty_boxes
                if Xcat_val is not None:
                    self.D = D_pruned_excl_empty_boxes
                self.C = C_pruned_excl_empty_boxes
                self.N_samples = N_samples_excl_empty_boxes
            else:
                if Xl_val is not None:
                    self.V = V_pruned_incl_empty_boxes
                    self.W = W_pruned_incl_empty_boxes
                if Xcat_val is not None:
                    self.D = D_pruned_incl_empty_boxes
                self.C = C_pruned_incl_empty_boxes
                self.N_samples = N_samples_incl_empty_boxes

        return self


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
    optional.add_argument('--delta', type=float, default=0.5,
                          help='Maximum changing entropy for categorical features (in the range of (0, 1]) (default: 0.5)')
    optional.add_argument('--gamma', type=float, default=1,
                          help='A sensitivity parameter describing the speed of decreasing of the membership function in each continous dimension (larger than 0) (default: 1)')
    optional.add_argument('--alpha', type=float, default=0.5,
                          help='The trade-off weighting factor between categorical features and numerical features for membership values (in the range of [0, 1]) (default: 0.5)')

    args = parser.parse_args()

    if args.theta <= 0 or args.theta > 1:
        parser.error("--theta has to be in the range of (0, 1]")

    if args.delta <= 0 or args.delta > 1:
        parser.error("--delta has to be in the range of (0, 1]")

    if args.alpha < 0 or args.alpha > 1:
        parser.error("--alpha has to be in the range of [0, 1]")

    if args.gamma <= 0:
        parser.error("--gamma has to be larger than 0")

    gamma = args.gamma
    theta = args.theta
    delta = args.delta
    alpha = args.alpha
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

    eiol_gfmm_clf = ExtendedImprovedOnlineGFMM(theta=theta, gamma=gamma, delta=delta, alpha=alpha)
    eiol_gfmm_clf.fit(Xtr, ytr, categorical_features, type_cat_expansion=0)
    print('Number of hyperboxes = %d'%eiol_gfmm_clf.get_n_hyperboxes())
    
    y_pred = eiol_gfmm_clf.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # sample_need_explain = 1
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes, dict_cat_bound_classes = eiol_gfmm_clf.get_sample_explanation(Xtest[sample_need_explain])
    # print("Explain samples:")
    # print("Membership values for classes: ", mem_val_classes)
    # print("Predicted class = ", y_pred_input_0)
    # print("Minimum continuous points of the selected hyperbox for each class: ", min_points_classes)
    # print("Maximum continuous points of the selected hyperbox for each class: ", max_points_classes)
    # print("Categorical bounds of the selected hyperbox for each class: ", dict_cat_bound_classes)

    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/japanese_credit_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # eiol_gfmm_clf.simple_pruning(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%eiol_gfmm_clf.get_n_hyperboxes())
    
    # y_pred_2 = eiol_gfmm_clf.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy (using a probability measure for samples on the boundary) = {acc_pruned * 100: .2f}%')
    