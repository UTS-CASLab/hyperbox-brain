"""
General fuzzy min-max neural network trained by the batch incremental
learning algorithm, in which categorical features are encoded using
the ordinal encoding method and the similarity among categorical
values are computed using their frequency of occurence with respect to all
class labels in a training set.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder

from hbbrain.base.base_estimator import BaseHyperboxClassifier
from hbbrain.base.base_gfmm_estimator import (
    convert_format_missing_input_zero_one,
    is_contain_missing_value,
)
from hbbrain.utils.membership_calc import (
    membership_func_freq_cat_gfmm,
    get_membership_freq_cat_gfmm_all_classes,
)
from hbbrain.utils.adjust_hyperbox import (
    overlap_resolving_num_data,
    is_two_hyperboxes_overlap_num_data_general,
    hyperbox_overlap_test_freq_cat_gfmm,
    hyperbox_contraction_freq_cat_gfmm,
)
from hbbrain.utils.dist_metrics import (
    manhattan_distance,
    manhattan_distance_with_missing_val,
)
from hbbrain.utils.matrix_transformation import hashing, hashing_mat
from hbbrain.constants import UNLABELED_CLASS, DEFAULT_CATEGORICAL_VALUE


def ordinal_encode_categorical_features(X, categorical_features, encoder=None):
    """
    Encode categorical features as an integer array.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        An input data matrix includes both continuous and categorical features.
    categorical_features : a list of integer
        Indices of categorical features in `X`.
    encoder : sklearn.preprocessing.OrdinalEncoder, optional, default=None
        An existing ordinal encoder is used to encode categorical features.

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        An input data matrix with the encoded categorical features.
    encoder : sklearn.preprocessing.OrdinalEncoder
        An ordinal encoder was used to encode categorical features.

    """
    X_cat = X[:, categorical_features]
    id_missing_values = pd.isna(X_cat)
    if id_missing_values.any() == True:
        X_cat[id_missing_values] = DEFAULT_CATEGORICAL_VALUE

    if encoder is None:
        encoder = OrdinalEncoder()
        encoder.fit(X_cat)

    X_cat_trans = encoder.transform(X_cat)
    if id_missing_values.any() == True:
        X_cat_trans[id_missing_values] = DEFAULT_CATEGORICAL_VALUE

    X_trans = X.copy()
    X_trans[:, categorical_features] = X_cat_trans

    return X_trans, encoder


def compute_similarity_among_categorical_values(X_cat, y):
    """
    Compute the similarity among pairs of categorical values
    for each categorical feature.

    Parameters
    ----------
    X_cat : array-like of shape (n_samples, n_cat_features)
        Input patterns contain only categorical features.
    y : array-like of shape (n_samples, )
        The class label corresponds to each input pattern.

    Returns
    -------
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical values
        for each categorical feature index. Each element in this array is an dictionary
        with keys being a hashed value of two categorical values and values of this
        dictionary being a similarity value.

    """
    similarity_of_cat_vals = np.full(X_cat.shape[1], None, dtype=np.object)
    unique_cls = np.unique(y)
    for i in range(X_cat.shape[1]):
        # Get unique values in each categorical feature
        cat_vals = np.unique(X_cat[:, i])
        if len(cat_vals) > 2:
            # return the number of elements for each class
            # corresponding to each unique categorical value
            store_prob_freq = {}
            for cat_val in cat_vals:
                cat_vals_cls, cat_vals_count = np.unique(y[X_cat[:, i] == cat_val], return_counts=True)
                if len(cat_vals_cls) == len(unique_cls):
                    store_prob_freq[cat_val] = cat_vals_count / np.sum(cat_vals_count)
                else:
                    # if a class with no categorical feature values, then cat_vals_cls does not contain
                    # that class id, and so we need to add probability values of zeros manually.
                    prob_cat_features = np.full(len(unique_cls), 0)
                    for cls_id, cls_val in enumerate(unique_cls):
                        count_cls_ele = cat_vals_count[cat_vals_cls == cls_val]
                        if len(count_cls_ele) > 0:
                            prob_cat_features[cls_id] = count_cls_ele[0]
                    store_prob_freq[cat_val] = prob_cat_features / np.sum(prob_cat_features)
            # compute the similarity among categorical values for each categorical feature
            sim_cat_vals_each_feature = {}
            cat_vals_keys = np.fromiter(store_prob_freq.keys(), dtype=np.int)
            for j in range(len(cat_vals_keys)):
                sim_cat_vals_each_feature[hashing(cat_vals_keys[j], DEFAULT_CATEGORICAL_VALUE)] = 0
                for k in range(j, len(cat_vals_keys)):
                    sim_cat_vals_each_feature[hashing(cat_vals_keys[j], cat_vals_keys[k])] = np.linalg.norm(store_prob_freq[cat_vals_keys[j]] - store_prob_freq[cat_vals_keys[k]])    

            max_val = max(sim_cat_vals_each_feature.values())
            sim_cat_vals_each_feature = {k: v/max_val for k, v in sim_cat_vals_each_feature.items()}
            similarity_of_cat_vals[i] = sim_cat_vals_each_feature
        else:
            sim_cat_vals_each_feature = {}
            sim_cat_vals_each_feature[hashing(cat_vals[0], cat_vals[0])] = 0
            sim_cat_vals_each_feature[hashing(cat_vals[1], cat_vals[1])] = 0
            sim_cat_vals_each_feature[hashing(cat_vals[0], DEFAULT_CATEGORICAL_VALUE)] = 0
            sim_cat_vals_each_feature[hashing(cat_vals[1], DEFAULT_CATEGORICAL_VALUE)] = 0
            sim_cat_vals_each_feature[hashing(cat_vals[0], cat_vals[1])] = 1
            similarity_of_cat_vals[i] = sim_cat_vals_each_feature
    
    return similarity_of_cat_vals


def predict_freq_cat_feature_manhanttan(V, W, E, F, C, Xl, Xu, X_cat, similarity_of_cat_vals, g=1):
    """
    Predict class labels for samples in the form of hyperboxes with continuous
    features represented by low bounds `Xl` and upper bounds `Xu` and categorical
    features stored in `X_cat`. The predicted results will be computed from
    existing hyperboxes with continuous features matrices for lower bounds `V`
    and upper bounds `W` and categorical features matrices for lower bounds `E`
    and upper bounds `F`.

    .. note::

        In the case there are many winner hyperboxes representing different
        class labels but with the same membership value with respect to the
        input pattern :math:`X_i` in the form of an hyperbox represented by a
        lower bound :math:`Xl_i` and an upper bound :math:`Xu_i` for continous
        features and a matrix :math:`Xcat_i` for categorical features, an
        additional criterion based on the minimum Manhattan distance between
        the central point of continous features in the input hyperbox
        :math:`X_i = [Xl_i, Xu_i]` and the central points of continous features
        in winner hyperboxes are used to find the final winner hyperbox that
        its class label is used for predicting the class label of the input
        hyperbox :math:`X_i`.

    .. warning::

        Another important point to pay attention is that the categorical
        features storing in :math:`X_cat` need to be encoded by using the
        function :func:`ordinal_encode_categorical_features` before pushing the
        values to this method.

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
    E : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all lower bounds for all categorical features of all
        hyperboxes of a trained hyperbox-based model, in which each row is a
        lower bound for categorical features of a hyperbox.
    F : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all upper bounds for all categorical features of all
        hyperboxes of a trained hyperbox-based model, in which each row is a
        upper bound for categorical features of a hyperbox.
    C : array-like of shape (n_hyperboxes,)
        An array contains all class lables for all hyperboxes of a trained
        hyperbox-based model.
    Xl : array-like of shape (n_samples, n_continuous_features)
        The data matrix contains lower bounds for continuous features of input
        patterns for which we want to predict the targets.
    Xu : array-like of shape (n_samples, n_continuous_features)
        The data matrix contains upper bounds for continuous features of input
        patterns for which we want to predict the targets.
    X_cat : array-like of shape (n_samples, n_cat_features)
        The data matrix contains categorical bounds for categorical features
        of input patterns for which we want to predict the targets.
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical
        values for each categorical feature index. Each element in this array
        is an dictionary with keys being a hashed value of two categorical
        values and values of this dictionary being a similarity value.
    g : float or array-like of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous dimension.

    Returns
    -------
    y_pred : array-like of shape (n_samples,)
        Predicted class labels for all input patterns.

    """
    if Xl is not None:
        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
        if Xu.ndim == 1:
            Xu = Xu.reshape(1, -1)
        if is_contain_missing_value(Xl) == True or is_contain_missing_value(Xu) == True:
            Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

    if X_cat is not None:
        if X_cat.ndim == 1:
            X_cat = X_cat.reshape(1, -1)

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
    np.random.seed(0)
    for i in range(n_samples):
        if (Xl is not None) and (X_cat is not None):
            if not is_exist_missing_continous_value:
                mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], X_cat[i], V, W, E, F, similarity_of_cat_vals, g)
            else:
                mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], X_cat[i], np.minimum(V, W), np.maximum(W, V), E, F, similarity_of_cat_vals, g)
        else:
            if Xl is not None:
                if not is_exist_missing_continous_value:
                    mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], None, V, W, E, F, similarity_of_cat_vals, g)
                else:
                    mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], None, np.minimum(V, W), np.maximum(W, V), E, F, similarity_of_cat_vals, g)
            else:
                mem_vals = membership_func_freq_cat_gfmm(None, None, X_cat[i], V, W, E, F, similarity_of_cat_vals, g)

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
                    # Find all average points of all hyperboxes with the same membership value
                    avg_point_mat = (V[max_mem_box_ids] + W[max_mem_box_ids]) / 2
                    # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                    maht_dist = manhattan_distance(avg_point_mat, Xg_mat)

                id_min_dist = maht_dist.argmin()
                y_pred[i] = C[max_mem_box_ids[id_min_dist]]
        else:
            y_pred[i] = C[max_mem_box_ids[0]]

    return y_pred


class FreqCatOnlineGFMM(BaseHyperboxClassifier):
    """Batch Incremental learning algorithm with mixed-attribute data for a
    general fuzzy min-max neural network, in which categorical features are
    encoded using the ordinal encoding method and the similarity degrees among
    categorical values are computed using their frequency of occurence with
    respect to all class labels in a training set.

    This algorithm uses a distance measure between any two values of a categorical
    variable based on the occurrence probability of such categorical values with
    respect to the values of the class variable. This distance is then normalised
    and used to compute the membership values for categorical features in conjunction
    with membership values of continuous features to generate the final membership
    values for mixed-attribute data.

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
    eta : float, optional, default=0.5
        Maximum hyperbox size for the categorical features.
    alpha : float, optional, default=0.9
        Multiplier factor to reduce the value of maximum hyperbox size after 
        each training loop.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all minimal points for continuous features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all maximal points for continuous features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    E : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all lower bounds for categorical features of all
        existing hyperboxes, in which each row is a lower bound of a hyperbox.
    F : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all upper bounds for categorical features of all
        existing hyperboxes, in which each row is an upper bound of a hyperbox.
    C : array-like of shape (n_hyperboxes,)
        A vector stores all class labels correponding to existing hyperboxes.

    Attributes
    ----------
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical values
        for each categorical feature index. Each element in this array is an dictionary
        with keys being a hashed value of two categorical values and values of this
        dictionary being a similarity value.
    categorical_features_ : int array of shape (n_cat_features,)
        Indices of categorical features in the training data and hyperboxes.
    continuous_features_ : int array of shape (n_continuous_features,)
        Indices of continuous features in the training data and hyperboxes.
    encoder_ : sklearn.preprocessing.OrdinalEncoder
        An ordinal encoder was used to encode categorical features.
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
    >>> from hbbrain.mixed_data.freq_cat_onln_gfmm import FreqCatOnlineGFMM
    >>> from hbbrain.datasets import load_japanese_credit
    >>> X, y = load_japanese_credit()
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> numerical_features = [1, 2, 7, 10, 13, 14]
    >>> categorical_features = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    >>> scaler.fit(X[:, numerical_features])
    MinMaxScaler()
    >>> X[:, numerical_features] = scaler.transform(X[:, numerical_features])
    >>> clf = FreqCatOnlineGFMM(theta=0.1, eta=0.6)
    >>> clf.fit(X, y, categorical_features)
    >>> print("Number of hyperboxes = %d"%clf.get_n_hyperboxes())
    Number of hyperboxes = 416
    >>> clf.predict(X[[10, 100]])
    array([1, 0])

    """

    def __init__(self, theta=0.5, theta_min=1, eta=0.5, gamma=1, alpha=0.9, V=None, W=None, E=None, F=None, C=None):
        BaseHyperboxClassifier.__init__(self, theta, False, V, W, C)
        if E is not None:
            self.E = E
        else:
            self.E = np.array([])
        if F is not None:
            self.F = F
        else:
            self.F = np.array([])
        self.gamma = gamma
        self.theta_min = theta_min
        self.alpha = alpha
        self.eta = eta

    def _validate_data(self):
        """
        Validate the initial values for parameters and initialise default
        values for parameters.

        Returns
        -------
        None.

        """
        if self.theta > 1:
            self.theta = 1

        if (self.theta_min > self.theta):
            self.theta_min = self.theta

        if self.eta > 1:
            self.eta = 1

        self._init_hyperboxes()
        if self.E is None:
            self.E = np.array([])
        if self.F is None:
            self.F = np.array([])

    def is_satisfied_cat_expansion_conds(self, Ej, Fj, x_cat):
        """
        Check whether the expansion condition for categorical features `x_cat`
        of an input pattern can be covered by categorical bounds of the
        hyperbox `Bj` with the categorical features stored in the lower bound
        `Ej` and the upper bound `Fj`.

        Parameters
        ----------
        Ej : array-like of shape (n_cat_features,)
            Lower bound of categorical features in the hyperbox `Bj` which can
            be extended to cover the input pattern.
        Fj : array-like of shape (n_cat_features,)
            Upper bound of categorical features in the hyperbox `Bj` which can
            be extended to cover the input pattern.
        x_cat : array-like of shape (n_cat_features,)
            Categorical features of an input pattern.

        Returns
        -------
        bool
            If True, the categorical features in `Dj` are satisfied with the
            expansion conditions for the categorical feature so that it can be
            expanded to cover the input pattern. Otherwise, the conditions for
            the categorical features are not met.

        """
        n_cat_features = len(Ej)
        for i in range(n_cat_features):
            if (x_cat[i] != Ej[i]) and (x_cat[i] != Fj[i]):
                if (Ej[i] == DEFAULT_CATEGORICAL_VALUE) and (Fj[i] == DEFAULT_CATEGORICAL_VALUE):
                    return True
                else:
                    if Ej[i] != DEFAULT_CATEGORICAL_VALUE:
                        if Fj[i] == DEFAULT_CATEGORICAL_VALUE:
                            if self.similarity_of_cat_vals[i][hashing(Ej[i], x_cat[i])] > self.eta:
                                return False
                        else:
                            cur_dist = self.similarity_of_cat_vals[i][hashing(Ej[i], Fj[i])]
                            change_low_dist = self.similarity_of_cat_vals[i][hashing(Ej[i], x_cat[i])]
                            change_up_dist = self.similarity_of_cat_vals[i][hashing(Fj[i], x_cat[i])]
                            if change_low_dist <= cur_dist and change_up_dist <= cur_dist:
                                return False
                            else:
                                if change_low_dist > self.eta and change_up_dist > self.eta:
                                    return False

        return True

    def fit(self, X, y, categorical_features=None):
        """
        Build a general fuzzy min-max neural network from the training set
        (X, y) using the original incremental learning algorithm, in which
        categorical features are encoded using the ordinal encoding method and
        the similarity among categorical values are computed using their frequency
        of occurence with respect to all class labels in a training set.

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
            X, self.encoder_ = ordinal_encode_categorical_features(X, categorical_features)
            X_cat = X[:, categorical_features]
            
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
                    X_cat = X_cat[:n_samples, :]
                    self.similarity_of_cat_vals = compute_similarity_among_categorical_values(X_cat, y)
                    return self._fit(Xl, Xu, X_cat, y)
            else:
                if categorical_features is None:
                    return self._fit(X_con, X_con, None, y)
                else:
                    self.similarity_of_cat_vals = compute_similarity_among_categorical_values(X_cat, y)
                    return self._fit(X_con, X_con, X_cat, y)
        else:
            self.continuous_features_ = None
            self.similarity_of_cat_vals = compute_similarity_among_categorical_values(X_cat, y)
            return self._fit(None, None, X_cat, y)

    def _fit(self, Xl, Xu, X_cat, y, is_compute_similarity_cat_vals=False):
        """
        Build a general fuzzy min-max neural network from the training set
        using the original incremental learning algorithm, in which
        categorical features are encoded using the ordinal encoding method and
        the similarity among categorical values are computed using their frequency
        of occurence with respect to all class labels in a training set. Input training
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
        is_compute_similarity_cat_vals : boolean, optional, default=False
            Whether a matrix of similarity values among categorical values of
            all features is computed or not.

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
            n_samples = X_cat.shape[0]
            n_continuous_features = 0
            
        if X_cat is not None:
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(-1, 1)
            n_cat_features = X_cat.shape[1]
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
                
        if is_compute_similarity_cat_vals == True and X_cat is not None:
            self.similarity_of_cat_vals = compute_similarity_among_categorical_values(X_cat, y)

        theta = self.theta
        training_acc = 0
        self.n_passes = 0

        while theta >= self.theta_min and training_acc < 1:
            self.n_passes += 1
            threshold_mem_val = 1 - np.max(self.gamma) * theta
            # Loop through each training input pattern
            for i in range(n_samples):
                if (n_continuous_features > 0 and self.V.size == 0) or (n_cat_features > 0 and self.E.size == 0):
                    # no model provided, start from scratch
                    if Xl is not None:
                        self.V = np.array([Xl[i]])
                        self.W = np.array([Xu[i]])
                    if X_cat is not None:
                        self.E = X_cat[i].reshape(1, -1)
                        self.F = np.full((1, n_cat_features), DEFAULT_CATEGORICAL_VALUE)

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
                            E_sameX = self.E[id_same_input_label_group]
                            F_sameX = self.F[id_same_input_label_group]
                        else:
                            E_sameX = None
                            F_sameX = None

                        lb_sameX = self.C[id_same_input_label_group]
                        id_range = np.arange(len(self.C))
                        id_processing = id_range[id_same_input_label_group]
                        
                        if n_continuous_features > 0 and n_cat_features > 0:
                            if not self.is_exist_continuous_missing_value:
                                b = membership_func_freq_cat_gfmm(Xl[i], Xu[i], X_cat[i], V_sameX, W_sameX, E_sameX, F_sameX, self.similarity_of_cat_vals, self.gamma)
                            else:
                                b = membership_func_freq_cat_gfmm(Xl[i], Xu[i], X_cat[i], np.minimum(V_sameX, W_sameX), np.maximum(W_sameX, V_sameX), E_sameX, F_sameX, self.similarity_of_cat_vals, self.gamma)
                        else:
                            if n_continuous_features > 0:
                                if not self.is_exist_continuous_missing_value:
                                    b = membership_func_freq_cat_gfmm(Xl[i], Xu[i], None, V_sameX, W_sameX, E_sameX, F_sameX, self.similarity_of_cat_vals, self.gamma)
                                else:
                                    b = membership_func_freq_cat_gfmm(Xl[i], Xu[i], None, np.minimum(V_sameX, W_sameX), np.maximum(W_sameX, V_sameX), E_sameX, F_sameX, self.similarity_of_cat_vals, self.gamma)
                            else:
                                b = membership_func_freq_cat_gfmm(None, None, X_cat[i], V_sameX, W_sameX, E_sameX, F_sameX, self.sim_vec, self.gamma)

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
                                        if (((np.maximum(self.W[j], Xu[i]) - np.minimum(self.V[j], Xl[i])) <= theta).all() == True) and (self.is_satisfied_cat_expansion_conds(self.E[j], self.F[j], X_cat[i]) == True):
                                            is_met_expansion = True
                                    else:
                                        if (n_continuous_features > 0) and (((np.maximum(self.W[j], Xu[i]) - np.minimum(self.V[j], Xl[i])) <= theta).all() == True):
                                            is_met_expansion = True
                                        if (n_cat_features > 0) and (self.is_satisfied_cat_expansion_conds(self.E[j], self.F[j], X_cat[i]) == True):
                                            is_met_expansion = True

                                    if is_met_expansion == True:
                                        # adjust the j-th hyperbox
                                        if n_continuous_features > 0:
                                            self.V[j] = np.minimum(self.V[j], Xl[i])
                                            self.W[j] = np.maximum(self.W[j], Xu[i])
                                        if n_cat_features > 0:
                                            for tt in range(n_cat_features):
                                                if (self.E[j, tt] == DEFAULT_CATEGORICAL_VALUE) and (self.F[j, tt] == DEFAULT_CATEGORICAL_VALUE):
                                                    self.E[j, tt] = X_cat[i, tt]
                                                else:
                                                    if self.F[j, tt] == DEFAULT_CATEGORICAL_VALUE:
                                                        self.F[j, tt] = X_cat[i, tt]
                                                    else:
                                                        cur_dist = self.similarity_of_cat_vals[tt][hashing(self.E[j, tt], self.F[j, tt])]
                                                        expand_up_dist = self.similarity_of_cat_vals[tt][hashing(self.E[j, tt], X_cat[i, tt])]
                                                        if (expand_up_dist > cur_dist) and (expand_up_dist <= self.eta):
                                                            self.F[j, tt] = X_cat[i, tt]
                                                        else:
                                                            self.E[j, tt] = X_cat[i, tt]
                                        
                                        id_of_winner_hyperbox = j
                                        adjust = True
                                        if y[i] != UNLABELED_CLASS and self.C[j] == UNLABELED_CLASS:
                                            self.C[j] = y[i]

                                        break

                            # if i-th sample did not fit into any existing box, create a new one
                            if not adjust:
                                if n_continuous_features > 0:
                                    self.V = np.concatenate((self.V, Xl[i].reshape(1, -1)), axis = 0)
                                    self.W = np.concatenate((self.W, Xu[i].reshape(1, -1)), axis = 0)
                                self.C = np.concatenate((self.C, [y[i]]))
                                if n_cat_features > 0:
                                    self.E = np.vstack((self.E, X_cat[i]))
                                    X_f_tmp = np.full((1, n_cat_features), DEFAULT_CATEGORICAL_VALUE)
                                    self.F = np.vstack((self.F, X_f_tmp))
                            else:
                                id_overlap_box_resolve_only_cat = []
                                for ii in range(len(self.C)):
                                    if ii != id_of_winner_hyperbox and (self.C[ii] != self.C[id_of_winner_hyperbox] or self.C[id_of_winner_hyperbox] == UNLABELED_CLASS):
                                        if n_cat_features > 0:                                       
                                            dim_cat = hyperbox_overlap_test_freq_cat_gfmm(self.E, self.F, id_of_winner_hyperbox, ii, X_cat, self.similarity_of_cat_vals, id_overlap_box_resolve_only_cat)
                                            if (n_continuous_features > 0) and (self.V.shape[0] > 1):
                                                if (len(dim_cat) > 0):
                                                    # overlap for both types of features => need to resolve
                                                    if dim_cat[0] != -1 and (dim_cat[1][0] is not None or dim_cat[1][1] is not None):
                                                        # resolve overlap in cat feature
                                                        self.E[id_of_winner_hyperbox], self.F[id_of_winner_hyperbox] = hyperbox_contraction_freq_cat_gfmm(self.E[id_of_winner_hyperbox], self.F[id_of_winner_hyperbox], dim_cat)
                                                        id_overlap_box_resolve_only_cat.append(ii)
                                                    else:
                                                        # Doing overlap resolving for continuous features
                                                        is_overlap = is_two_hyperboxes_overlap_num_data_general(
                                                            self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii])
                                                        if is_overlap == True:
                                                            self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii] = overlap_resolving_num_data(
                                                                self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.C[id_of_winner_hyperbox], self.V[ii], self.W[ii], self.C[ii])
                                                else:
                                                    # No overlap in cat features, but if overlap in continous features then we need to keep the id of this hypebox
                                                    # because when resolve overlap for cat feature by changing its values in the future, maybe we reverse overlap
                                                    is_overlap = is_two_hyperboxes_overlap_num_data_general(
                                                        self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii])
                                                    if is_overlap == True:
                                                        id_overlap_box_resolve_only_cat.append(ii)
                                            else:
                                                if (len(dim_cat) > 0) and (dim_cat[0] != -1): 
                                                    self.E[id_of_winner_hyperbox], self.F[id_of_winner_hyperbox] = hyperbox_contraction_freq_cat_gfmm(self.E[id_of_winner_hyperbox], self.F[id_of_winner_hyperbox], dim_cat)
                                                    id_overlap_box_resolve_only_cat.append(ii)
                                        else:
                                            is_overlap = is_two_hyperboxes_overlap_num_data_general(
                                                self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii])
                                            if is_overlap == True:
                                                self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii] = overlap_resolving_num_data(
                                                    self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.C[id_of_winner_hyperbox], self.V[ii], self.W[ii], self.C[ii])

                    else:
                        if n_continuous_features > 0:
                            self.V = np.concatenate((self.V, Xl[i].reshape(1, -1)), axis = 0)
                            self.W = np.concatenate((self.W, Xu[i].reshape(1, -1)), axis = 0)
                        self.C = np.concatenate((self.C, [y[i]]))
                        if n_cat_features > 0:
                            self.E = np.vstack((self.E, X_cat[i]))
                            X_f_tmp = np.full((1, n_cat_features), DEFAULT_CATEGORICAL_VALUE)
                            self.F = np.vstack((self.F, X_f_tmp))

            if n_continuous_features > 0:
                theta = theta * self.alpha
                if theta >= self.theta_min:
                    y_pred = self._predict(Xl, Xu, X_cat)
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
            X, _ = ordinal_encode_categorical_features(X, self.categorical_features_, self.encoder_)
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                X_con = X[:, self.continuous_features_].astype(float)
            else:
                X_con = None
            X_cat = X[:, self.categorical_features_]
            y_pred = self._predict(X_con, X_con, X_cat)
        else:
            y_pred = self._predict(X, X, None)

        return y_pred

    def _predict(self, Xl, Xu, X_cat):
        """
        Predict class labels for samples in the form of hyperboxes represented
        by low bounds `Xl` and upper bounds `Xu`.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i` in the form of an hyperbox represented by
            a lower bound :math:`Xl_i` and an upper bound :math:`Xu_i` for
            continous features and a matrix :math:`Xcat_i` for categorical
            features, an additional criterion based on the minimum Manhattan
            distance between the central point of continous features in the
            input hyperbox :math:`X_i = [Xl_i, Xu_i]` and the central points of
            continous features in the winner hyperboxes are used to find the
            final winner hyperbox that its class label is used for predicting
            the class label of the input hyperbox :math:`X_i`.

        .. warning::

            Another important point to pay attention is that the categorical
            features storing in :math:`X_cat` need to be encoded by using the
            function :func:`ordinal_encode_categorical_features` before pushing
            the values to this method.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains the lower bounds of input patterns
            for which we want to predict the targets.
        Xu : array-like of shape (n_samples, n_continuous_features)
            The data matrix contains the upper bounds of input patterns
            for which we want to predict the targets.
        X_cat : array-like of shape (n_samples, n_cat_features)
            The data matrix contains  the bounds for categorical features
            of input patterns for which we want to predict the targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        y_pred = predict_freq_cat_feature_manhanttan(self.V, self.W, self.E, self.F, self.C, Xl, Xu, X_cat, self.similarity_of_cat_vals, self.gamma)

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
            X, _ = ordinal_encode_categorical_features(X, self.categorical_features_, self.encoder_)
            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                X_con = X[:, self.continuous_features_].astype(float)
            else:
                X_con = None
            X_cat = X[:, self.categorical_features_]
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
            if Xu.ndim == 1:
                Xu = Xu.reshape(1, -1)
            if is_contain_missing_value(Xl) == True or is_contain_missing_value(Xu) == True:
                Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

        if X_cat is not None:
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(1, -1)

        mem_vals, _ = get_membership_freq_cat_gfmm_all_classes(Xl, Xu, X_cat, self.V, self.W, self.E, self.F, self.C, self.similarity_of_cat_vals, self.gamma)

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
            X_val, _ = ordinal_encode_categorical_features(X_val, self.categorical_features_, self.encoder_)
            Xcat_val = X_val[:, self.categorical_features_]

            if (self.continuous_features_ is not None) and (len(self.continuous_features_) > 0):
                Xval_con = X_val[:, self.continuous_features_].astype(float)
                if Xval_con.shape[0] > n_val_samples:
                    Xl_val = Xval_con[:n_val_samples, :]
                    Xu_val = Xval_con[n_val_samples:, :]
                    Xcat_val = Xcat_val[:n_val_samples, :]
                    self._simple_pruning(Xl_val, Xu_val, Xcat_val, y_val, acc_threshold, keep_empty_boxes)
                else:
                    self._simple_pruning(Xval_con, Xval_con, Xcat_val, y_val, acc_threshold, keep_empty_boxes)
            else:
                # No continous features
                self._simple_pruning(None, None, Xcat_val, y_val, acc_threshold, keep_empty_boxes)
        else:
            # Handle the case of no categorical features
            if Xval_con.shape[0] > n_val_samples:
                Xl_val = Xval_con[:n_val_samples, :]
                Xu_val = Xval_con[n_val_samples:, :]
                self._simple_pruning(Xl_val, Xu_val, None, y_val, acc_threshold, keep_empty_boxes)
            else:
                self._simple_pruning(Xval_con, Xval_con, None, y_val, acc_threshold, keep_empty_boxes)

        return self

    def _simple_pruning(self, Xl_val, Xu_val, Xcat_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Private function for simply pruning low qualitied hyperboxes based on
        a pre-defined accuracy threshold for each hyperbox. This method handles
        the case of continuous features under the form of hyperboxes.

        Parameters
        ----------
        Xl_val : array-like of shape (n_samples, n_con_features)
            The data matrix contains lower bounds of continous features in
            validation patterns.
        Xu_val : array-like of shape (n_samples, n_con_features)
            The data matrix contains upper bounds of continous features in
            validation patterns.
        Xcat_val : array-like of shape (n_samples, n_cat_features)
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
            n_samples = Xcat_val.shape[0]

        rnd = np.random
        rnd.seed(0)
        # Matrix stores the classification accuracy for each created hyperbox
        # in the trained model. The first column stores the number of corrected
        # classification samples and the second column stores the number of
        # wrong classification samples
        hyperboxes_performance = np.zeros((len(self.C), 2))

        for i in range(n_samples):
            if (Xl_val is not None) and (Xcat_val is not None):
                # calculate memberships for all hyperboxes
                if not self.is_exist_continuous_missing_value:
                    mem_vals = membership_func_freq_cat_gfmm(Xl_val[i], Xu_val[i], Xcat_val[i], self.V, self.W, self.E, self.F, self.similarity_of_cat_vals, self.gamma)
                else:
                    mem_vals = membership_func_freq_cat_gfmm(Xl_val[i], Xu_val[i], Xcat_val[i], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.E, self.F, self.similarity_of_cat_vals, self.gamma)
            else:
                if Xl_val is not None:
                    if not self.is_exist_continuous_missing_value:
                        mem_vals = membership_func_freq_cat_gfmm(Xl_val[i], Xu_val[i], None, self.V, self.W, self.E, self.F, self.similarity_of_cat_vals, self.gamma)
                    else:
                        mem_vals = membership_func_freq_cat_gfmm(Xl_val[i], Xu_val[i], None, np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.E, self.F, self.similarity_of_cat_vals, self.gamma)
                else:
                    mem_vals = membership_func_freq_cat_gfmm(None, None, Xcat_val[i], self.V, self.W, self.E, self.F, self.similarity_of_cat_vals, self.gamma)

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
            if Xcat_val is not None:
                self.E = self.E[id_remained_incl_empty_boxes]
                self.F = self.F[id_remained_incl_empty_boxes]
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

            if Xcat_val is not None:
                E_pruned_excl_empty_boxes = self.E[id_remained_excl_empty_boxes]
                E_pruned_incl_empty_boxes = self.E[id_remained_incl_empty_boxes]
                F_pruned_excl_empty_boxes = self.F[id_remained_excl_empty_boxes]
                F_pruned_incl_empty_boxes = self.F[id_remained_incl_empty_boxes]
            else:
                E_pruned_excl_empty_boxes = None
                E_pruned_incl_empty_boxes = None
                F_pruned_excl_empty_boxes = None
                F_pruned_incl_empty_boxes = None

            C_pruned_excl_empty_boxes = self.C[id_remained_excl_empty_boxes]
            C_pruned_incl_empty_boxes = self.C[id_remained_incl_empty_boxes]

            y_val_pred_excl_empty_boxes = predict_freq_cat_feature_manhanttan(
                V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes,
                E_pruned_excl_empty_boxes, E_pruned_excl_empty_boxes,
                C_pruned_excl_empty_boxes, Xl_val, Xu_val, Xcat_val,
                self.similarity_of_cat_vals, self.gamma)
            y_val_pred_incl_empty_boxes = predict_freq_cat_feature_manhanttan(
                V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes,
                E_pruned_incl_empty_boxes, F_pruned_incl_empty_boxes,
                C_pruned_incl_empty_boxes, Xl_val, Xu_val, Xcat_val,
                self.similarity_of_cat_vals, self.gamma)

            if (accuracy_score(y_val, y_val_pred_excl_empty_boxes) >= accuracy_score(y_val, y_val_pred_incl_empty_boxes)):
                if Xl_val is not None:
                    self.V = V_pruned_excl_empty_boxes
                    self.W = W_pruned_excl_empty_boxes
                if Xcat_val is not None:
                    self.E = E_pruned_excl_empty_boxes
                    self.F = F_pruned_excl_empty_boxes
                self.C = C_pruned_excl_empty_boxes
            else:
                if Xl_val is not None:
                    self.V = V_pruned_incl_empty_boxes
                    self.W = W_pruned_incl_empty_boxes
                if Xcat_val is not None:
                    self.E = E_pruned_incl_empty_boxes
                    self.F = F_pruned_incl_empty_boxes
                self.C = C_pruned_incl_empty_boxes

        return self

    def get_n_hyperboxes(self):
        """
        Get number of hyperboxes in the trained hyperbox-based model

        Returns
        -------
        int
            Number of hyperboxes in the trained hyperbox-based classifier.

        """
        if self.categorical_features_ is not None:
            return self.E.shape[0]
        else:
            return self.V.shape[0]
        
    def get_sample_explanation(self, x):
        """
        Get useful information for explaining the reason behind the predicted
        result for the input pattern represented by upper and lower bounds for
        continous features together with the lower and upper bounds for the
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
        dict_min_point_cat_classes: dictionary
            A dictionary stores all lower bounds of categorical features for
            the hyperboxes having the maximum membership value for each class.
            The key is the class label and the value is the lower bound of
            categorical features for the hyperboxes corresponding to each class.
        dict_max_point_cat_classes: dictionary
            A dictionary stores all upper bounds of categorical features for
            the hyperboxes having the maximum membership value for each class.
            The key is the class label and the value is the upper bound of
            categorical features for the hyperboxes corresponding to each class.

        """
        if self.categorical_features_ is not None:
            x = ordinal_encode_categorical_features(x.reshape(1, -1), self.categorical_features_, self.encoder_)[0][0]
            x_cat = x[self.categorical_features_]
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
        continous features together with the lower and upper bounds for
        categorical features.

        .. note::

            The categorical features storing in `x_cat` need to be encoded by
            using the function :func:`ordinal_encode_categorical_features`
            before pushing the values to this method.

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
        dict_min_point_cat_classes: dictionary
            A dictionary stores all lower bounds of categorical features for
            the hyperboxes having the maximum membership value for each class.
            The key is the class label and the value is the lower bound of
            categorical features for the hyperboxes corresponding to each class.
        dict_max_point_cat_classes: dictionary
            A dictionary stores all upper bounds of categorical features for
            the hyperboxes having the maximum membership value for each class.
            The key is the class label and the value is the upper bound of
            categorical features for the hyperboxes corresponding to each class.

        """
        mem_vals_for_classes, hyperbox_id_for_classes = get_membership_freq_cat_gfmm_all_classes(xl, xu, x_cat, self.V, self.W, self.E, self.F, self.C, self.similarity_of_cat_vals, self.gamma)
        class_values = np.unique(self.C)
        # get predicted class label for the input sample
        y_pred = self._predict(xl, xu, x_cat)[0]
        # create dictionaries with keys being class labels and values being membership values, maximum and minimum points
        dict_mem_val_classes = {}
        dict_min_point_classes = {}
        dict_max_point_classes = {}
        dict_min_point_cat_classes = {}
        dict_max_point_cat_classes = {}
        for _id, c in enumerate(class_values):
            dict_mem_val_classes[c] = mem_vals_for_classes[0][_id]
            box_id = hyperbox_id_for_classes[0][_id]
            if xl is not None:
                dict_min_point_classes[c] = self.V[box_id]
                dict_max_point_classes[c] = self.W[box_id]
            if x_cat is not None:
                dict_min_point_cat_classes[c] = self.E[box_id]
                dict_max_point_cat_classes[c] = self.F[box_id]

        return (y_pred, dict_mem_val_classes, dict_min_point_classes, dict_max_point_classes, dict_min_point_cat_classes, dict_max_point_cat_classes)


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
    optional.add_argument('--eta', type=float, default=0.5,
                          help='Maximum similarity value for each pair of categorical values (in the range of (0, 1] (default: 0.5')
    optional.add_argument('--gamma', type=float, default=1,
                          help='A sensitivity parameter describing the speed of decreasing of the membership function in each continuous dimension (larger than 0) (default: 1)')
    optional.add_argument('--alpha', type=float, default=0.9,
                          help='Multiplier showing the decrease of theta in each step (default: 0.9)')

    args = parser.parse_args()

    if args.theta <= 0 or args.theta > 1:
        parser.error("--theta has to be in the range of (0, 1]")

    if args.theta_min <= 0 or args.theta_min > 1:
        parser.error("--theta_min has to be in the range of (0, 1]")

    if args.eta <= 0 or args.eta > 1:
        parser.error("--eta has to be in the range of (0, 1]")

    if args.alpha <= 0 or args.alpha >= 1:
        parser.error("--alpha has to be in the range of (0, 1)")

    if args.gamma <= 0:
        parser.error("--gamma has to be larger than 0")

    gamma = args.gamma
    theta = args.theta
    theta_min = args.theta_min
    alpha = args.alpha
    eta = args.eta
    training_file = args.training_file
    testing_file = args.testing_file

    import json
    import pandas as pd

    categorical_features = json.loads(args.categorical_features)

    df_train = pd.read_csv(training_file, header=None, na_values=pd._libs.parsers.STR_NA_VALUES)
    df_test = pd.read_csv(testing_file, header=None, na_values=pd._libs.parsers.STR_NA_VALUES)

    Xy_train = df_train.to_numpy()
    Xy_test = df_test.to_numpy()

    Xtr = Xy_train[:, :-1]
    ytr = Xy_train[:, -1].astype(int)

    Xtest = Xy_test[:, :-1]
    ytest = Xy_test[:, -1].astype(int)

    freq_cat_onln_gfmm_clf = FreqCatOnlineGFMM(theta=theta, theta_min=theta_min, eta=eta, gamma=gamma, alpha=alpha)
    freq_cat_onln_gfmm_clf.fit(Xtr, ytr, categorical_features)
    print('Number of hyperboxes = %d'%freq_cat_onln_gfmm_clf.get_n_hyperboxes())
    
    y_pred = freq_cat_onln_gfmm_clf.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # sample_need_explain = 10
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes, dict_min_point_cat_classes, dict_max_point_cat_classes = freq_cat_onln_gfmm_clf.get_sample_explanation(Xtest[sample_need_explain])
    # print("Explain samples:")
    # print("Membership values for classes: ", mem_val_classes)
    # print("Predicted class = ", y_pred_input_0)
    # print("Minimum continuous points of the selected hyperbox for each class: ", min_points_classes)
    # print("Maximum continuous points of the selected hyperbox for each class: ", max_points_classes)
    # print("Minimum categorical points of the selected hyperbox for each class: ", dict_min_point_cat_classes)
    # print("Maximum categorical points of the selected hyperbox for each class: ", dict_max_point_cat_classes)
    
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/japanese_credit_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # freq_cat_onln_gfmm_clf.simple_pruning(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%freq_cat_onln_gfmm_clf.get_n_hyperboxes())
    
    # y_pred_2 = freq_cat_onln_gfmm_clf.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy (using a probability measure for samples on the boundary) = {acc_pruned * 100: .2f}%')
    