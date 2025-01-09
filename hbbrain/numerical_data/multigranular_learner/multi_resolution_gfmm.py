"""
A multi-resolution hierarchical granular representation based classifier using
general fuzzy min-max neural network.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from hbbrain.base.base_estimator import BaseHyperboxClassifier
from hbbrain.base.base_gfmm_estimator import (
    convert_format_missing_input_zero_one,
    is_contain_missing_value
)
from hbbrain.numerical_data.multigranular_learner.base_granular_learner import BaseGranular
from hbbrain.utils.membership_calc import (
    membership_func_gfmm,
    get_membership_gfmm_all_classes
)
from hbbrain.utils.adjust_hyperbox import (
    is_overlap_one_many_hyperboxes_num_data_general,
    is_two_hyperboxes_overlap_num_data_general,
    overlap_resolving_num_data
)
from hbbrain.utils.drawing_func import get_cmap, draw_box
from hbbrain.constants import (
    UNLABELED_CLASS,
    HOMOGENEOUS_CLASS_LEARNING, HETEROGENEOUS_CLASS_LEARNING,
)
from hbbrain.utils.drawing_func import (
    generate_grid_decision_boundary_2D,
    draw_decision_boundary_2D,
)


def convert_granular_theta_to_level(granular_thetas):
    """
    Convert a list of maximum hyperbox sizes to the corresponding granular
    levels.

    Parameters
    ----------
    granular_thetas : list
        A list contains all maximum hyperbox sizes for all granularity levels.

    Returns
    -------
    level_dic : dict
        A mapping between the maximum hyperbox size and the granular level.

    """
    level_dic = {}
    for i, val in enumerate(granular_thetas):
        level_dic[val] = i
    return level_dic


def remove_contained_hyperboxes(V, W, C, N_samples, Centroids):
    """
    Remove all hyperboxes contained in other hyperboxes with the same class
    label and update the centroids of larger hyperboxes included the removed
    hyperboxes.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points for numerical features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximum points for numerical features of all
        existing hyperboxes, in which each row is a maximum point of a hyperbox.
    C : array-like of shape (n_hyperboxes,)
        A vector stores all class labels correponding to existing hyperboxes.
    N_samples : array-like of shape (n_hyperboxes,)
        A vector stores the number of samples fully included in each existing
        hyperbox.
    Centroids : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all centroid points of all existing hyperboxes, in
        which each row is a centroid point of a hyperbox.

    Returns
    -------
    new_V : array-like of shape (n_new_hyperboxes, n_features)
        A matrix stores all minimal points for numerical features of all
        hyperboxes after removal of fully contained hyperboxes, in which each
        row is a minimal point of a hyperbox.
    new_W : array-like of shape (n_new_hyperboxes, n_features)
        A matrix stores all maximal points for numerical features of all
        hyperboxes after removal of fully contained hyperboxes, in which each
        row is a maximal point of a hyperbox.
    new_C : array-like of shape (n_new_hyperboxes,)
        A vector stores all class labels correponding to remaining hyperboxes
        after removal of fully contained hyperboxes.
    new_N_samples : array-like of shape (n_new_hyperboxes,)
        A vector stores the number of samples fully included in each hyperbox.
    new_Centroids : array-like of shape (n_new_hyperboxes, n_features)
        A matrix stores all centroid points of all remaining hyperboxes after
        removal of fully contained hyperboxes, in which each row is a centroid
        point of a hyperbox.
    n_removed_hyperboxes : int
        Numer of hyperboxes has been removed because they are included in at
        least one larger hyperbox with the same class label.

    """
    n_hyperboxes = len(C)
    # an array of indices showing the position of all hyperboxes kept
    ids_kept_boxes = np.ones(n_hyperboxes, dtype=bool)
    n_removed_hyperboxes = 0
    for i in range(n_hyperboxes):
        # Filter hypeboxes with the sample label as hyperbox i
        id_hyperbox_same_label = C == C[i]
        id_hyperbox_same_label[i] = False # remove hyperbox i
        if id_hyperbox_same_label.any() == True:
            # exist at least one hyperbox with the same label as hyperbox i
            V_same = V[id_hyperbox_same_label]
            W_same = W[id_hyperbox_same_label]

            mem_vals = membership_func_gfmm(V[i], W[i], V_same, W_same, 1)
            equal_one_index = mem_vals == 1

            if np.sum(equal_one_index) > 0:
                original_index = np.arange(0, n_hyperboxes)
                original_index_same_label = original_index[id_hyperbox_same_label]
                # Find indices of hyperboxes that contain hyperbox i
                ids_parent_hyperboxes = original_index_same_label[np.nonzero(equal_one_index)[0]]

                is_exist_parent_hyperboxes = len(ids_parent_hyperboxes) > 0

                if is_exist_parent_hyperboxes == True:
                    ids_kept_boxes[i] = False
                    
                    # Update centroid of a larger parent hyperbox
                    if len(ids_parent_hyperboxes) == 1:
                        parent_selection = ids_parent_hyperboxes[0]
                        if ids_kept_boxes[ids_parent_hyperboxes[0]] == False:
                            ids_kept_boxes[i] = True
                    elif len(ids_parent_hyperboxes) > 1:
                        start_id = 0
                        while start_id < len(ids_parent_hyperboxes) and ids_kept_boxes[ids_parent_hyperboxes[start_id]] == False:
                            start_id = start_id + 1 # remove cases that parent hyperboxes are merged
                        
                        if start_id < len(ids_parent_hyperboxes):
                            # Compute the distance from the centroid of hyperbox i to centroids of other hyperboxes and choose the hyperbox with the smallest distance to merge
                            min_dis = np.linalg.norm(Centroids[i] - Centroids[ids_parent_hyperboxes[start_id]])
                            parent_selection = ids_parent_hyperboxes[start_id]
                            for jj in range(start_id + 1, len(ids_parent_hyperboxes)):
                                if ids_kept_boxes[ids_parent_hyperboxes[jj]] == True:
                                    dist = np.linalg.norm(Centroids[i] - Centroids[ids_parent_hyperboxes[jj]])
                                    if min_dis < dist:
                                        min_dis = dist
                                        parent_selection = ids_parent_hyperboxes[jj]
                        else:
                            ids_kept_boxes[i] = True

                    # Merge centroids and number of hyperboxes
                    if ids_kept_boxes[i] == False:
                        n_removed_hyperboxes = n_removed_hyperboxes + 1
                        Centroids[parent_selection] = (N_samples[parent_selection] * Centroids[parent_selection] + N_samples[i] * Centroids[i]) / (N_samples[i] + N_samples[parent_selection])
                        N_samples[parent_selection] = N_samples[parent_selection] + N_samples[i]

    # remove hyperboxes contained in other hyperboxes
    new_V = V[ids_kept_boxes, :]
    new_W = W[ids_kept_boxes, :]
    new_C = C[ids_kept_boxes]
    new_Centroids = Centroids[ids_kept_boxes]
    new_N_samples = N_samples[ids_kept_boxes]

    return (new_V, new_W, new_C, new_N_samples, new_Centroids, n_removed_hyperboxes)


def predict_with_centroids(V, W, C, N_samples, Centroids, Xl, Xu, g=1):
    """
    Predict class labels for samples in `X` represented in the form of invervals
    `[Xl, Xu]`. This is a common function to determine the right class labels
    for X wrt. a trained hyperbox-based classifier represented by `[V, W, C]`.
    It uses the winner-takes-all principle to predict class labels for each
    sample in X by assigning the class label of the sample to the class 
    label of the hyperbox with the maximum membership value to that sample.
    It will use an Euclidean distance from the input pattern to the centroid
    point of the hyperbox in the case of many winner hyperboxes with different
    classes having the same maximum membership value. If two winner hyperboxes
    show the same Euclidean distance to their centroid points, the winner
    hyperbox with a higher number of included samples will be selected.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points for numerical features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points for numerical features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    C : array-like of shape (n_hyperboxes,)
        A vector stores all class labels correponding to existing hyperboxes.
    N_samples : array-like of shape (n_hyperboxes,)
        A vector stores the number of samples fully included in each existing
        hyperbox.
    Centroids : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all centroid points of all existing hyperboxes, in
        which each row is a centroid point of a hyperbox.
    Xl : array-like of shape (n_samples, n_features)
        The data matrix contains lower bounds of input patterns for which we
        want to predict the targets.
    Xu : array-like of shape (n_samples, n_features)
        The data matrix contains upper bounds of input patterns for which we
        want to predict the targets.
    g : float or array-like of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        A vector contains the predictions. In binary and multiclass problems,
        this is a vector containing `n_samples`.

    """
    if Xl.ndim == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)

    if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
        Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

    is_exist_missing_value = (V > W).any()

    n_samples = Xl.shape[0]
    y_pred = np.full(n_samples, 0)
    sample_id = 0
    for i in range(n_samples):
        sample_id += 1
        if is_exist_missing_value == False:
            mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], V, W, g) # calculate memberships for all hyperboxes
        else:
            mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], np.minimum(V, W), np.maximum(W, V), g) # calculate memberships for all hyperboxes

        bmax = mem_val.max() # get the maximum membership value

        if ((Xl[i] < 0).any() == True) or ((Xu[i] > 1).any() == True):
            print(">>> The testing sample %d with the coordinate %s is outside the range [0, 1]. Membership value = %f. The prediction is more likely incorrect." % (sample_id, Xl[i], bmax))

        # get indices of all hyperboxes with max membership
        max_mem_V_id = np.nonzero(mem_val == bmax)[0]

        if len(np.unique(C[max_mem_V_id])) == 1:
            # only one hyperbox with the highest membership value
            y_pred[i] = C[max_mem_V_id[0]]
        else:
            # at least one pair of hyperboxes with different class
            # => compare the centroid, and classify the input to the hyperboxes
            # with the nearest distance to the input pattern
            centroid_input_pat = (Xl[i] + Xu[i]) / 2
            id_min = max_mem_V_id[0]
            min_dist = np.linalg.norm(Centroids[id_min] - centroid_input_pat)

            for j in range(1, len(max_mem_V_id)):
                id_j = max_mem_V_id[j]
                dist_j = np.linalg.norm(Centroids[id_j] - centroid_input_pat)

                if dist_j < min_dist or (dist_j == min_dist and N_samples[id_j] > N_samples[id_min]):
                    id_min = id_j
                    min_dist = dist_j

            y_pred[i] = C[id_min]

    return y_pred


def predict_with_membership(V, W, C, Xl, Xu, g=1):
    """
    Return class membership values for samples in `X` represented in the form
    of invervals `[Xl, Xu]`. This is a common function to determine the
    membership values from an input X to a trained hyperbox-based classifier
    represented by `[V, W, C]`.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points for numerical features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points for numerical features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    C : array-like of shape (n_hyperboxes,)
        A vector stores all class labels correponding to existing hyperboxes.
    Xl : array-like of shape (n_samples, n_features)
        The data matrix contains lower bounds of input patterns for which we
        want to predict the targets.
    Xu : array-like of shape (n_samples, n_features)
        The data matrix contains upper bounds of input patterns for which we
        want to predict the targets.
    g : float or array-like of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    mem_vals : ndarray of shape (n_samples, n_classes)
        A vector contains the membership values for all classes for each input
        sample which needs to get the membership values.

    """
    if Xl.ndim == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)

    if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
        Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)

    mem_vals, _ = get_membership_gfmm_all_classes(Xl, Xu, V, W, C, g)
    
    return mem_vals


class MultiGranularGFMM(BaseHyperboxClassifier):
    """
    A multi-resolution hierarchical granular representation based classifier
    using general fuzzy min-max neural network.

    This class implements the multi-granular learning algorithm to construct
    classifiers from multiresolution hierarchical granular representations
    using hyperbox fuzzy sets. This algorithm forms a series of granular
    inferences hierarchically through many levels of abstraction. An attractive
    characteristic of our classifier is that it can maintain a high accuracy in
    comparison to other fuzzy min-max models at a low degree of granularity
    based on reusing the knowledge learned from lower levels of abstraction.
    In addition, our approach can reduce the data size significantly as well as
    handle the uncertainty and incompleteness associated with data in
    real-world applications. The construction process of the classifier
    consists of two phases. The first phase is to formulate the model at the
    greatest level of granularity, while the later stage aims to reduce the
    complexity of the constructed model and deduce it from data at higher
    abstraction levels. The details of this algorithm can be found in [1]_.

    Parameters
    ----------
    n_partitions : int, default=4
        Number of partitions to split the original training set into disjoint
        training sets to build base learners.
    granular_theta : list of float, optional, default=[0.1, 0.2, 0.3]
        Maximum hyperbox sizes at granularity levels.
    gamma : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous feature.
    min_membership_aggregation : float, optional, default=0.5
        Minimum membership value between two hyperboxes aggregated to form a
        larger sized hyperbox at a higher level of abstraction.
    random_state : int, RandomState instance or None, default=None
        Controls the stratified random sampling rate of the original dataset
        to form disjoint subsets for training base learners.

    Attributes
    ----------
    granularity_level : dict
        A mapping between the maximum hyperbox size and the granular level.
    smallest_theta : float
        Maximum hyperbox size at the highest granularity level.
    higher_level_theta : list of float
        Maximum hyperbox sizes of higher abstraction levels apart form the
        highest granularity level.
    granular_classifiers_ : ndarray of BaseGranular objects with shape (n_granularity_levels,)
        A list of general fuzzy min-max neural networks at all granularity levels.
    base_learners_ : list
        A list of base learners trained from disjoint subsets of input training
        patterns.
    is_exist_missing_value : boolean
        Is there any missing values in continuous features in the training data.
    elapsed_training_time : float
        Training time in seconds.

    References
    ----------
    .. [1] T.T. Khuat, F. Chen, and B. Gabrys, "An Effective Multiresolution
           Hierarchical Granular Representation Based Classifier Using General
           Fuzzy Min-Max Neural Network," IEEE Transactions on Fuzzy Systems,
           vol. 29, no. 2, pp. 427-441, 2021.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from hbbrain.numerical_data.multigranular_learner.multi_resolution_gfmm import MultiGranularGFMM
    >>> X, y = load_iris(return_X_y=True)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = MultiGranularGFMM(n_partitions=2, granular_theta=[0.1, 0.2, 0.3, 0.4, 0.5], gamma=1, min_membership_aggregation=0.6, random_state=0)
    >>> clf.fit(X, y)
    >>> clf.predict(X[[10, 50, 100]])
    array([0, 1, 2])
    >>> clf.predict(X[[10, 50, 100]], level=0)
    array([0, 1, 2])
    >>> print("Number of hyperboxes at granularity 1 = %d"%clf.get_n_hyperboxes(0))
    Number of hyperboxes at granularity 1 = 77
    >>> clf.predict(X[[10, 50, 100]], level=4)
    array([0, 1, 2])
    >>> print("Number of hyperboxes at granularity 5 = %d"%clf.get_n_hyperboxes(4))
    Number of hyperboxes at granularity 5 = 11

    """

    def __init__(self, n_partitions=4, granular_theta=[0.1, 0.2, 0.3], gamma=1, min_membership_aggregation=0.5, random_state=0):
        BaseHyperboxClassifier.__init__(self, granular_theta[0])
        self.granularity_level = convert_granular_theta_to_level(granular_theta)
        self.granular_theta = granular_theta
        self.smallest_theta = granular_theta[0]
        self.higher_level_theta = granular_theta[1:]
        self.min_membership_aggregation = min_membership_aggregation
        self.n_partitions = n_partitions
        self.gamma = gamma
        self.random_state = random_state
        self.granular_classifiers_ = np.empty(len(self.higher_level_theta)+1, dtype=object)

    def _build_homogeneous_class_base_learner(self, Xl, Xu, y):
        """
        Train a general fuzzy min-max neural network base learner. Training
        samples are grouped by class labels one by one. Therefore, training
        samples of a new class only appear when training samples of prior
        class are completely absorbed.

        Parameters
        ----------
        Xl : array-like of shape (n_samples_for_learner, n_features)
            The data matrix contains lower bounds of input training patterns
            used to train a base learner.
        Xu : array-like of shape (n_samples_for_learner, n_features)
            The data matrix contains upper bounds of input training patterns
            used to train a base learner.
        y : array-like of shape (n_samples_for_learner,)
            Target vector relative to input training hyperboxes `[Xl, Xu]`.

        Returns
        -------
        BaseGranular
            A trained base learner with lower and upper bounds, class labels,
            number of fully included samples within each hyperbox and centroids
            of hyperboxes.

        """
        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
            Xu = Xu.reshape(1, -1)
        V = np.array([])
        W = np.array([])
        C = np.array([])
        N_samples = np.array([])
        Centroids = np.array([])
        class_labels = np.unique(y[y != UNLABELED_CLASS])
        # for unlabelled samples, they need to be handled at the last iteration
        if (y == UNLABELED_CLASS).any():
            class_labels = np.concatenate((class_labels, [UNLABELED_CLASS]))

        threshold_mem_val = 1 - np.max(self.gamma) * self.smallest_theta
        for c in class_labels:
            Xl_ho = Xl[y == c]
            Xu_ho = Xu[y == c]
            y_ho = y[y == c]
            n_samples = len(y_ho)
            for i in range(n_samples):
                if V.size == 0:
                    V = np.array([Xl_ho[i]])
                    W = np.array([Xu_ho[i]])
                    C = np.array([y_ho[i]])
                    N_samples = np.array([1])
                    Centroids = np.array([(Xl_ho[i] + Xu_ho[i])/2])
                else:
                    if y_ho[i] == UNLABELED_CLASS:
                        id_same_input_label_group = np.arange(len(C))
                    else:
                        id_same_input_label_group = (C == y_ho[i]) | (C == UNLABELED_CLASS)

                    if id_same_input_label_group.any() == True:
                        V_sameX = V[id_same_input_label_group]
                        W_sameX = W[id_same_input_label_group]
                        # contain both class label as same as the input pattern and unlabelled
                        lb_sameX = C[id_same_input_label_group]
                        id_range = np.arange(len(C))
                        # determine the indices of samples with the same class label as the input sample
                        ids_hyperboxes_same_input_label = id_range[id_same_input_label_group]

                        if self.is_exist_missing_value:
                            b = membership_func_gfmm(Xl_ho[i], Xu_ho[i], np.minimum(
                                V_sameX, W_sameX), np.maximum(V_sameX, W_sameX), self.gamma)
                        else:
                            b = membership_func_gfmm(
                                Xl_ho[i], Xu_ho[i], V_sameX, W_sameX, self.gamma)

                        id_descending_mem_val = np.argsort(b)[::-1]
                        if b[id_descending_mem_val[0]] != 1 or (y_ho[i] != lb_sameX[id_descending_mem_val[0]] and y_ho[i] != UNLABELED_CLASS):
                            adjust = False
                            count = 0
                            for j in ids_hyperboxes_same_input_label[id_descending_mem_val]:
                                if b[id_descending_mem_val[count]] < threshold_mem_val:
                                    break

                                count = count + 1
                                # Check for violation of max hyperbox size and class labels
                                Vj_new = np.minimum(V[j], Xl_ho[i])
                                Wj_new = np.maximum(W[j], Xu_ho[i])
                                if (y_ho[i] == C[j] or C[j] == UNLABELED_CLASS or y_ho[i] == UNLABELED_CLASS) and (((Wj_new - Vj_new) <= self.smallest_theta).all() == True):
                                    # adjust the j-th hyperbox
                                    V[j] = Vj_new
                                    W[j] = Wj_new
                                    Centroids[j] = (Centroids[j] * N_samples[j] + (Xl_ho[j] + Xu_ho[j])/2) / (N_samples[j] + 1)
                                    N_samples[j] = N_samples[j] + 1
                                                                   
                                    adjust = True
                                    if (y_ho[i] != UNLABELED_CLASS) and (C[j] == UNLABELED_CLASS):
                                        C[j] = y_ho[i]
                                    # found out the winner hyperbox to adjust => break the loop
                                    break

                            # if the ith sample did not fit into any existing hyperboxes, create a new one
                            if not adjust:
                                V = np.concatenate((V, Xl_ho[i].reshape(1, -1)), axis=0)
                                W = np.concatenate((W, Xu_ho[i].reshape(1, -1)), axis=0)
                                C = np.concatenate((C, [y_ho[i]]))
                                N_samples = np.concatenate((N_samples, [1]))
                                tmp_new_centroid = (Xl_ho[i] + Xu_ho[i]) / 2
                                Centroids = np.concatenate((Centroids, tmp_new_centroid.reshape(1, -1)), axis=0)
                    else:
                        # There are no existing hyperboxes representing the same class label as the input patter
                        # We need to create a new hyperbox for the input sample
                        V = np.concatenate((V, Xl_ho[i].reshape(1, -1)), axis=0)
                        W = np.concatenate((W, Xu_ho[i].reshape(1, -1)), axis=0)
                        C = np.concatenate((C, [y_ho[i]]))
                        N_samples = np.concatenate((N_samples, [1]))
                        tmp_new_centroid = (Xl_ho[i] + Xu_ho[i]) / 2
                        Centroids = np.concatenate((Centroids, tmp_new_centroid.reshape(1, -1)), axis=0)

        return BaseGranular(V=V, W=W, C=C, N_samples=N_samples, Centroids=Centroids)

    def _build_heterogeneous_class_base_learner(self, Xl, Xu, y):
        """
        Train a general fuzzy min-max neural network base learner. The class
        labels of input patterns are in any order.

        Parameters
        ----------
        Xl : array-like of shape (n_samples_for_learner, n_features)
            The data matrix contains lower bounds of input training patterns
            used to train a base learner.
        Xu : array-like of shape (n_samples_for_learner, n_features)
            The data matrix contains upper bounds of input training patterns
            used to train a base learner.
        y : array-like of shape (n_samples_for_learner,)
            Target vector relative to input training hyperboxes `[Xl, Xu]`.

        Returns
        -------
        BaseGranular
            A trained base learner with lower and upper bounds, class labels,
            number of fully included samples within each hyperbox and centroids
            of hyperboxes.

        """
        if Xl.ndim == 1:
            Xl = Xl.reshape(1, -1)
            Xu = Xu.reshape(1, -1)
        V = np.array([])
        W = np.array([])
        C = np.array([])
        N_samples = np.array([])
        Centroids = np.array([])
        threshold_mem_val = 1 - np.max(self.gamma) * self.smallest_theta
        n_samples = len(y)

        for i in range(n_samples):
            if V.size == 0:
                V = np.array([Xl[i]])
                W = np.array([Xu[i]])
                C = np.array([y[i]])
                N_samples = np.array([1])
                Centroids = np.array([(Xl[i] + Xu[i])/2])
            else:
                if y[i] == UNLABELED_CLASS:
                    id_same_input_label_group = np.ones(len(C), dtype=bool)
                else:
                    id_same_input_label_group = (C == y[i]) | (C == UNLABELED_CLASS)

                if id_same_input_label_group.any() == True:
                    V_sameX = V[id_same_input_label_group]
                    W_sameX = W[id_same_input_label_group]
                    # contain both class label as same as the input pattern and unlabelled
                    lb_sameX = C[id_same_input_label_group]
                    id_range = np.arange(len(C))
                    # determine the indices of samples with the same class label as the input sample
                    ids_hyperboxes_same_input_label = id_range[id_same_input_label_group]

                    if self.is_exist_missing_value:
                        b = membership_func_gfmm(Xl[i], Xu[i], np.minimum(
                            V_sameX, W_sameX), np.maximum(V_sameX, W_sameX), self.gamma)
                    else:
                        b = membership_func_gfmm(
                            Xl[i], Xu[i], V_sameX, W_sameX, self.gamma)

                    id_descending_mem_val = np.argsort(b)[::-1]
                    if b[id_descending_mem_val[0]] != 1 or (y[i] != lb_sameX[id_descending_mem_val[0]] and y[i] != UNLABELED_CLASS):
                        adjust = False
                        count = 0
                        for j in ids_hyperboxes_same_input_label[id_descending_mem_val]:
                            if b[id_descending_mem_val[count]] < threshold_mem_val:
                                break

                            count = count + 1
                            # Check for violation of max hyperbox size and class labels
                            Vj_new = np.minimum(V[j], Xl[i])
                            Wj_new = np.maximum(W[j], Xu[i])
                            if (y[i] == C[j] or C[j] == UNLABELED_CLASS or y[i] == UNLABELED_CLASS) and (((Wj_new - Vj_new) <= self.smallest_theta).all() == True):
                                # adjust the j-th hyperbox
                                V[j] = Vj_new
                                W[j] = Wj_new
                                Centroids[j] = (Centroids[j] * N_samples[j] + (Xl[j] + Xu[j])/2) / (N_samples[j] + 1)
                                N_samples[j] = N_samples[j] + 1
                                                               
                                adjust = True
                                if (y[i] != UNLABELED_CLASS) and (C[j] == UNLABELED_CLASS):
                                    C[j] = y[i]
                                # found out the winner hyperbox to adjust => break the loop
                                break

                        # if the ith sample did not fit into any existing hyperboxes, create a new one
                        if not adjust:
                            V = np.concatenate((V, Xl[i].reshape(1, -1)), axis=0)
                            W = np.concatenate((W, Xu[i].reshape(1, -1)), axis=0)
                            C = np.concatenate((C, [y[i]]))
                            N_samples = np.concatenate((N_samples, [1]))
                            tmp_new_centroid = (Xl[i] + Xu[i]) / 2
                            Centroids = np.concatenate((Centroids, tmp_new_centroid.reshape(1, -1)), axis=0)
                else:
                    # There are no existing hyperboxes representing the same class label as the input patter
                    # We need to create a new hyperbox for the input sample
                    V = np.concatenate((V, Xl[i].reshape(1, -1)), axis=0)
                    W = np.concatenate((W, Xu[i].reshape(1, -1)), axis=0)
                    C = np.concatenate((C, [y[i]]))
                    N_samples = np.concatenate((N_samples, [1]))
                    tmp_new_centroid = (Xl[i] + Xu[i]) / 2
                    Centroids = np.concatenate((Centroids, tmp_new_centroid.reshape(1, -1)), axis=0)

        return BaseGranular(V=V, W=W, C=C, N_samples=N_samples, Centroids=Centroids)

    def simple_pruning(self, V, W, C, N_samples, Centroids, Xl_val, Xu_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Simply prune low qualitied hyperboxes based on a pre-defined accuracy threshold for each hyperbox

        Parameters
        ----------
        V : array-like of shape (n_hyperboxes, n_features)
            A matrix stores all minimal points for numerical features of all
            existing hyperboxes, in which each row is a minimal point of a hyperbox.
        W : array-like of shape (n_hyperboxes, n_features)
            A matrix stores all maximal points for numerical features of all
            existing hyperboxes, in which each row is a minimal point of a hyperbox.
        C : array-like of shape (n_hyperboxes,)
            A vector stores all class labels correponding to existing hyperboxes.
        N_samples : array-like of shape (n_hyperboxes,)
            A vector stores the number of samples fully included in each existing
            hyperbox.
        Centroids : array-like of shape (n_hyperboxes, n_features)
            A matrix stores all centroid points of all existing hyperboxes, in
            which each row is a centroid point of a hyperbox.
        Xl_val : array-like of shape (n_samples, n_features)
            The data matrix contains lower bounds of validation patterns.
        Xu_val : array-like of shape (n_samples, n_features)
            The data matrix contains upper bounds of validation patterns.
        y_val : ndarray of shape (n_samples,)
            A vector contains the true class label corresponding to each validation pattern.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction
            process on the validation set. If True, keep them, else the decision
            for keeping or removing based on the classification accuracy on the
            validation dataset

        Returns
        -------
        new_V : array-like of shape (n_new_hyperboxes, n_features)
            A matrix stores all minimal points for numerical features of all
            remaining hyperboxes after pruning, in which each row is a minimal
            point of a hyperbox.
        new_W : array-like of shape (n_new_hyperboxes, n_features)
            A matrix stores all maximal points for numerical features of all
            remaining hyperboxes after pruning, in which each row is a maximal
            point of a hyperbox.
        new_C : array-like of shape (n_new_hyperboxes,)
            A vector stores all class labels correponding to remaining
            hyperboxes after pruning.
        new_N_samples : array-like of shape (n_new_hyperboxes,)
            A vector stores the number of samples fully included in each
            remaining hyperbox after pruning.
        new_Centroids : array-like of shape (n_new_hyperboxes, n_features)
            A matrix stores all centroid points of all remaining hyperboxes
            after pruning, in which each row is a centroid point of a hyperbox.

        """
        n_samples = Xl_val.shape[0]
        rnd = np.random
        rnd.seed(0)
        # Matrices storing the classification accuracy for each created hyperbox in the trained model
        # The first column stores the number of corrected classification samples and the second column stores the number of wrong classification samples
        hyperboxes_performance = np.zeros((len(C), 2))

        if (is_contain_missing_value(Xl_val) == True) or (is_contain_missing_value(Xu_val) == True):
            Xl_val, Xu_val, y_val = convert_format_missing_input_zero_one(Xl_val, Xu_val, y_val)

        for i in range(n_samples):
            if self.is_exist_missing_value == False:
                mem_val = membership_func_gfmm(Xl_val[i], Xu_val[i], V, W, self.gamma) # calculate memberships for all hyperboxes
            else:
                mem_val = membership_func_gfmm(Xl_val[i], Xu_val[i], np.minimum(V, W), np.maximum(W, V), self.gamma)

            bmax = mem_val.max() # get max membership value
            # get indices of all hyperboxes with max membership
            max_mem_V_id = np.nonzero(mem_val == bmax)[0]

            if len(max_mem_V_id) == 1:
                # Only one hyperbox with the highest membership function
                if C[max_mem_V_id[0]] == y_val[i]:
                    hyperboxes_performance[max_mem_V_id[0], 0] = hyperboxes_performance[max_mem_V_id[0], 0] + 1                 
                else:
                    hyperboxes_performance[max_mem_V_id[0], 1] = hyperboxes_performance[max_mem_V_id[0], 1] + 1
            else:
                # More than one hyperbox with highest membership => using Manhattan distance
                # More than one hyperbox with highest membership => compare with centroid
                centroid_input_pat = (Xl_val[i] + Xu_val[i]) / 2
                id_min = max_mem_V_id[0]
                min_dist = np.linalg.norm(Centroids[id_min] - centroid_input_pat)

                for j in range(1, len(max_mem_V_id)):
                    id_j = max_mem_V_id[j]
                    dist_j = np.linalg.norm(Centroids[id_j] - centroid_input_pat)

                    if dist_j < min_dist or (dist_j == min_dist and N_samples[id_j] > N_samples[id_min]):
                        id_min = id_j
                        min_dist = dist_j

                if C[id_min] != y_val[i] and y_val[i] != UNLABELED_CLASS:
                    hyperboxes_performance[id_min, 1] = hyperboxes_performance[id_min, 1] + 1
                else:
                    hyperboxes_performance[id_min, 0] = hyperboxes_performance[id_min, 0] + 1

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
            new_V = V[id_remained_incl_empty_boxes]
            new_W = W[id_remained_incl_empty_boxes]
            new_C = C[id_remained_incl_empty_boxes]
            new_N_samples = N_samples[id_remained_incl_empty_boxes]
            new_Centroids = Centroids[id_remained_incl_empty_boxes]
        else:
            # keep one hyperbox for the class for which all of its hyperboxes
            # are prunned
            current_classes = np.unique(C)
            class_tmp = C[id_remained_excl_empty_boxes]
            for c in current_classes:
                if c not in class_tmp:
                    pos = np.nonzero(C == c)[0]
                    id_kept = rnd.randint(len(pos))
                    id_remained_excl_empty_boxes[pos[id_kept]] = True

            V_pruned_excl_empty_boxes = V[id_remained_excl_empty_boxes]
            W_pruned_excl_empty_boxes = W[id_remained_excl_empty_boxes]
            C_pruned_excl_empty_boxes = C[id_remained_excl_empty_boxes]
            N_samples_pruned_excl_empty_boxes = N_samples[id_remained_excl_empty_boxes]
            Centroids_pruned_excl_empty_boxes = Centroids[id_remained_excl_empty_boxes]

            W_pruned_incl_empty_boxes = W[id_remained_incl_empty_boxes]
            V_pruned_incl_empty_boxes = V[id_remained_incl_empty_boxes]
            C_pruned_incl_empty_boxes = C[id_remained_incl_empty_boxes]
            N_samples_pruned_incl_empty_boxes = N_samples[id_remained_incl_empty_boxes]
            Centroids_pruned_incl_empty_boxes = Centroids[id_remained_incl_empty_boxes]

            y_val_pred_excl_empty_boxes = predict_with_centroids(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, N_samples_pruned_excl_empty_boxes, Centroids_pruned_excl_empty_boxes, Xl_val, Xu_val, self.gamma)
            y_val_pred_incl_empty_boxes = predict_with_centroids(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, N_samples_pruned_incl_empty_boxes, Centroids_pruned_incl_empty_boxes, Xl_val, Xu_val, self.gamma)

            if (accuracy_score(y_val, y_val_pred_excl_empty_boxes) >= accuracy_score(y_val, y_val_pred_incl_empty_boxes)):
                new_V = V_pruned_excl_empty_boxes
                new_W = W_pruned_excl_empty_boxes
                new_C = C_pruned_excl_empty_boxes
                new_N_samples = N_samples_pruned_excl_empty_boxes
                new_Centroids = Centroids_pruned_excl_empty_boxes
            else:
                new_V = V_pruned_incl_empty_boxes
                new_W = W_pruned_incl_empty_boxes
                new_C = C_pruned_incl_empty_boxes
                new_N_samples = N_samples_pruned_incl_empty_boxes
                new_Centroids = Centroids_pruned_incl_empty_boxes

        return (new_V, new_W, new_C, new_N_samples, new_Centroids)

    def granular_learning_phase_1(self, Xl, Xu, y, learning_type=HETEROGENEOUS_CLASS_LEARNING, X_val=None, y_val=None, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Training a granular general fuzzy min-max neural network using a
        learning algorithm in phase 1 to distribute disjoint subsets into
        working processes to build base learners. After that, resulting
        hyperboxes from all base learners will merged and pruned.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The data matrix contains lower bounds of input training patterns.
        Xu : array-like of shape (n_samples, n_features)
            The data matrix contains upper bounds of input training patterns.
        y : array-like of shape (n_samples,)
            Target vector relative to input training hyperboxes `[Xl, Xu]`.
        learning_type : enum (int), optional, default=HETEROGENEOUS_CLASS_LEARNING
            Learning type is used to build base learners from disjoint datasets.
            It gets two defined enum values being HETEROGENEOUS_CLASS_LEARNING
            and HOMOGENEOUS_CLASS_LEARNING. Heterogeneous class learning means
            that base learners are trained based on the order of input samples.
            Homogeneous class learning means that input data are sorted and
            grouped according to class labels before starting the training
            process.
        X_val : array-like of shape (n_samples, n_features)
            The data matrix contains validation patterns.
        y_val : ndarray of shape (n_samples,)
            A vector contains the true class label corresponding to each
            validation pattern.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction
            process on the validation set. If True, keep them, else the decision
            for keeping or removing based on the classification accuracy on the
            validation dataset

        Returns
        -------
        self : object
            A granular general fuzzy min-max neural network trained by a
            phase-1 learning algorithm.

        """
        # Split training data into disjoint partitions
        skf = StratifiedKFold(n_splits=self.n_partitions, shuffle=True, random_state=self.random_state)
        partition_sample_ids = []
        for _, part_sample_id in skf.split(Xl, y):
            partition_sample_ids.append(part_sample_id)

        self.base_learners_ = list()
        # Map procedure
        if learning_type == HETEROGENEOUS_CLASS_LEARNING:
            all_results = Parallel(
                n_jobs=self.n_partitions, verbose=1
            )(
                delayed(self._build_heterogeneous_class_base_learner)(
                    Xl[partition_sample_ids[i]],
                    Xu[partition_sample_ids[i]],
                    y[partition_sample_ids[i]]
                )
                for i in range(self.n_partitions)
            )
        else:
            all_results = Parallel(
                n_jobs=self.n_partitions
            )(
                delayed(self._build_homogeneous_class_base_learner)(
                    Xl[partition_sample_ids[i]],
                    Xu[partition_sample_ids[i]],
                    y[partition_sample_ids[i]]
                )
                for i in range(self.n_partitions)
            )

        # Reduce
        for t in all_results:
            self.base_learners_.append(t)

        # Merging procedure for all resulting hyperboxes from base learners
        V = self.base_learners_[0].V.copy()
        W = self.base_learners_[0].W.copy()
        C = self.base_learners_[0].C.copy()
        N_samples = self.base_learners_[0].N_samples.copy()
        Centroids = self.base_learners_[0].Centroids.copy()

        for i in range(1, self.n_partitions):
            V = np.concatenate((V, self.base_learners_[i].V), axis=0)
            W = np.concatenate((W, self.base_learners_[i].W), axis=0)
            C = np.concatenate((C, self.base_learners_[i].C))
            N_samples = np.concatenate((N_samples, self.base_learners_[i].N_samples))
            Centroids = np.concatenate((Centroids, self.base_learners_[i].Centroids), axis=0)

        # Remove the hyperboxes included in other hyperboxes with the same
        # class label
        V, W, C, N_samples, Centroids, self.n_removed_hyperboxes = remove_contained_hyperboxes(V, W, C, N_samples, Centroids)
        # Pruning
        if X_val is not None:
            self.classifier_before_pruning_ = BaseGranular(V=V, W=W, C=C, N_samples=N_samples, Centroids=Centroids)
            V, W, C, N_samples, Centroids = self.simple_pruning(V, W, C, N_samples, Centroids, X_val, X_val, y_val, acc_threshold, keep_empty_boxes)

        # Store generated hyperboxes after phase 1
        self.granular_classifiers_[0] = BaseGranular(V=V, W=W, C=C, N_samples=N_samples, Centroids=Centroids)

        return self

    def granular_learning_phase_2(self):
        """
        Training a granular general fuzzy min-max neural network using a
        learning algorithm in phase 2 to reduce number of hyperboxes while
        keeping a good classification performance.

        Returns
        -------
        self : object
            A granular general fuzzy min-max neural network trained by a
            phase-2 learning algorithm.

        """
        for level, theta in enumerate(self.higher_level_theta):
            threshold = max(self.min_membership_aggregation, 1 - np.max(self.gamma) * theta)
            V = np.array([])
            n_pre_hyperboxes = len(self.granular_classifiers_[level].C)
            for i in range(n_pre_hyperboxes):
                if V.size == 0:
                    V = np.array([self.granular_classifiers_[level].V[i]])
                    W = np.array([self.granular_classifiers_[level].W[i]])
                    C = np.array([self.granular_classifiers_[level].C[i]])
                    N_samples = np.array([self.granular_classifiers_[level].N_samples[i]])
                    Centroids = np.array([self.granular_classifiers_[level].Centroids[i]])
                else:
                    class_input_sample = self.granular_classifiers_[level].C[i]
                    if class_input_sample == UNLABELED_CLASS:
                        id_same_input_label_group = np.ones(len(C), dtype=bool)
                    else:
                        id_same_input_label_group = (C == class_input_sample) | (C == UNLABELED_CLASS)

                    is_create_new_hyperbox = False
                    if id_same_input_label_group.any() == True: 
                        V_sameX = V[id_same_input_label_group]
                        W_sameX = W[id_same_input_label_group]
                        lb_sameX = C[id_same_input_label_group]
                        N_samples_sameX = N_samples[id_same_input_label_group]
                        Centroids_sameX = Centroids[id_same_input_label_group]
                        id_range = np.arange(len(C))
                        ids_hyperboxes_same_input_label = id_range[id_same_input_label_group]
                        
                        if self.is_exist_missing_value:
                            mem_vals = membership_func_gfmm(self.granular_classifiers_[level].V[i], self.granular_classifiers_[level].W[i], np.minimum(
                                V_sameX, W_sameX), np.maximum(V_sameX, W_sameX), self.gamma)
                        else:
                            mem_vals = membership_func_gfmm(
                                self.granular_classifiers_[level].V[i], self.granular_classifiers_[level].W[i], V_sameX, W_sameX, self.gamma)

                        id_descending_mem_val = np.argsort(mem_vals)[::-1]
                        sorted_mem_vals = mem_vals[id_descending_mem_val]

                        if sorted_mem_vals[0] != 1 or (class_input_sample != lb_sameX[id_descending_mem_val[0]] and class_input_sample != UNLABELED_CLASS):
                            adjust = False
                            considered_mem_vals = sorted_mem_vals[sorted_mem_vals >= threshold]

                            if len(considered_mem_vals) > 0:
                                id_considered_hyperboxes = id_descending_mem_val[sorted_mem_vals >= threshold]
                                for j in ids_hyperboxes_same_input_label[id_considered_hyperboxes]:
                                    # test violation of max hyperbox size and class labels
                                    if (class_input_sample == C[j] or C[j] == UNLABELED_CLASS or class_input_sample == UNLABELED_CLASS) and ((np.maximum(W[j], self.granular_classifiers_[level].W[i]) - np.minimum(V[j], self.granular_classifiers_[level].V[i])) <= theta).all() == True:
                                        # save old value
                                        Vj_old = V[j].copy()
                                        Wj_old = W[j].copy()
                                        C_old = C[j]
                                        
                                        # adjust the j-th hyperbox
                                        V[j] = np.minimum(V[j], self.granular_classifiers_[level].V[i])
                                        W[j] = np.maximum(W[j], self.granular_classifiers_[level].W[i])
                                        
                                        if class_input_sample != UNLABELED_CLASS and C[j] == UNLABELED_CLASS:
                                            C[j] = class_input_sample
                                        
                                        # Test overlap        
                                        if is_overlap_one_many_hyperboxes_num_data_general(V, W, C, j) == True:
                                            # Exist overlapping regions with hyperboxes belonging to other classes
                                            # revert change and Choose other hyperbox
                                            V[j] = Vj_old
                                            W[j] = Wj_old
                                            C[j] = C_old
                                        else:
                                            # Keep changes and update centroid, stopping the process of choosing hyperboxes
                                            Centroids[j] = (Centroids[j] * N_samples[j] + self.granular_classifiers_[level].Centroids[i] * self.granular_classifiers_[level].N_samples[i]) / (N_samples[j] + self.granular_classifiers_[level].N_samples[i])
                                            N_samples[j] = N_samples[j] + self.granular_classifiers_[level].N_samples[i]

                                            adjust = True
                                            break

                            # if the i-th sample did not fit into any existing
                            # box, create a new one
                            if not adjust:
                                is_create_new_hyperbox = True
                    else:
                        is_create_new_hyperbox = True

                    if is_create_new_hyperbox == True:
                        V = np.concatenate((V, self.granular_classifiers_[level].V[i].reshape(1, -1)), axis = 0)
                        W = np.concatenate((W, self.granular_classifiers_[level].W[i].reshape(1, -1)), axis = 0)
                        C = np.concatenate((C, [class_input_sample]))
                        N_samples = np.concatenate((N_samples, [self.granular_classifiers_[level].N_samples[i]]))
                        Centroids = np.concatenate((Centroids, self.granular_classifiers_[level].Centroids[i].reshape(1, -1)), axis = 0)
                        # Test overlap and do contraction with current hyperbox because phase 1 create overlapping regions
                        n_existed_hyperboxes = V.shape[0]
                        id_of_winner_hyperbox = len(C) - 1
                        for ii in range(n_existed_hyperboxes):
                            if (ii != id_of_winner_hyperbox) and (C[ii] != C[id_of_winner_hyperbox] or C[id_of_winner_hyperbox] == UNLABELED_CLASS):
                                # overlap test
                                is_overlap = is_two_hyperboxes_overlap_num_data_general(
                                    V[id_of_winner_hyperbox], W[id_of_winner_hyperbox], V[ii], W[ii])

                                if is_overlap == True:
                                    V[id_of_winner_hyperbox], W[id_of_winner_hyperbox], V[ii], W[ii] = overlap_resolving_num_data(
                                        V[id_of_winner_hyperbox], W[id_of_winner_hyperbox], C[id_of_winner_hyperbox], V[ii], W[ii], C[ii])
                        
            self.granular_classifiers_[level + 1] = BaseGranular(V=V, W=W, C=C, N_samples=N_samples, Centroids=Centroids)

        return self

    def _fit(self, Xl, Xu, y, learning_type=HETEROGENEOUS_CLASS_LEARNING, X_val=None, y_val=None, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Fit the model according to the given training data using the multi
        granularity learning algorithm. The input training patterns are given
        in the form of hyperboxes.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            Lower bounds of the training data.
        Xu : array-like of shape (n_samples, n_features)
            Upper bounds of the training data.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        learning_type : enum (int), optional, default=HETEROGENEOUS_CLASS_LEARNING
            Learning type is used to build base learners from disjoint datasets.
            It gets two defined enum values being HETEROGENEOUS_CLASS_LEARNING
            and HOMOGENEOUS_CLASS_LEARNING. Heterogeneous class learning means
            that base learners are trained based on the order of input samples.
            Homogeneous class learning means that input data are sorted and grouped
            according to class labels before starting the training process.
        X_val : array-like of shape (n_val_samples, n_features), optional, default=None
            A matrix contains a validation set, where `n_val_samples` is the
            number of validation samples and `n_features` is the number of features.
        y_val : array-like of shape (n_val_samples,), optional, default=None
            Target vector relative to X_val.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction
            process on the validation set. If True, keep them, else the decision
            for keeping or removing based on the classification accuracy on the
            validation dataset.

        Returns
        -------
        self : object
            Fitted multigranular general fuzzy min-max neural network.

        """
        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            self.is_exist_missing_value = True
            Xl, Xu, y = convert_format_missing_input_zero_one(Xl, Xu, y)
        else:
            self.is_exist_missing_value = False
            
        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        time_start = time.perf_counter()
        self.granular_learning_phase_1(Xl, Xu, y, learning_type, X_val, y_val, acc_threshold, keep_empty_boxes)
        self.granular_learning_phase_2()
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start

        return self

    def fit(self, X, y, learning_type=HETEROGENEOUS_CLASS_LEARNING, X_val=None, y_val=None, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Fit the model according to the given training data using the multi
        granularity learning algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        learning_type : enum (int), optional, default=HETEROGENEOUS_CLASS_LEARNING
            Learning type is used to build base learners from disjoint datasets.
            It gets two defined enum values being HETEROGENEOUS_CLASS_LEARNING
            and HOMOGENEOUS_CLASS_LEARNING. Heterogeneous class learning means
            that base learners are trained based on the order of input samples.
            Homogeneous class learning means that input data are sorted and grouped
            according to class labels before starting the training process.
        X_val : array-like of shape (n_val_samples, n_features), optional, default=None
            A matrix contains a validation set, where `n_val_samples` is the
            number of validation samples and `n_features` is the number of features.
        y_val : array-like of shape (n_val_samples,), optional, default=None
            Target vector relative to X_val.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction
            process on the validation set. If True, keep them, else the decision
            for keeping or removing based on the classification accuracy on the
            validation dataset.

        Returns
        -------
        self : object
            Fitted multigranular general fuzzy min-max neural network.

        """
        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        y = y.astype('int')
        n_samples = len(y)
        if X.shape[0] > n_samples:
            # Matrix X contains both lower and upper bounds which are stacked into a single matrix
            # We need to split it into two matrices for lower and upper bounds
            Xl = X[:n_samples, :]
            Xu = X[n_samples:, :]
            return self._fit(Xl, Xu, y, learning_type, X_val, y_val, acc_threshold, keep_empty_boxes)
        else:
            return self._fit(X, X, y, learning_type, X_val, y_val, acc_threshold, keep_empty_boxes)

    def predict(self, X, level=-1):
        """
        Predict class labels for samples in `X` at a given granularity level.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i`, an additional criterion based on the
            minimum distance between the input samples and the centroids of the
            winner hyperboxes is used to find the final winner hyperbox that
            its class label is used for predicting the class label of the input
            pattern :math:`X_i`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.
            
        level : int, optional, default=-1
            The granularity level is used to generate predicted classes for
            the input testing samples. If this variable gets the values of -1,
            then the predicted class for each sample is the class getting the
            most votes from all available granularity levels.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        X = np.array(X)
        y_pred = self._predict(X, X, level)
        
        return y_pred

    def _predict(self, Xl, Xu, level=-1):
        """
        Predict class labels for samples in the form of hyperboxes represented 
        by low bounds `Xl` and upper bounds `Xu` at a given granularity level.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i` in the form of an hyperbox represented by
            a lower bound :math:`Xl_i` and an upper bound :math:`Xu_i`, an
            additional criterion based on the minimum distance between the
            centroids of the winner hyperboxes and the input sample is used to
            find the final winner hyperbox that its class label is used for
            predicting the class label of the input hyperbox :math:`X_i`.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The data matrix containing the lower bounds of input patterns
            for which we want to predict the targets.
        Xu : array-like of shape (n_samples, n_features)
            The data matrix containing the upper bounds of input patterns 
            for which we want to predict the targets.
        level : int, optional, default=-1
            The granularity level is used to generate predicted classes for
            the input testing samples. If this variable gets the values of -1,
            then the predicted class for each sample is the class getting the
            most votes from all available granularity levels.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        if level == -1:
            n_levels = len(self.higher_level_theta) + 1
            if Xl.ndim == 1:
                n_samples = 1
            else:
                n_samples = Xl.shape[0]

            voting_y_pred = np.full((n_levels, n_samples), 0)
            for i in range(n_levels):
                voting_y_pred[i] = predict_with_centroids(self.granular_classifiers_[i].V,
                                                          self.granular_classifiers_[i].W,
                                                          self.granular_classifiers_[i].C,
                                                          self.granular_classifiers_[i].N_samples,
                                                          self.granular_classifiers_[i].Centroids,
                                                          Xl, Xu, self.gamma)
            y_pred = np.full(n_samples, 0)
            for i in range(n_samples):
                y_pred[i] = np.bincount(voting_y_pred[:, i]).argmax()
        else:
            y_pred = predict_with_centroids(self.granular_classifiers_[level].V,
                                            self.granular_classifiers_[level].W,
                                            self.granular_classifiers_[level].C,
                                            self.granular_classifiers_[level].N_samples,
                                            self.granular_classifiers_[level].Centroids,
                                            Xl, Xu, self.gamma)
        
        return y_pred

    def predict_at_partitions(self, Xl, Xu, partition=0):
        """
        Predict class labels for samples in the form of hyperboxes represented 
        by low bounds `Xl` and upper bounds `Xu` at a given granularity level.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i` in the form of an hyperbox represented by
            a lower bound :math:`Xl_i` and an upper bound :math:`Xu_i`, an
            additional criterion based on the minimum distance between the
            centroids of winner hyperboxes and the input sample is used to find
            the final winner hyperbox that its class label is used for predicting
            the class label of the input hyperbox :math:`X_i`.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The data matrix containing the lower bounds of input patterns
            for which we want to predict the targets.
        Xu : array-like of shape (n_samples, n_features)
            The data matrix containing the upper bounds of input patterns 
            for which we want to predict the targets.
        partition : int, optional, default=0
            The base learner at a given partition is used to generate predicted
            classes for the input testing samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        y_pred = predict_with_centroids(self.base_learners_[partition].V,
                                        self.base_learners_[partition].W,
                                        self.base_learners_[partition].C,
                                        self.base_learners_[partition].N_samples,
                                        self.base_learners_[partition].Centroids,
                                        Xl, Xu, self.gamma)

        return y_pred
    
    def predict_proba(self, X, level=-1):
        """
        Predict class probabilities of the input samples X at a given
        granularity level.

        The predicted class probability at a given granularity level is the
        fraction of the membership value of the representative hyperbox of
        that class at the given granularity level and the sum of all membership
        values of all representative hyperboxes of all classes joining the
        prediction procedure.


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        level : int, optional, default=-1
            The granularity level is used to generate predicted class
            probabilities for the input testing samples. If this variable gets
            the values of -1, then the predicted class probability value for
            each sample is the average of probability values at all
            available granularity levels.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        X = np.array(X)
        proba = self._predict_proba(X, X, level)

        return proba

    def _predict_proba(self, Xl, Xu, level=-1):
        """
        Predict class probabilities of the input hyperboxes represented
        by low bounds `Xl` and upper bounds `Xu` at a given granularity level.

        The predicted class probability is the fraction of the membership value
        of the representative hyperbox of that class at a given granularity
        level and the sum of all membership values of all representative
        hyperboxes of all classes joining the prediction procedure.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The data matrix containing the lower bounds of input patterns
            for which we want to predict the class probability.
        Xu : array-like of shape (n_samples, n_features)
            The data matrix containing the upper bounds of input patterns 
            for which we want to predict the class probability.
        level : int, optional, default=-1
            The granularity level is used to generate predicted class
            probabilities for the input testing samples. If this variable gets
            the values of -1, then the predicted class probability value for
            each sample is the average of probability values at all
            available granularity levels.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        mem_vals = self._predict_with_membership(Xl, Xu, level)
        normalizer = mem_vals.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = mem_vals / normalizer

        return proba

    def predict_with_membership(self, X, level=-1):
        """
        Predict class memberships of the input samples X at a given
        granularity level.

        The predicted class memberships are the membership values of the
        representative hyperbox of that class at a given granularity level.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        level : int, optional, default=-1
            The granularity level is used to generate predicted classes for
            the input testing samples. If this variable gets the values of -1,
            then the predicted class memberhip value for each sample is the
            average of all class memberships of all granularity levels.

        Returns
        -------
        mem_vals : ndarray of shape (n_samples, n_classes)
            The class memberships of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        X = np.array(X)
        mem_vals = self._predict_with_membership(X, X, level)

        return mem_vals
    
    def _predict_with_membership(self, Xl, Xu, level=-1):
        """
        Predict class memberships of the input hyperboxes represented 
        by low bounds `Xl` and upper bounds `Xu` at a given granularity level.

        The predicted class memberships are the membership values
        of the representative hyperbox of that class at a given granularity
        level.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The data matrix containing the lower bounds of input patterns
            for which we want to predict the class probability.
        Xu : array-like of shape (n_samples, n_features)
            The data matrix containing the upper bounds of input patterns 
            for which we want to predict the class probability.
        level : int, optional, default=-1
            The granularity level is used to generate predicted classes for
            the input testing samples. If this variable gets the values of -1,
            then the predicted class membership for each sample is the average
            value of class memberships over all available granularity levels.

        Returns
        -------
        mem_vals : ndarray of shape (n_samples, n_classes)
            The class memberships of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        if level == -1:
            n_levels = len(self.higher_level_theta) + 1
            if Xl.ndim == 1:
                n_samples = 1
            else:
                n_samples = Xl.shape[0]

            n_classes = len(np.unique(self.granular_classifiers_[0].C))

            prob_accumulation = np.full((n_samples, n_classes), 0)
            for i in range(n_levels):
                prob_accumulation = prob_accumulation + predict_with_membership(self.granular_classifiers_[i].V,
                                                                                self.granular_classifiers_[i].W,
                                                                                self.granular_classifiers_[i].C,
                                                                                Xl, Xu, self.gamma)
            mem_vals = prob_accumulation / n_levels
        else:
            mem_vals = predict_with_membership(self.granular_classifiers_[level].V,
                                               self.granular_classifiers_[level].W,
                                               self.granular_classifiers_[level].C,
                                               Xl, Xu, self.gamma)

        return mem_vals

    def get_n_hyperboxes(self, level=-1):
        """
        Get number of hyperboxes at a given granularity level.

        Parameters
        ----------
        level : int, optional, default=-1
            The granularity level needs to get number of hyperboxes. If `level`
            gets a value of -1, return number of hyperboxes in all granularity
            levels.

        Returns
        -------
        int
            Number of hyperboxes at the given granularity level.

        """
        if level == -1:
            n_boxes = 0
            for i in range(len(self.higher_level_theta) + 1):
                n_boxes += len(self.granular_classifiers_[i].C)
            return n_boxes
        else:
            return len(self.granular_classifiers_[level].C)

    def get_n_hyperboxes_at_partition(self, partition=0):
        """
        Get number of hyperboxes at a given granularity level.

        Parameters
        ----------
        partition : int, optional, default=0
            The partition needs to get number of base learners.

        Returns
        -------
        int
            Number of hyperboxes at the given partition.

        """
        return len(self.base_learners_[partition].C)

    def _initialise_canvas_graph(self, n_dims=2, figure_name='A trained hyperbox-based learning model', fig_num=1):
        """
        Initialise a canvas to draw hyperboxes

        Parameters
        ----------
        n_dims : int, optional, default=2
            The number of dimensions of hyperboxes shown in the canvas (2D or 3D).
        figure_name : str, optional, default='A trained hyperbox-based learning model'
            Title name of the window containing hyperboxes.
        fig_num : int, optional, default=1
            Index of canvas.

        Returns
        -------
        drawing_canvas : `axes.SubplotBase`, or another subclass of `Axes` in the matplotlib library
            Plotting object of matplotlib.

        """
        fig = plt.figure(fig_num)
        plt.ion()
        if n_dims == 2:
            drawing_canvas = fig.add_subplot(1, 1, 1)
            drawing_canvas.axis([0, 1, 0, 1])
            drawing_canvas.set_title(figure_name)
        else:
            drawing_canvas = Axes3D(fig)
            drawing_canvas.set_xlim3d(0, 1)
            drawing_canvas.set_ylim3d(0, 1)
            drawing_canvas.set_zlim3d(0, 1)

        return drawing_canvas

    def draw_2D_hyperbox_and_boundary_granular_level(self, window_name="Hyperbox-based classifier and its decision boundaries", level=0):
        """
        Draw the existing hyperboxes and their decision boundaries among classes
        at a given granularity level.

        .. note::

            This method only works on 2-dimensional datasets.

        Parameters
        ----------
        window_name : str, optional, default="Hyperbox-based classifier and its decision boundaries"
            Name of plotting window showing hyperboxes and their decision boundaries.

        level : int, optional, default=0
            The granularity level needs to draw hyperboxes and its boundaries.

        Returns
        -------
        None.

        """
        class_ids = np.unique(self.granular_classifiers_[level].C)
        n_hyperboxes, n_dims = self.granular_classifiers_[level].V.shape
        n_classes = len(class_ids)
        color_map = get_cmap(n_classes)
        # build a dictionary with the class label being key and color being value
        colors = {}
        for i in range(n_classes):
            colors[class_ids[i]] = color_map(i)

        drawing_canvas = self._initialise_canvas_graph(n_dims, window_name, level+1)
        # create a list of colors for the created hyperboxes
        box_colors = np.full(n_hyperboxes, None)
        for i in range(n_hyperboxes):
            box_colors[i] = colors[self.granular_classifiers_[level].C[i]]
        # draw hyperboxes
        draw_box(drawing_canvas, self.granular_classifiers_[level].V, self.granular_classifiers_[level].W, box_colors)
        # Generate a grid of points in a 2D plane to determine corresponding
        # classes for various areas
        grid, xx, yy = generate_grid_decision_boundary_2D(0, 1, 0, 1, 0.005)
        # make predictions for the points in the grid
        yhat = self.predict(grid, level)
        # Draw decision boundary
        draw_decision_boundary_2D(drawing_canvas, xx, yy, yhat)

    def draw_2D_hyperbox_and_boundary_partitions(self, window_name="Base learners and its decision boundaries", partition=0, fig_num=100):
        """
        Draw the existing hyperboxes and their decision boundaries among classes
        in a given partition.

        .. note::

            This method only works on 2-dimensional datasets.

        Parameters
        ----------
        window_name : str, optional, default="Hyperbox-based classifier and its decision boundaries"
            Name of plotting window showing hyperboxes and their decision boundaries.
        
        partition : int, optional, default=0
            The partition needs to draw hyperboxes and its boundary.
            
        fig_num : int, optional, default=100
            Index of the drawing canvas.
            
        Returns
        -------
        None.

        """
        class_ids = np.unique(self.base_learners_[partition].C)
        n_hyperboxes, n_dims = self.base_learners_[partition].V.shape
        n_classes = len(class_ids)
        color_map = get_cmap(n_classes)
        # build a dictionary with the class label being key and color being value
        colors = {}
        for i in range(n_classes):
            colors[class_ids[i]] = color_map(i)

        drawing_canvas = self._initialise_canvas_graph(n_dims, window_name, fig_num)
        # create a list of colors for the created hyperboxes
        box_colors = np.full(n_hyperboxes, None)
        for i in range(n_hyperboxes):
            box_colors[i] = colors[self.base_learners_[partition].C[i]]
        # draw hyperboxes
        draw_box(drawing_canvas, self.base_learners_[partition].V, self.base_learners_[partition].W, box_colors)
        # Generate a grid of points in a 2D plane to determine corresponding
        # classes for various areas
        grid, xx, yy = generate_grid_decision_boundary_2D(0, 1, 0, 1, 0.005)
        # make predictions for the points in the grid
        yhat = self.predict_at_partitions(grid, grid, partition)
        # Draw decision boundary
        draw_decision_boundary_2D(drawing_canvas, xx, yy, yhat)

    def get_sample_explanation_granular_level(self, xl, xu, level=0):
        """
        Get useful information for explaining the reason behind the predicted result for the input pattern

        Parameters
        ----------
        xl : ndarray of shape (n_feature,)
            Minimum point of the input pattern which needs to be explained.
        xu : ndarray of shape (n_feature,)
            Maximum point of the input pattern which needs to be explained.
        level : int, optional, default=0
            The granularity level is used to generate prediction.
        
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
            is the minimal points of all hyperboxes coressponding to each class
        dict_max_point_classes : dictionary
            A dictionary stores all maximal points of hyperboxes having the maximum 
            membership value for each class. The key is the class label and the value 
            is the maximal points of all hyperboxes coressponding to each class

        """
        mem_vals_for_classes, hyperbox_id_for_classes = get_membership_gfmm_all_classes(xl, xu, self.granular_classifiers_[level].V, self.granular_classifiers_[level].W, self.granular_classifiers_[level].C, self.gamma)
        class_values = np.unique(self.granular_classifiers_[level].C)
        # get predicted class label for the input sample
        y_pred = self._predict(xl, xu, level)[0]
        # create dictionaries with keys being class labels and values being membership values, maximum and minimum points
        dict_mem_val_classes = {}
        dict_min_point_classes = {}
        dict_max_point_classes = {}
        for _id, c in enumerate(class_values):
            dict_mem_val_classes[c] = mem_vals_for_classes[0][_id]
            box_id = hyperbox_id_for_classes[0][_id]
            dict_min_point_classes[c] = self.granular_classifiers_[level].V[box_id]
            dict_max_point_classes[c] = self.granular_classifiers_[level].W[box_id]

        return(y_pred, dict_mem_val_classes, dict_min_point_classes, dict_max_point_classes)


if __name__ == '__main__':
    import argparse
    import os

    def dir_path(path):
        if path == "":
            return None
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

    # Optional arguments
    optional.add_argument('--val_file', type=dir_path, default="",
                          help='The path to validation data file (including file name)')
    optional.add_argument('--n_partitions', type=int, default=4,
                          help='Number of disjoint partitions to train base learners (default: 4)')
    optional.add_argument('--granular_theta', type=str, default="[0.1, 0.2, 0.3, 0.4, 0.5]",
                          help='Granular maximum hyperbox sizes (default: [0.1, 0.2, 0.3, 0.4, 0.5])')
    optional.add_argument('--gamma', type=float, default=1,
                          help='A sensitivity parameter describing the speed of decreasing of the membership function in each dimension (larger than 0) (default: 1)')
    optional.add_argument('--min_membership_aggregation', type=float, default=0,
                          help='Minimum membership value for hyperbox aggregration at higher granular levels (in the range of [0, 1]) (default: 0)')

    args = parser.parse_args()

    if args.n_partitions <= 0:
        parser.error("--n_partitions has to be larger than 0")

    if args.min_membership_aggregation < 0 or args.min_membership_aggregation > 1:
        parser.error("--min_membership_aggregation has to be in the range of [0, 1]")

    if args.gamma <= 0:
        parser.error("--gamma has to be larger than 0")

    granular_theta = json.loads(args.granular_theta)
    n_partitions = args.n_partitions
    gamma = args.gamma
    min_membership_aggregation = args.min_membership_aggregation

    training_file = args.training_file
    testing_file = args.testing_file
    validation_file = args.val_file

    import pandas as pd
    df_train = pd.read_csv(training_file, header=None)
    df_test = pd.read_csv(testing_file, header=None)

    Xy_train = df_train.to_numpy()
    Xy_test = df_test.to_numpy()

    Xtr = Xy_train[:, :-1]
    ytr = Xy_train[:, -1]

    Xtest = Xy_test[:, :-1]
    ytest = Xy_test[:, -1]

    if validation_file is not None:
        df_val = pd.read_csv(validation_file, header=None)
        Xy_val = df_val.to_numpy()
        X_val = Xy_val[:, :-1]
        y_val = Xy_val[:, -1]
    else:
        X_val = None
        y_val = None

    gra_clf = MultiGranularGFMM(n_partitions=n_partitions, granular_theta=granular_theta, gamma=gamma, min_membership_aggregation=min_membership_aggregation, random_state=0)
    gra_clf.fit(Xtr, ytr, learning_type=HETEROGENEOUS_CLASS_LEARNING, X_val=X_val, y_val=y_val, acc_threshold=0.5, keep_empty_boxes=False)
    print("Training time: %.3f (s)"%(gra_clf.elapsed_training_time))

    y_pred = gra_clf.predict(Xtest, level=-1)
    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy (using voting from all granularity levels) = {acc * 100: .2f}%')
    
    print("Prediction of each base learner at a given partition:")
    for i in range(n_partitions):
        y_pred_partition = gra_clf.predict_at_partitions(Xtest, Xtest, i)
        acc_partition = accuracy_score(ytest, y_pred_partition)
        n_boxes_partition = gra_clf.get_n_hyperboxes_at_partition(i)
        print(f'Partition {i} - Testing accuracy = {acc_partition * 100: .2f}% - No boxes = {n_boxes_partition}')
        
    print("Prediction for each granularity level:")
    for i in range(len(granular_theta)):
        y_pred_lv = gra_clf.predict(Xtest, level=i)
        acc_lv = accuracy_score(ytest, y_pred_lv)
        n_boxes = gra_clf.get_n_hyperboxes(i)
        print(f'Level {i + 1} - Testing accuracy = {acc_lv * 100: .2f}% - No boxes = {n_boxes}')

    # Explain the predicted results
    # sample_need_explain = 0
    # level_explain = 5
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes = gra_clf.get_sample_explanation_granular_level(Xtest[sample_need_explain], Xtest[sample_need_explain], level_explain)
    # gra_clf.show_sample_explanation(Xtest[sample_need_explain], Xtest[sample_need_explain], min_points_classes, max_points_classes, y_pred_input_0, "2D")
    
    # for i in range(n_partitions):
    #     gra_clf.draw_2D_hyperbox_and_boundary_partitions(f"Partition {i + 1}", i, 100+i)
        
    # for i in range(len(granular_theta)):
    #     gra_clf.draw_2D_hyperbox_and_boundary_granular_level(f"Granularity Level {i + 1}", i)