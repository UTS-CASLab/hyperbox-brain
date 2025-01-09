"""
General fuzzy min-max neural network trained by the original incremental learning algorithm.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

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
from hbbrain.utils.membership_calc import membership_func_gfmm, get_membership_gfmm_all_classes
from hbbrain.utils.adjust_hyperbox import overlap_resolving_num_data, is_two_hyperboxes_overlap_num_data_general
from hbbrain.utils.drawing_func import get_cmap, draw_box
from hbbrain.utils.dist_metrics import manhattan_distance, manhattan_distance_with_missing_val
from hbbrain.constants import UNLABELED_CLASS, MARKER_LIST


class OnlineGFMM(BaseGFMMClassifier):
    """
    General fuzzy min-max neural network model using the original incremental
    learning algorithm.

    This class implements the original online learning algorithm to train the
    general fuzzy min-max neural network. The details of this algorithm can
    be found in [1]_.

    .. note::

        This implementation uses the accelerated mechanism presented in [2]_ to
        accelerate the improved online learning algorithm. Compared to the
        original online learning algorithm proposed in [1]_, this implementation
        uses the similarity measure between two hyperboxes by shortest gap
        distance presented in [3]_ for overlap test. In addition, we extend the
        number of hyperbox contraction cases from four in the original algorithm
        to eight cases aiming to cover more overlapping cases between two
        hyperboxes.

    Parameters
    ----------
    theta : float, optional, default=0.5
        Maximum hyperbox size for numerical features.
    theta_min : float, optional, default=1
        Minimum value of the maximum hyperbox size for continuous features so
        that the training loop is still performed. If the value of `theta_min`
        is larger than the value of `theta`, it will be automatically assigned
        a value equal to `theta`.
    gamma : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous feature.
    alpha : float, optional, default=0.9
        Multiplier factor to reduce the value of maximum hyperbox size after
        each training loop.
    is_draw : boolean, optional, default=False
        Whether the construction of hyperboxes can be progressively shown
        during the training process on a canvas window.
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points for numerical features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points for numerical features of all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    C : array-like of shape (n_hyperboxes,)
        A vector stores all class labels correponding to existing hyperboxes.

    Attributes
    ----------
    is_exist_missing_value : boolean
        Is there any missing values in continuous features in the training data.
    elapsed_training_time : float
        Training time in seconds.
    n_passes : int
        Number of training loops.

    References
    ----------
    .. [1] B. Gabrys and A. Bargiela, "General fuzzy min-max neural network for
           clustering and classification," IEEE Transactions on Neural Networks,
           vol. 11, no. 3, pp. 769-783, 2000.
    .. [2] T.T. Khuat and B. Gabrys, "Accelerated learning algorithms of general fuzzy min-max neural network using a novel hyperbox selection rule,"
           Information Sciences, vol. 547, pp. 887-909, 2021.
    .. [3] B. Gabrys, "Agglomerative learning algorithms for general fuzzy
           min-max neural network", Journal of VLSI Signal Processing Systems
           for Signal, Image and Video Technology, vol. 32, no. 1, pp. 67-82, 2002.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
    >>> X, y = load_iris(return_X_y=True)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = OnlineGFMM(theta=0.1).fit(X, y)
    >>> clf.predict(X[[10, 50, 100]])
    array([0, 1, 2])

    """

    def __init__(self, theta=0.5, theta_min=1, gamma=1, alpha=0.9, is_draw=False, V=None, W=None, C=None):
        BaseGFMMClassifier.__init__(self, theta=theta, gamma=gamma, is_draw=is_draw, V=V, W=W, C=C)
        self.theta_min = theta_min
        self.alpha = alpha

    def _init_data(self):
        """
        Initialise default values for coordinates of hyperboxes and other
        parameters.

        Returns
        -------
        None.

        """
        if (self.theta_min > self.theta):
            self.theta_min = self.theta
        self._init_hyperboxes()

    def fit(self, X, y):
        """
        Fit the model according to the given training data using the original
        incremental learning algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Fitted general fuzzy min-max neural network.

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
            return self._fit(Xl, Xu, y)
        else:
            return self._fit(X, X, y)

    def _fit(self, Xl, Xu, y):
        """
        Fit the model according to the given training data using the original
        incremental learning algorithm. The input data are provided in the
        form of hyperboxes.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            Lower bounds of training features.
        Xu : array-like of shape (n_samples, n_features)
            Upper bounds of training features.
        y : array-like of shape (n_samples,)
            Target vector relative to input hyperboxes.

        Returns
        -------
        self : object
            Fitted general fuzzy min-max neural network.

        """
        self._init_data()

        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            self.is_exist_missing_value = True
            Xl, Xu, y = convert_format_missing_input_zero_one(Xl, Xu, y)
        else:
            self.is_exist_missing_value = False
            
        if is_contain_missing_value(y) == True:
            y = np.where(np.isnan(y), UNLABELED_CLASS, y)

        n_samples, n_features = Xl.shape
        class_ids = np.unique(y)  # list of class labels of input patterns
        if len(self.C) > 0:
            # there are pre-trained hyperboxes, we need to add the class labels to the current list of labels if they are not existed in this list
            existed_class_ids = np.unique(self.C)
            class_ids = np.append(class_ids, existed_class_ids)
            class_ids = np.unique(class_ids)

        n_classes = len(class_ids)

        time_start = time.perf_counter()

        if self.is_draw:
            marker_map = itertools.cycle(MARKER_LIST)
            color_map = get_cmap(n_classes)
            # build a dictionary with the class label being key and color being value
            colors = {}
            # build a dictionary of markers corresponding to class labels. Key: class labels, value: marker type
            markers = {}
            for i in range(n_classes):
                colors[class_ids[i]] = color_map(i)
                markers[class_ids[i]] = next(marker_map)

            list_drawn_hyperboxes = list()

            drawing_canvas = self.initialise_canvas_graph(
                n_features, "GFMM - Original Online learning")
            n_existed_hyperboxes = len(self.C)

            if n_existed_hyperboxes > 0:
                # draw existing hyperboxes
                color_ = np.array(['k'] * n_existed_hyperboxes, dtype=object)
                for c in range(n_existed_hyperboxes):
                    color_[c] = colors[self.C[c]]
                hyperboxes = draw_box(drawing_canvas, self.V[:, 0:np.minimum(
                    n_features, 3)], self.W[:, 0:np.minimum(n_features, 3)], color_)
                list_drawn_hyperboxes.extend(hyperboxes)
                self.delay()

        theta = self.theta
        training_acc = 0
        self.n_passes = 0
        while theta >= self.theta_min and training_acc < 1:
            self.n_passes += 1
            #print("Training data pass = %d" % self.n_passes)
            # Loop through each training input pattern
            threshold_mem_val = 1 - np.max(self.gamma) * theta
            for i in range(n_samples):
                if self.is_draw:
                    # draw input samples
                    color_ = colors[y[i]]

                    if (Xl[i, :] == Xu[i, :]).all():
                        # input samples are points not hyperboxes
                        marker_ = markers[y[i]]
                        if n_features == 2:
                            input_points = drawing_canvas.plot(
                                Xl[i, 0], Xl[i, 1], color=color_, marker=marker_)
                        else:
                            input_points = drawing_canvas.plot(
                                [Xl[i, 0]], [Xl[i, 1]], [Xl[i, 2]], color=color_, marker=marker_)
                    else:
                        input_points = draw_box(drawing_canvas, np.asmatrix(Xl[i, 0:np.minimum(
                            n_features, 3)]), np.asmatrix(Xu[i, 0:np.minimum(n_features, 3)]), color_)
                    self.delay(0.11)
                    # remove input point to create hyperboxes
                    input_points[0].remove()

                # Training loop
                if self.V.size == 0:
                    # no model provided - starting from scratch
                    self.V = np.array([Xl[i]])
                    self.W = np.array([Xu[i]])
                    self.C = np.array([y[i]])

                    if self.is_draw == True:
                        # draw hyperbox
                        box_color = colors[y[i]]
                        hyperbox = draw_box(drawing_canvas, np.asmatrix(self.V[0, 0:np.minimum(
                            n_features, 3)]), np.asmatrix(self.W[0, 0:np.minimum(n_features, 3)]), box_color)
                        list_drawn_hyperboxes.append(hyperbox[0])
                        self.delay()
                else:
                    if y[i] == UNLABELED_CLASS:
                        id_same_input_label_group = np.ones(len(self.C), dtype=bool)
                    else:
                        id_same_input_label_group = (self.C == y[i]) | (
                            self.C == UNLABELED_CLASS)

                    if id_same_input_label_group.any() == True:
                        # if we have small number of hyperboxes with low dimension, this operation takes more time compared to computing membership value with all hyperboxes and ignore
                        # hyperboxes with different class (the membership computation on small dimensionality is so rapidly). However, if we have hyperboxes with high dimensionality,
                        # the membership computing on all hyperboxes take so long => The reduction to only hyperboxes with the
                        # same class will significantly decrease the running time
                        V_sameX = self.V[id_same_input_label_group]
                        W_sameX = self.W[id_same_input_label_group]
                        # contain both class label as same as the input pattern and unlabelled
                        lb_sameX = self.C[id_same_input_label_group]
                        id_range = np.arange(len(self.C))
                        # determine the indices of samples with the same class label as the input sample
                        id_processing = id_range[id_same_input_label_group]

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
                            for j in id_processing[id_descending_mem_val]:
                                if b[id_descending_mem_val[count]] < threshold_mem_val:
                                    break

                                count = count + 1
                                # Check for violation of max hyperbox size and class labels
                                Vj_new = np.minimum(self.V[j], Xl[i])
                                Wj_new = np.maximum(self.W[j], Xu[i])
                                if (y[i] == self.C[j] or self.C[j] == UNLABELED_CLASS or y[i] == UNLABELED_CLASS) and (((Wj_new - Vj_new) <= theta).all() == True):
                                    # adjust the j-th hyperbox
                                    self.V[j] = Vj_new
                                    self.W[j] = Wj_new
                                    id_of_winner_hyperbox = j
                                    adjust = True
                                    if (y[i] != UNLABELED_CLASS) and (self.C[j] == UNLABELED_CLASS):
                                        self.C[j] = y[i]

                                    if self.is_draw:
                                        # Drawing hyperboxes
                                        box_color = colors[self.C[j]]
                                        try:
                                            list_drawn_hyperboxes[j].remove()
                                        except:
                                            pass

                                        hyperbox = draw_box(drawing_canvas, np.asmatrix(self.V[j, 0:np.minimum(
                                            n_features, 3)]), np.asmatrix(self.W[j, 0:np.minimum(n_features, 3)]), box_color)
                                        list_drawn_hyperboxes[j] = hyperbox[0]
                                        self.delay()
                                    # found out the winner hyperbox to adjust => break the loop
                                    break

                            # if the ith sample did not fit into any existing hyperboxes, create a new one
                            if not adjust:
                                self.V = np.concatenate(
                                    (self.V, Xl[i].reshape(1, -1)), axis=0)
                                self.W = np.concatenate(
                                    (self.W, Xu[i].reshape(1, -1)), axis=0)
                                self.C = np.concatenate((self.C, [y[i]]))

                                if self.is_draw:
                                    # Draw the newly created hyperbox
                                    box_color = colors[y[i]]
                                    hyperbox = draw_box(drawing_canvas, np.asmatrix(Xl[i, 0:np.minimum(
                                        n_features, 3)]), np.asmatrix(Xu[i, 0:np.minimum(n_features, 3)]), box_color)
                                    list_drawn_hyperboxes.append(hyperbox[0])
                                    self.delay()
                            elif self.V.shape[0] > 1:
                                n_existed_hyperboxes = self.V.shape[0]
                                # test for overlap and hyperbox contraction if needed
                                for ii in range(n_existed_hyperboxes):
                                    if (ii != id_of_winner_hyperbox) and (self.C[ii] != self.C[id_of_winner_hyperbox] or self.C[id_of_winner_hyperbox] == UNLABELED_CLASS):
                                        # overlap test
                                        is_overlap = is_two_hyperboxes_overlap_num_data_general(
                                            self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii])

                                        if is_overlap == True:
                                            self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.V[ii], self.W[ii] = overlap_resolving_num_data(
                                                self.V[id_of_winner_hyperbox], self.W[id_of_winner_hyperbox], self.C[id_of_winner_hyperbox], self.V[ii], self.W[ii], self.C[ii])
                                            if self.is_draw:
                                                # Draw the adjusted hyperboxes
                                                boxii_color = colors[self.C[ii]]
                                                boxwin_color = colors[self.C[id_of_winner_hyperbox]]

                                                try:
                                                    list_drawn_hyperboxes[ii].remove()
                                                    list_drawn_hyperboxes[id_of_winner_hyperbox].remove()
                                                except:
                                                    pass

                                                hyperboxes = draw_box(drawing_canvas, self.V[[ii, id_of_winner_hyperbox], 0:np.minimum(n_features, 3)], self.W[[ii, id_of_winner_hyperbox], 0:np.minimum(n_features, 3)], [boxii_color, boxwin_color])
                                                list_drawn_hyperboxes[ii] = hyperboxes[0]
                                                list_drawn_hyperboxes[id_of_winner_hyperbox] = hyperboxes[1]
                                                self.delay()
                    else:
                        # There are no existing hyperboxes representing the same class label as the input patter
                        # We need to create a new hyperbox for the input sample
                        self.V = np.concatenate(
                            (self.V, Xl[i].reshape(1, -1)), axis=0)
                        self.W = np.concatenate(
                            (self.W, Xu[i].reshape(1, -1)), axis=0)
                        self.C = np.concatenate((self.C, [y[i]]))

                        if self.is_draw:
                            # Draw the newly created hyperbox
                            box_color = colors[y[i]]
                            hyperbox = draw_box(drawing_canvas, np.asmatrix(Xl[i, 0:np.minimum(
                                n_features, 3)]), np.asmatrix(Xu[i, 0:np.minimum(n_features, 3)]), box_color)
                            list_drawn_hyperboxes.append(hyperbox[0])
                            self.delay()

            theta = theta * self.alpha
            if (theta >= self.theta_min):
                # Predict the training set again
                y_pred = self._predict(Xl, Xu)
                training_acc = accuracy_score(y, y_pred)
                print("Training accuracy = %f" % training_acc)

        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start

        return self

    def simple_pruning(self, Xl_val, Xu_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Simply prune low qualitied hyperboxes based on a pre-defined accuracy threshold for each hyperbox

        Parameters
        ----------
        Xl_val : array-like of shape (n_samples, n_features)
            The data matrix contains lower bounds of validation patterns.
        Xu_val : array-like of shape (n_samples, n_features)
            The data matrix contains upper bounds of validation patterns.
        y_val : ndarray of shape (n_samples,)
            A vector contains the true class label corresponding to each validation pattern.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction process on the validation set.
            If True, keep them, else the decision for keeping or removing based on the classification accuracy on the validation dataset

        Returns
        -------
        self
            A hyperbox-based model with the low-qualitied hyperboxes pruned.

        """
        n_samples = Xl_val.shape[0]
        rnd = np.random
        rnd.seed(0)
        # Matrices storing the classification accuracy for each created hyperbox in the trained model
        # The first column stores the number of corrected classification samples and the second column stores the number of wrong classification samples
        hyperboxes_performance = np.zeros((len(self.C), 2))
        
        if (is_contain_missing_value(Xl_val) == True) or (is_contain_missing_value(Xu_val) == True):
            Xl_val, Xu_val, y_val = convert_format_missing_input_zero_one(Xl_val, Xu_val, y_val)
            
        for i in range(n_samples):
            if self.is_exist_missing_value == False:
                mem_val = membership_func_gfmm(Xl_val[i], Xu_val[i], self.V, self.W, self.gamma) # calculate memberships for all hyperboxes
            else:
                mem_val = membership_func_gfmm(Xl_val[i], Xu_val[i], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.gamma)
            
            bmax = mem_val.max() # get max membership value
            max_mem_V_id = np.nonzero(mem_val == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(max_mem_V_id) == 1:
                # Only one hyperbox with the highest membership function
                if self.C[max_mem_V_id[0]] == y_val[i]:
                    hyperboxes_performance[max_mem_V_id[0], 0] = hyperboxes_performance[max_mem_V_id[0], 0] + 1                 
                else:
                    hyperboxes_performance[max_mem_V_id[0], 1] = hyperboxes_performance[max_mem_V_id[0], 1] + 1
            else:
                # More than one hyperbox with highest membership => using Manhattan distance
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
            self.V = self.V[id_remained_incl_empty_boxes]
            self.W = self.W[id_remained_incl_empty_boxes]
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
               
            V_pruned_excl_empty_boxes = self.V[id_remained_excl_empty_boxes]
            W_pruned_excl_empty_boxes = self.W[id_remained_excl_empty_boxes]
            C_pruned_excl_empty_boxes = self.C[id_remained_excl_empty_boxes]
            
            W_pruned_incl_empty_boxes = self.W[id_remained_incl_empty_boxes]
            V_pruned_incl_empty_boxes = self.V[id_remained_incl_empty_boxes]
            C_pruned_incl_empty_boxes = self.C[id_remained_incl_empty_boxes]
            
            y_val_pred_excl_empty_boxes = predict_with_manhattan(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, Xl_val, Xu_val, self.gamma)
            y_val_pred_incl_empty_boxes = predict_with_manhattan(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, Xl_val, Xu_val, self.gamma)
            
            if (accuracy_score(y_val, y_val_pred_excl_empty_boxes) >= accuracy_score(y_val, y_val_pred_incl_empty_boxes)):
                self.V = V_pruned_excl_empty_boxes
                self.W = W_pruned_excl_empty_boxes
                self.C = C_pruned_excl_empty_boxes
            else:
                self.V = V_pruned_incl_empty_boxes
                self.W = W_pruned_incl_empty_boxes
                self.C = C_pruned_incl_empty_boxes
                
        return self

    def get_sample_explanation(self, xl, xu):
        """
        Get useful information for explaining the reason behind the predicted result for the input pattern

        Parameters
        ----------
        xl : ndarray of shape (n_feature,)
            Minimum point of the input pattern which needs to be explained.
        xu : ndarray of shape (n_feature,)
            Maximum point of the input pattern which needs to be explained.
        
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
        mem_vals_for_classes, hyperbox_id_for_classes = get_membership_gfmm_all_classes(xl, xu, self.V, self.W, self.C, self.gamma)
        class_values = np.unique(self.C)
        # get predicted class label for the input sample
        y_pred = self._predict(xl, xu)[0]
        # create dictionaries with keys being class labels and values being membership values, maximum and minimum points
        dict_mem_val_classes = {}
        dict_min_point_classes = {}
        dict_max_point_classes = {}
        for _id, c in enumerate(class_values):
            dict_mem_val_classes[c] = mem_vals_for_classes[0][_id]
            box_id = hyperbox_id_for_classes[0][_id]
            dict_min_point_classes[c] = self.V[box_id]
            dict_max_point_classes[c] = self.W[box_id]

        return(y_pred, dict_mem_val_classes, dict_min_point_classes, dict_max_point_classes)


if __name__ == '__main__':

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

    # Optional arguments
    optional.add_argument('--theta', type=float, default=0.5,
                          help='Maximum hyperbox size (in the range of (0, 1]) (default: 0.5)')
    optional.add_argument('--theta_min', type=float, default=0.5,
                          help='Mimimum value of the maximum hyperbox size to escape the training loop (in the range of (0, 1]) (default: 0.5)')
    optional.add_argument('--gamma', type=float, default=1,
                          help='A sensitivity parameter describing the speed of decreasing of the membership function in each dimension (larger than 0) (default: 1)')
    optional.add_argument('--alpha', type=float, default=0.9,
                          help='Multiplier showing the decrease of theta in each step (default: 0.9)')
    optional.add_argument('--is_draw', type=str2bool, default=False,
                          help='Show the existing hyperboxes during the training process on the screen (default: False)')

    args = parser.parse_args()

    if args.theta <= 0 or args.theta > 1:
        parser.error("--theta has to be in the range of (0, 1]")

    if args.theta_min <= 0 or args.theta_min > 1:
        parser.error("--theta_min has to be in the range of (0, 1]")

    if args.alpha <= 0 or args.alpha >= 1:
        parser.error("--alpha has to be in the range of (0, 1)")

    if args.gamma <= 0:
        parser.error("--gamma has to be larger than 0")

    gamma = args.gamma
    theta = args.theta
    theta_min = args.theta_min
    is_draw = args.is_draw
    alpha = args.alpha
    training_file = args.training_file
    testing_file = args.testing_file

    df_train = pd.read_csv(training_file, header=None)
    df_test = pd.read_csv(testing_file, header=None)

    Xy_train = df_train.to_numpy()
    Xy_test = df_test.to_numpy()

    Xtr = Xy_train[:, :-1]
    ytr = Xy_train[:, -1]

    Xtest = Xy_test[:, :-1]
    ytest = Xy_test[:, -1]

    onln_gfmm_clf = OnlineGFMM(
        theta=theta, theta_min=theta_min, gamma=gamma, alpha=alpha, is_draw=is_draw)
    onln_gfmm_clf.fit(Xtr, ytr)
    print('Number of hyperboxes = %d'%onln_gfmm_clf.get_n_hyperboxes())
    
    y_pred = onln_gfmm_clf.predict(Xtest)

    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # sample_need_explain = 10
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes = onln_gfmm_clf.get_sample_explanation(Xtest[sample_need_explain], Xtest[sample_need_explain])
    # onln_gfmm_clf.show_sample_explanation(Xtest[sample_need_explain], Xtest[sample_need_explain], min_points_classes, max_points_classes, y_pred_input_0, "par_cord")
    
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/syn_num_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # onln_gfmm_clf.simple_pruning(X_val, X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%onln_gfmm_clf.get_n_hyperboxes())
    # onln_gfmm_clf.draw_hyperbox_and_boundary()
    
    # y_pred_2 = onln_gfmm_clf.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy (using a probability measure for samples on the boundary) = {acc_pruned * 100: .2f}%')
    
    # from sklearn.datasets import load_iris, load_breast_cancer
    # X, y = load_iris(return_X_y=True)
    # #X = X[:, [0, 1]]
    # from sklearn.model_selection import train_test_split
    
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    # print("No training samples = ", X_train.shape[0])
    # clf = OnlineGFMM(theta=0.1, theta_min=1, is_draw=True).fit(X_train, y_train)
    # print('Number of hyperboxes = %d'%clf.get_n_hyperboxes())
    
    # y_pred = clf.predict(X_test)

    # acc = accuracy_score(y_test, y_pred)
    # print(f'Testing accuracy = {acc * 100: .2f}%')
