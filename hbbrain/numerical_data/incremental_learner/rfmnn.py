"""
Refined fuzzy min-max neural network classifier trained by the incremental
learning algorithm.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import time
import itertools
from sklearn.metrics import accuracy_score

from hbbrain.base.base_fmnn_estimator import BaseFMNNClassifier
from hbbrain.utils.membership_calc import membership_func_fmnn, get_membership_fmnn_all_classes
from hbbrain.utils.adjust_hyperbox import is_overlap_diff_labels_num_data_rfmnn, hyperbox_contraction_rfmnn
from hbbrain.utils.drawing_func import get_cmap, draw_box
from hbbrain.utils.dist_metrics import rfmnn_distance
from hbbrain.constants import MARKER_LIST


def predict_rfmnn(V, W, C, X, g):
    """
    Predict class labels for samples in `X`.
    
    .. note::

        This is a function to determine the right class labels for X with regard
        to a trained hyperbox-based classifier represented by `[V, W, C]`. It
        uses the winner-takes-all principle to predict class labels for each
        sample in X by assigning the class label of the sample to the class
        label of the hyperbox with the maximum membership value to that sample.
        It will use a specific distance desgined for the refined fuzzy min-max
        neural networks in the case of many hyperboxes with different classes
        having the same maximum membership value.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points of all hyperboxes of a trained hyperbox-based model, 
        in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points of all hyperboxes of a trained hyperbox-based model, 
        in which each row is a maximal point of a hyperbox.
    C : ndarray of shape (n_hyperboxes,)
        An array contains all class lables for all hyperboxes of a trained hyperbox-based model.
    X : array-like of shape (n_samples, n_features)
        The data matrix contains input patterns for which we want to predict the targets.
    g : float or array-like of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the membership function in each dimension.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        A vector contains the predictions. In binary and multiclass problems, this 
        is a vector containing `n_samples`.

    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
        
    #initialization
    n_samples = X.shape[0]
    y_pred = np.full(n_samples, 0)
    # classifications
    sample_id = 0
    for i in range(n_samples):
        sample_id += 1
        mem_val = membership_func_fmnn(X[i, :], V, W, g) # calculate memberships for all hyperboxes
        
        mem_max = mem_val.max() # get the maximum membership value
        
        if (X[i] < 0).any() == True:
            print(">>> The testing sample %d with the coordinate %s is outside the range [0, 1]. Membership value = %f. The prediction is more likely incorrect." % (sample_id, X[i], mem_max))
            
        max_mem_V_id = np.nonzero(mem_val == mem_max)[0] # get indices of all hyperboxes with the maximum membership values
        
        if len(np.unique(C[max_mem_V_id])) > 1:
            # compute the rfmnn distance from input X to all winner hyperboxes
            dist = rfmnn_distance(X[i], V[max_mem_V_id], W[max_mem_V_id])
            id_min_dist = dist.argmin()
            y_pred[i] = C[max_mem_V_id[id_min_dist]]
        else:
            y_pred[i] = C[max_mem_V_id[0]]
            
    return y_pred


class RFMNNClassifier(BaseFMNNClassifier):
    """
    Refined fuzzy min-max neural network classifier.
    
    This class implements essential functions for a refined online learning
    algorithm to train a fuzzy min-max neural network. This algorithm proposes
    a new expansion procedure for addressing the problems of overlap leniency
    and irregularity of hyperbox expansion. It avoids the overlap cases between
    hyperboxes from different classes, reducing the number of overlap cases to
    one (containment case) as in the improved online learning algorithm. It
    also introduces a new formula that simplifies the overlap test procedure.
    Moreover, it introduces a new contraction procedure for overcoming the
    data distortion problem and providing more accurate decision boundaries
    for the contracted hyperboxes is proposed. The details of this algorithm
    can be found in [1]_.

    Parameters
    ----------
    theta : float, optional, default=0.5
        Maximum hyperbox size for numerical features.
    gamma : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous feature.
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
    elapsed_training_time : float
        Training time in seconds.

    References
    ----------
    .. [1] O. N. Al-Sayaydeh, M. F. Mohammed, E. Alhroob, H. Tao, and C. P. Lim,
           "A refined fuzzy min-max neural network with new learning procedures
           for pattern classification," IEEE Transactions on Fuzzy Systems,
           vol. 28, no. 10, pp. 2480-2494, 2019.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from hbbrain.numerical_data.incremental_learner.rfmnn import RFMNNClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = RFMNNClassifier(theta=0.1).fit(X, y)
    >>> clf.predict(X[[10, 50, 100]])
    array([0, 1, 2])

    """

    def __init__(self, theta=0.5, gamma=1, is_draw=False, V=None, W=None, C=None):
        BaseFMNNClassifier.__init__(self, theta, gamma, is_draw, V, W, C)

    def _init_data(self):
        """
        Initialise default values for coordinates of hyperboxes.

        Returns
        -------
        None.

        """
        self._init_hyperboxes()

    def fit(self, X, y):
        """
        Fit the model according to the given training data using the refined
        online learning algorithm.

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
            Fitted fuzzy min-max neural network.

        """
        y = y.astype('int')
        self._init_data()

        n_samples, n_features = X.shape
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
                n_features, "RFMNN - Refined Online learning Algorithm")
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

        for i in range(n_samples):
            if self.is_draw:
                # draw input samples
                color_ = colors[y[i]]
                marker_ = markers[y[i]]
                if n_features == 2:
                    input_points = drawing_canvas.plot(
                        X[i, 0], X[i, 1], color=color_, marker=marker_)
                else:
                    input_points = drawing_canvas.plot(
                        [X[i, 0]], [X[i, 1]], [X[i, 2]], color=color_, marker=marker_)
                self.delay(0.11)
                # remove input point to create hyperboxes
                input_points[0].remove()

            # Training loop
            if self.V.size == 0:
                # no model provided - starting from scratch
                self.V = np.array([X[i]])
                self.W = np.array([X[i]])
                self.C = np.array([y[i]])

                if self.is_draw == True:
                    # draw hyperbox
                    box_color = colors[y[i]]
                    hyperbox = draw_box(drawing_canvas, np.asmatrix(self.V[0, 0:np.minimum(
                        n_features, 3)]), np.asmatrix(self.W[0, 0:np.minimum(n_features, 3)]), box_color)
                    list_drawn_hyperboxes.append(hyperbox[0])
                    self.delay()
            else:
                id_same_input_label_group = np.nonzero(self.C == y[i])[0]
                id_diff_input_label_group = np.nonzero(self.C != y[i])[0]
                
                V_diff = self.V[id_diff_input_label_group]
                W_diff = self.W[id_diff_input_label_group]
                
                is_create_new_hyperbox = False

                if len(id_same_input_label_group) > 0:
                    V_sameX = self.V[id_same_input_label_group]
                    W_sameX = self.W[id_same_input_label_group]
                    
                    b = membership_func_fmnn(X[i], V_sameX, W_sameX, self.gamma)
                    
                    max_mem_id = np.argmax(b)
                    
                    # store the index of the winner hyperbox in the list of all hyperboxes of all classes
                    j = id_same_input_label_group[max_mem_id]
                    
                    if b[max_mem_id] != 1:
                        adjust = False
                        
                        # Check for violation of max hyperbox size and class labels
                        V_cmp = np.minimum(self.V[j], X[i])
                        W_cmp = np.maximum(self.W[j], X[i])
                        
                        if ((W_cmp - V_cmp) <= self.theta).all() == True:
                            if is_overlap_diff_labels_num_data_rfmnn(V_diff, W_diff, V_cmp, W_cmp, False) == False:
                                # adjust the j-th hyperbox
                                self.V[j] = V_cmp
                                self.W[j] = W_cmp
                                adjust = True
                        
                                if self.is_draw:
                                    # Drawing hyperboxes
                                    box_color = colors[self.C[j]]
                                    try:
                                        list_drawn_hyperboxes[j].remove()
                                    except:
                                        print("Bug remove box")
                                        pass
    
                                    hyperbox = draw_box(drawing_canvas, np.asmatrix(self.V[j, 0:np.minimum(
                                        n_features, 3)]), np.asmatrix(self.W[j, 0:np.minimum(n_features, 3)]), box_color)
                                    list_drawn_hyperboxes[j] = hyperbox[0]
                                    self.delay()

                        # if the ith sample did not fit into any existing hyperboxes, create a new one
                        if not adjust:
                            self.V = np.concatenate(
                                (self.V, X[i].reshape(1, -1)), axis=0)
                            self.W = np.concatenate(
                                (self.W, X[i].reshape(1, -1)), axis=0)
                            self.C = np.concatenate((self.C, [y[i]]))
                            is_create_new_hyperbox = True
                else:
                    self.V = np.concatenate(
                        (self.V, X[i].reshape(1, -1)), axis=0)
                    self.W = np.concatenate(
                        (self.W, X[i].reshape(1, -1)), axis=0)
                    self.C = np.concatenate((self.C, [y[i]]))
                    is_create_new_hyperbox = True
                    
                if is_create_new_hyperbox == True:
                    if self.is_draw:
                        # Draw the newly created hyperbox
                        box_color = colors[y[i]]
                        hyperbox = draw_box(drawing_canvas, np.asmatrix(X[i, 0:np.minimum(
                            n_features, 3)]), np.asmatrix(X[i, 0:np.minimum(n_features, 3)]), box_color)
                        list_drawn_hyperboxes.append(hyperbox[0])
                        self.delay()
                    
                    if len(id_diff_input_label_group) > 0:
                        is_ovl, hyperbox_ids_overlap, min_overlap_dimensions = is_overlap_diff_labels_num_data_rfmnn(V_diff, W_diff, self.V[-1], self.W[-1], True)
                        if is_ovl == True:
                            # convert hyperbox_ids_overlap of hyperboxes with other classes to ids of all existing hyperboxes
                            hyperbox_ids_overlap = id_diff_input_label_group[hyperbox_ids_overlap]
                            # do contraction for parent hyperboxes with indices contained in hyperbox_ids_overlap
                            self.V, self.W, self.C = hyperbox_contraction_rfmnn(self.V, self.W, self.C, hyperbox_ids_overlap, -1, min_overlap_dimensions)
                            
                            if self.is_draw:
                                n_existed_hyperboxes = len(self.C)
                                # draw existing hyperboxes
                                color_ = np.array(['k'] * n_existed_hyperboxes, dtype=object)
                                for c in range(n_existed_hyperboxes):
                                    color_[c] = colors[self.C[c]]
                                drawing_canvas.cla()
                                hyperboxes = draw_box(drawing_canvas, self.V[:, 0:np.minimum(
                                    n_features, 3)], self.W[:, 0:np.minimum(n_features, 3)], color_)
                                
                                list_drawn_hyperboxes = list()
                                list_drawn_hyperboxes.extend(hyperboxes)
                                self.delay()

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
            minimum rfmnn distance between the input patter :math:`X_i` and the
            winner hyperboxes are used to find the final winner hyperbox that
            its class label is used for predicting the class label of the input
            pattern :math:`X_i`.

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
        y_pred = predict_rfmnn(self.V, self.W, self.C, X, self.gamma)
                
        return y_pred

    def simple_pruning(self, X_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Simply prune low qualitied hyperboxes based on a pre-defined accuracy threshold for each hyperbox

        Parameters
        ----------
        X_val : array-like of shape (n_samples, n_features)
            The data matrix contains validation patterns.
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
        n_samples = X_val.shape[0]
        rnd = np.random
        rnd.seed(0)
        # Matrices storing the classification accuracy for each created hyperbox in the trained model
        # The first column stores the number of corrected classification samples and the second column stores the number of wrong classification samples
        hyperboxes_performance = np.zeros((len(self.C), 2))
        
        for i in range(n_samples):
            mem_val = membership_func_fmnn(X_val[i], self.V, self.W, self.gamma) # calculate memberships for all hyperboxes
            
            mem_max = mem_val.max() # get max membership value
            max_mem_V_id = np.nonzero(mem_val == mem_max)[0]  # get indexes of all hyperboxes with max membership
            
            if len(max_mem_V_id) == 1:
                # Only one hyperbox with the highest membership function
                if self.C[max_mem_V_id[0]] == y_val[i]:
                    hyperboxes_performance[max_mem_V_id[0], 0] = hyperboxes_performance[max_mem_V_id[0], 0] + 1                 
                else:
                    hyperboxes_performance[max_mem_V_id[0], 1] = hyperboxes_performance[max_mem_V_id[0], 1] + 1
            else:
                # More than one hyperbox with highest membership
                # compute the rfmnn distance from input X to all winner hyperboxes
                dist = rfmnn_distance(X_val[i], self.V[max_mem_V_id], self.W[max_mem_V_id])
                id_min_dist = dist.argmin()
            
                # the id of the selected hyperbox
                id_min_hyperbox = max_mem_V_id[id_min_dist]
                   
                if self.C[id_min_hyperbox] != y_val[i]:
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
            
            y_val_pred_excl_empty_boxes = predict_rfmnn(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, X_val, self.gamma)
            y_val_pred_incl_empty_boxes = predict_rfmnn(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, X_val, self.gamma)
            
            if (accuracy_score(y_val, y_val_pred_excl_empty_boxes) >= accuracy_score(y_val, y_val_pred_incl_empty_boxes)):
                self.V = V_pruned_excl_empty_boxes
                self.W = W_pruned_excl_empty_boxes
                self.C = C_pruned_excl_empty_boxes
            else:
                self.V = V_pruned_incl_empty_boxes
                self.W = W_pruned_incl_empty_boxes
                self.C = C_pruned_incl_empty_boxes
                
            return self

    def get_sample_explanation(self, x):
        """
        Get useful information for explaining the reason behind the predicted result for the input pattern

        Parameters
        ----------
        x : ndarray of shape (n_feature,)
            The input pattern which needs to be explained.
        
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
        mem_vals_for_classes, hyperbox_id_for_classes = get_membership_fmnn_all_classes(x, self.V, self.W, self.C, self.gamma)
        class_values = np.unique(self.C)
        # get predicted class label for the input sample
        y_pred = self.predict(x)[0]
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

    # Optional arguments
    optional.add_argument('--theta', type=float, default=0.5,
                          help='Maximum hyperbox size (in the range of (0, 1]) (default: 0.5)')
    optional.add_argument('--gamma', type=float, default=1,
                          help='A sensitivity parameter describing the speed of decreasing of the membership function in each dimension (larger than 0) (default: 1)')
    optional.add_argument('--is_draw', type=str2bool, default=False,
                          help='Show the existing hyperboxes during the training process on the screen (default: False)')

    args = parser.parse_args()

    if args.theta <= 0 or args.theta > 1:
        parser.error("--theta has to be in the range of (0, 1]")

    if args.gamma <= 0:
        parser.error("--gamma has to be larger than 0")

    gamma = args.gamma
    theta = args.theta
    is_draw = args.is_draw
    training_file = args.training_file
    testing_file = args.testing_file
    
    import pandas as pd
    df_train = pd.read_csv(training_file, header=None)
    df_test = pd.read_csv(testing_file, header=None)

    Xy_train = df_train.to_numpy()
    Xy_test = df_test.to_numpy()

    Xtr = Xy_train[:, :-1]
    ytr = Xy_train[:, -1]

    Xtest = Xy_test[:, :-1]
    ytest = Xy_test[:, -1]

    rfmnn_clf = RFMNNClassifier(theta=theta, gamma=gamma, is_draw=is_draw)
    rfmnn_clf.fit(Xtr, ytr)
    print('Number of hyperboxes = %d'%rfmnn_clf.get_n_hyperboxes())
    
    y_pred = rfmnn_clf.predict(Xtest)

    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # sample_need_explain = 10
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes = rfmnn_clf.get_sample_explanation(Xtest[sample_need_explain])
    # rfmnn_clf.show_sample_explanation(Xtest[sample_need_explain], Xtest[sample_need_explain], min_points_classes, max_points_classes, y_pred_input_0, "2D")
    
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/syn_num_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # rfmnn_clf.simple_pruning(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%rfmnn_clf.get_n_hyperboxes())
    # rfmnn_clf.draw_hyperbox_and_boundary()
    
    # y_pred_2 = rfmnn_clf.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy (using a RFMNN distance for samples on the boundary) = {acc_pruned * 100: .2f}%')
    