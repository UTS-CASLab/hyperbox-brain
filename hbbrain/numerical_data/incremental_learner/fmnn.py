"""
Simpson fuzzy min-max neural network classifier trained by the incremental
learning algorithm.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import time
import itertools
from sklearn.metrics import accuracy_score

from hbbrain.base.base_fmnn_estimator import BaseFMNNClassifier
from hbbrain.utils.membership_calc import membership_func_fmnn
from hbbrain.utils.adjust_hyperbox import hyperbox_overlap_test_fmnn, hyperbox_contraction_fmnn
from hbbrain.utils.drawing_func import get_cmap, draw_box
from hbbrain.constants import MARKER_LIST


class FMNNClassifier(BaseFMNNClassifier):
    """
    Simpson fuzzy min-max neural network classifier.

    This class implements an original incremental learning algorithm to train
    a fuzzy min-max neural network classifier. The details of this algorithm
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
    .. [1] P. Simpson, "Fuzzy min—max neural networks—Part 1: Classiﬁcation," 
           IEEE Transactions on Neural Networks, vol. 3, no. 5, pp. 776-786, 1992.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from hbbrain.numerical_data.incremental_learner.fmnn import FMNNClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = FMNNClassifier(theta=0.1).fit(X, y)
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
        Fit the fuzzy min-max neural network classifier according to
        the given training data using the Simpson's original incremental
        learning algorithm.

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
                n_features, "Simpson FMNN - Online learning")
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
                        Vj_new = np.minimum(self.V[j], X[i])
                        Wj_new = np.maximum(self.W[j], X[i])
                        
                        if (Wj_new - Vj_new).sum() <= self.theta * n_features:
                            # adjust the j-th hyperbox
                            self.V[j] = Vj_new
                            self.W[j] = Wj_new
                            id_of_winner_hyperbox = j
                            adjust = True
                        
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

                        # if the ith sample did not fit into any existing hyperboxes, create a new one
                        if not adjust:
                            self.V = np.concatenate(
                                (self.V, X[i].reshape(1, -1)), axis=0)
                            self.W = np.concatenate(
                                (self.W, X[i].reshape(1, -1)), axis=0)
                            self.C = np.concatenate((self.C, [y[i]]))

                            if self.is_draw:
                                # Draw the newly created hyperbox
                                box_color = colors[y[i]]
                                hyperbox = draw_box(drawing_canvas, np.asmatrix(X[i, 0:np.minimum(
                                    n_features, 3)]), np.asmatrix(X[i, 0:np.minimum(n_features, 3)]), box_color)
                                list_drawn_hyperboxes.append(hyperbox[0])
                                self.delay()
                        elif self.V.shape[0] > 1:
                            n_existed_hyperboxes = self.V.shape[0]
                            # test for overlap and hyperbox contraction if needed
                            for ii in range(n_existed_hyperboxes):
                                if (ii != id_of_winner_hyperbox) and self.C[ii] != self.C[id_of_winner_hyperbox]:
                                    # overlap test
                                    case_dim = hyperbox_overlap_test_fmnn(self.V, self.W, id_of_winner_hyperbox, ii)

                                    if case_dim.size > 0:
                                        self.V, self.W = hyperbox_contraction_fmnn(self.V, self.W, case_dim, id_of_winner_hyperbox, ii)
                                        
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
                        (self.V, X[i].reshape(1, -1)), axis=0)
                    self.W = np.concatenate(
                        (self.W, X[i].reshape(1, -1)), axis=0)
                    self.C = np.concatenate((self.C, [y[i]]))

                    if self.is_draw:
                        # Draw the newly created hyperbox
                        box_color = colors[y[i]]
                        hyperbox = draw_box(drawing_canvas, np.asmatrix(X[i, 0:np.minimum(
                            n_features, 3)]), np.asmatrix(X[i, 0:np.minimum(n_features, 3)]), box_color)
                        list_drawn_hyperboxes.append(hyperbox[0])
                        self.delay()

        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start

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

    fmnn_clf = FMNNClassifier(theta=theta, gamma=gamma, is_draw=is_draw)
    fmnn_clf.fit(Xtr, ytr)
    print('Number of hyperboxes = %d'%fmnn_clf.get_n_hyperboxes())
    
    y_pred = fmnn_clf.predict(Xtest)

    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # sample_need_explain = 10
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes = fmnn_clf.get_sample_explanation(Xtest[sample_need_explain])
    # fmnn_clf.show_sample_explanation(Xtest[sample_need_explain], Xtest[sample_need_explain], min_points_classes, max_points_classes, y_pred_input_0, "2D")
    
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/syn_num_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # fmnn_clf.simple_pruning(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%fmnn_clf.get_n_hyperboxes())
    # fmnn_clf.draw_hyperbox_and_boundary()
    
    # y_pred_2 = fmnn_clf.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy (using a Manhattan distance for samples on the boundary) = {acc_pruned * 100: .2f}%')
    
