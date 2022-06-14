"""
Base classes for all fuzzy min-max neural network estimators and their improved versions.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import matplotlib
matplotlib.interactive(True)
from sklearn.metrics import accuracy_score
from hbbrain.base.base_estimator import BaseHyperboxClassifier
from hbbrain.utils.membership_calc import membership_func_fmnn, get_membership_fmnn_all_classes
from hbbrain.utils.dist_metrics import manhattan_distance


def predict_with_manhattan_fmnn(V, W, C, X, g=1):
    """
    Predict class labels for samples in `X`.

    .. note::

        This is a common function to determine the right class labels for `X`
        with regard to a trained hyperbox-based classifier represented by
        `[V, W, C]`. It uses the winner-takes-all principle to predict
        class labels for each sample in X by assigning the class label of the
        sample to the class label of the hyperbox with the maximum membership
        value to that sample. It will use a Manhattan distance in the case of
        many hyperboxes with different classes having the same maximum
        membership value.

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
            Xg_mat = np.ones((len(max_mem_V_id), 1)) * X[i]
            # Find all average points of all hyperboxes with the same membership value
            avg_point_mat = (V[max_mem_V_id] + W[max_mem_V_id]) / 2
            # compute the Manhattan distance from Xg_mat to all average points of all hyperboxes with the same membership value
            maht_dist = manhattan_distance(avg_point_mat, Xg_mat)

            id_min_dist = maht_dist.argmin()
            y_pred[i] = C[max_mem_V_id[id_min_dist]]
        else:
            y_pred[i] = C[max_mem_V_id[0]]
            
    return y_pred


class BaseFMNNClassifier(BaseHyperboxClassifier):
    """
    Base class for all hyperbox-based estimators in hyperbox-brain.
    
    .. note::

        All estimators should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``). This class only initialises
        all common parameters for hyperbox-based estimators.

    Parameters
    ----------
    theta : float or ndarray of shape (n_features,), optional, defaut = 0.5
        A maximum hyperbox size parameter for each dimension.
    gamma : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the membership function in each dimension.
    is_draw : boolean, optional, default = False
        A parameter is used to indicate whether the process of hyperbox building can be dynamically displayed 
        on a canvas or not. This functionality displays hyperboxes in the form of 2D or 3D. In the case that 
        the number of dimensions is higher than 3, only the three features are shown.
    V : array-like of shape (n_hyperboxes, n_features), default = an empty ``ndarray``
        A matrix stores all minimal coordinates of all existing hyperboxes, in which each row is a minimal coordinate of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features), default = an empty ``ndarray``
        A matrix stores all maximal coordinates of all existing hyperboxes, in which each row is a maximal coordinate of a hyperbox.
    C : ndarray of shape (n_hyperboxes,), default = an empty ``ndarray``
        An array contains all class lables for all existing hyperboxes.
        
    Attributes
    ----------
    n_hyperboxes : int 
        Number of hyperboxes built during :term:`fit`.
    
    """

    def __init__(self, theta=0.5, gamma=1, is_draw=False, V=None, W=None, C=None):
        BaseHyperboxClassifier.__init__(self, theta, is_draw, V, W, C)
        self.gamma = gamma

    def predict(self, X):
        """
        Predict class labels for samples in `X`.

        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i`, an additional criterion based on the
            minimum Manhattan distance between the input patter :math:`X_i` and
            the central points of winner hyperboxes are used to find the final
            winner hyperbox that its class label is used for predicting the
            class label of the input pattern :math:`X_i`.

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
        y_pred = predict_with_manhattan_fmnn(self.V, self.W, self.C, X, self.gamma)

        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.
        
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

    def predict_with_membership(self, X):
        """
        Predict class membership values of the input samples X.
        
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
        mem_vals, _ = get_membership_fmnn_all_classes(X, self.V, self.W, self.C, self.gamma)
        
        return mem_vals   
    
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
                # More than one hyperbox with highest membership => using Manhattan distance
                XgT_mat = np.ones((len(max_mem_V_id), 1)) * X_val[i]
                # Find all average points of all hyperboxes with the same membership value
                avg_point_mat = (self.V[max_mem_V_id] + self.W[max_mem_V_id]) / 2
                # compute the manhattan distance from XgT_mat to all average points of all hyperboxes with the same membership value
                maht_dist = manhattan_distance(avg_point_mat, XgT_mat)
            
                id_min_dist = maht_dist.argmin()
                # the id of the selected hyperbox
                id_min_hyperbox = max_mem_V_id[id_min_dist]
                   
                if self.C[id_min_hyperbox] != y_val[i]:
                    hyperboxes_performance[id_min_hyperbox, 1] = hyperboxes_performance[id_min_hyperbox, 1] + 1
                else:
                    hyperboxes_performance[id_min_hyperbox, 0] = hyperboxes_performance[id_min_hyperbox, 0] + 1
                    
        # pruning handling based on the validation results
        n_hyperboxes = hyperboxes_performance.shape[0]
        id_remained_excl_empty_boxes = np.zeros(n_hyperboxes).astype(np.bool)
        id_remained_incl_empty_boxes = np.zeros(n_hyperboxes).astype(np.bool)
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
            
            y_val_pred_excl_empty_boxes = predict_with_manhattan_fmnn(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, X_val, self.gamma)
            y_val_pred_incl_empty_boxes = predict_with_manhattan_fmnn(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, X_val, self.gamma)
            
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
