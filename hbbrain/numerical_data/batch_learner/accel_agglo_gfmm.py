"""
General fuzzy min-max neural network trained by the accelerated agglomerative
learning algorithm.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: BSD 3 clause
import numpy as np
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
from hbbrain.utils.membership_calc import membership_func_gfmm, asym_similarity_val_one_many_hyperboxes, get_membership_gfmm_all_classes
from hbbrain.utils.adjust_hyperbox import is_overlap_one_many_hyperboxes_num_data_general
from hbbrain.utils.drawing_func import get_cmap, draw_box
from hbbrain.constants import UNLABELED_CLASS, MARKER_LIST, PROBABILITY_MEASURE, MANHATTAN_DIS


class AccelAgglomerativeLearningGFMM(BaseGFMMClassifier):
    """Accelerated agglomerative learning algorithm for a general fuzzy min-max
    neural network with numerical data.

    See [1]_ for more detailed information regarding this learning algorithm.

    Parameters
    ----------
    theta : float, optional, default=0.5
        Maximum hyperbox size for numerical features.
    gamma : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous feature.
    min_simil : float, optional, default=0.5
        Minimum similarity threshold so that two hyperboxes are agglomerated.
    simil_measure : {'short', 'long', 'mid'}, optional, default='mid'
        Type of similarity measures is used to compute similarity between
        two hyperboxes. It can get values of shorted gap, middel gap or longest
        gap between two hyperboxes.
    asimil_type : {'max', 'min'}, optional, default='max'
        Type of similarity measures is used in the case of `simil_measure`
        getting a value of `mid`. It can be the maximum or minimum values of
        two dissimilar values of a similarity measure based on middle distance.
    is_draw : boolean, optional, default=False
        Whether the construction of hyperboxes can be progressively shown
        during the training process on a canvas window.

    Attributes
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
    is_exist_missing_value : boolean
        Is there any missing values in continuous features in the training data.
    elapsed_training_time : float
        Training time in seconds.

    Notes
    -----
    Note that this implementation uses the accelerated mechanism presented in
    [2]_ to accelerate the improved online learning algorithm.

    References
    ----------
    .. [1] B. Gabrys, "Agglomerative learning algorithms for general fuzzy
    min-max neural network", Journal of VLSI signal processing systems for
    signal, image and video technology, vol. 32, no. 1, pp. 67-82, 2002.
    .. [2] T.T. Khuat and B. Gabrys, "Accelerated learning algorithms of general
    fuzzy min-max neural network using a novel hyperbox selection rule,"
    Information Sciences, vol. 547, pp. 887-909, 2021.

    Examples
    --------
    >>> from hbbrain.numerical_data.batch_learner.accel_agglo_gfmm import AccelAgglomerativeLearningGFMM
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = AccelAgglomerativeLearningGFMM(theta=0.1, min_simil=0.8, simil_measure='short')
    >>> clf.fit(X, y)
    >>> print("Number of hyperboxes = %d"%clf.get_n_hyperboxes())
    Number of hyperboxes = 69
    >>> clf.predict(X[[10, 50, 100]])
    array([0, 1, 2])
    """

    def __init__(self, theta=0.5, gamma=1, min_simil=0.5, simil_measure='mid', asimil_type='max', is_draw=False):
        BaseGFMMClassifier.__init__(self, theta, gamma, is_draw)

        self.min_simil = min_simil
        self.simil_measure = simil_measure

        if simil_measure == 'mid':
            self.asimil_type = asimil_type
        else:
            self.asimil_type = 'max'

        self.N_samples=np.array([])

    def _init_data(self):
        """
        Initialise default values for parameters.

        Returns
        -------
        None.

        """
        self._init_hyperboxes()
        if self.N_samples is None:
            self.N_samples=np.array([])

    def fit(self, X, y):
        """
        Fit the model according to the given training data using the accelerated
        agglomerative learning algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to `X`.

        Returns
        -------
        self : object
            Fitted hyperbox-based model.

        """
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

    def _fit(self, Xl, Xu, y, pre_incl_sample=None):
        """
        Fit the model according to the given training data using the accelerated
        agglomerative learning algorithm. The input data are provided in the
        form of hyperboxes.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            Lower bounds of training features.
        Xu : array-like of shape (n_samples, n_features)
            Upper bounds of training features.
        y : array-like of shape (n_samples,)
            Target vector relative to input hyperboxes.
        pre_incl_sample : array-like of shape (n_samples,), optional, default=None
            Number of samples is included in each input hyperboxes.

        Returns
        -------
        self : object
            Fitted hyperbox-based model.

        """
        self._init_data()

        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            self.is_exist_missing_value = True
            Xl, Xu, y = convert_format_missing_input_zero_one(Xl, Xu, y)
        else:
            self.is_exist_missing_value = False
            
        self.V = Xl.copy()
        self.W = Xu.copy()
        self.C = y.copy()
        
        n_samples, n_features = Xl.shape
		
        if pre_incl_sample is not None:
            self.N_samples = pre_incl_sample.copy()
        else:
            self.N_samples = np.ones(n_samples)
            
        class_ids = np.unique(y)  # list of class labels of input patterns
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

            drawing_canvas = self.initialise_canvas_graph(
                n_features, "GFMM - Accelerated Agglomerative Learning Algorithm")
            n_existed_hyperboxes = len(self.C)

            if n_existed_hyperboxes > 0:
                # draw existing hyperboxes
                color_ = np.array(['k'] * n_existed_hyperboxes, dtype=object)
                for c in range(n_existed_hyperboxes):
                    color_[c] = colors[self.C[c]]
                hyperboxes = draw_box(drawing_canvas, self.V[:, 0:np.minimum(
                    n_features, 3)], self.W[:, 0:np.minimum(n_features, 3)], color_)
                self.delay()
                list_drawn_hyperboxes = list(hyperboxes)
        
        threshold = max(self.min_simil, 1 - np.max(self.gamma) * self.theta)
        is_training = True
        while is_training:
            is_training = False
            
            k = 0 # input pattern index
            while k < len(self.C):
                if self.is_exist_missing_value == False:
                    if self.simil_measure == 'short':
                        mem_vals = membership_func_gfmm(self.W[k], self.V[k], self.V, self.W, self.gamma)
                    elif self.simil_measure == 'long':
                        mem_vals = membership_func_gfmm(self.V[k], self.W[k], self.W, self.V, self.gamma)
                    else:
                        mem_vals = asym_similarity_val_one_many_hyperboxes(self.V[k], self.W[k], self.V, self.W, self.gamma, self.asimil_type)
                else:
                    if self.simil_measure == 'short':
                        mem_vals = membership_func_gfmm(self.W[k], self.V[k], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.gamma)
                    elif self.simil_measure == 'long':
                        mem_vals = membership_func_gfmm(self.V[k], self.W[k], np.maximum(self.W, self.V), np.minimum(self.V, self.W), self.gamma)
                    else:
                        mem_vals = asym_similarity_val_one_many_hyperboxes(self.V[k], self.W[k], np.minimum(self.V, self.W), np.maximum(self.W, self.V), self.gamma, self.asimil_type)
                    
                id_sorted_mem_vals = np.argsort(mem_vals)[::-1]
                sorted_mem_vals = mem_vals[id_sorted_mem_vals]
                
                selected_mem_vals = sorted_mem_vals[sorted_mem_vals >= threshold]	# apply membership threshold
                
                if len(selected_mem_vals) > 0:
                    id_selected_mem_vals = id_sorted_mem_vals[sorted_mem_vals >= threshold]
                    # remove self-membership
                    selected_mem_vals = selected_mem_vals[id_selected_mem_vals != k]
                    id_selected_mem_vals = id_selected_mem_vals[id_selected_mem_vals != k]
                    
                    # remove memberships to hyperboxes representing other classes
                    idx_same_classes = (self.C[id_selected_mem_vals] == self.C[k]) | (self.C[id_selected_mem_vals] == UNLABELED_CLASS) | ((self.C[id_selected_mem_vals] != self.C[k]) & (self.C[k] == UNLABELED_CLASS))
                    selected_mem_vals = selected_mem_vals[idx_same_classes]
                    # leaving memeberships to unlabelled boxes
                    id_selected_mem_vals = id_selected_mem_vals[idx_same_classes]
                                
                    pairewise_k_selected_mem_vals = np.concatenate((np.minimum(k, id_selected_mem_vals)[:, np.newaxis], np.maximum(k,id_selected_mem_vals)[:, np.newaxis], selected_mem_vals[:, np.newaxis]), axis=1)

                    for i in range(pairewise_k_selected_mem_vals.shape[0]):
                        # calculate new coordinates of k-th hyperbox by including pairewise_k_selected_mem_vals(i,1)-th box, scrap the latter and leave the rest intact
                        # agglomorate pairewise_k_selected_mem_vals(i, 0) and pairewise_k_selected_mem_vals(i, 1) by adjusting pairewise_k_selected_mem_vals(i, 0)
                        # remove pairewise_k_selected_mem_vals(i, 1) by getting newV from 1 -> pairewise_k_selected_mem_vals(i, 0) - 1, new coordinates for pairewise_k_selected_mem_vals(i, 0), from pairewise_k_selected_mem_vals(i, 0) + 1 -> pairewise_k_selected_mem_vals(i, 1) - 1, pairewise_k_selected_mem_vals(i, 1) + 1 -> end    
                        row1 = int(pairewise_k_selected_mem_vals[i, 0])
                        row2 = int(pairewise_k_selected_mem_vals[i, 1])
                        newV = np.concatenate((self.V[:row1], np.minimum(self.V[row1], self.V[row2]).reshape(1, -1), self.V[row1 + 1:row2], self.V[row2 + 1:]), axis=0)
                        newW = np.concatenate((self.W[:row1], np.maximum(self.W[row1], self.W[row2]).reshape(1, -1), self.W[row1 + 1:row2], self.W[row2 + 1:]), axis=0)
                        newC = np.concatenate((self.C[:row2], self.C[row2 + 1:]))
                        if (newC[row1] == UNLABELED_CLASS):
                            newC[row1] = self.C[row2]

                        # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                        # position of adjustment is pairewise_k_selected_mem_vals[i, 0] in new bounds
                        if ((((newW[int(pairewise_k_selected_mem_vals[i, 0])] - newV[int(pairewise_k_selected_mem_vals[i, 0])]) <= self.theta).all() == True) and (not is_overlap_one_many_hyperboxes_num_data_general(newV, newW, newC, int(pairewise_k_selected_mem_vals[i, 0])))):
                            self.V = newV
                            self.W = newW
                            self.C = newC
                            
                            self.N_samples[int(pairewise_k_selected_mem_vals[i, 0])] = self.N_samples[int(pairewise_k_selected_mem_vals[i, 0])] + self.N_samples[int(pairewise_k_selected_mem_vals[i, 1])]
                            self.N_samples = np.append(self.N_samples[0:int(pairewise_k_selected_mem_vals[i, 1])], self.N_samples[int(pairewise_k_selected_mem_vals[i, 1]) + 1:])
                       
                            is_training = True
                            
                            if k != pairewise_k_selected_mem_vals[i, 0]: # position pairewise_k_selected_mem_vals[i, 1] (also k) is removed, so next step should start from pairewise_k_selected_mem_vals[i, 1]
                                k = k - 1
                                
                            if self.is_draw:
                                try:
                                    list_drawn_hyperboxes[int(pairewise_k_selected_mem_vals[i, 1])].remove()
                                    list_drawn_hyperboxes[int(pairewise_k_selected_mem_vals[i, 0])].remove()
                                except:
                                    print("No remove old hyperbox")
                                
                                color_ = colors[self.C[int(pairewise_k_selected_mem_vals[i, 0])]]
                                    
                                hyperbox = draw_box(drawing_canvas, np.asmatrix(self.V[int(pairewise_k_selected_mem_vals[i, 0]), 0:np.minimum(
                                    n_features, 3)]), np.asmatrix(self.W[int(pairewise_k_selected_mem_vals[i, 0]), 0:np.minimum(n_features, 3)]), color_)
                                self.delay()
                                list_drawn_hyperboxes[int(pairewise_k_selected_mem_vals[i, 0])] = hyperbox[0]
                                list_drawn_hyperboxes.remove(list_drawn_hyperboxes[int(pairewise_k_selected_mem_vals[i, 1])])
                                
                            break # if hyperbox adjusted there's no need to look at other hyperboxes
                                                   
                k = k + 1
                
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
        return self

    def predict(self, X, type_boundary_handling=PROBABILITY_MEASURE):
        """
        Predict class labels for samples in `X`.

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

        Notes
        -----
        In the case there are many winner hyperboxes representing different class labels
        but with the same membership value with respect to the input pattern :math:`X_i`, an 
        additional criterion based on the probability generated by number of samples included in 
        winner hyperboxes and membership values or the Manhattan distance between the central point 
        of winner hyperboxes and the input sample is used to find the final winner hyperbox that 
        its class label is used for predicting the class label of the input pattern :math:`X_i`.

        """
        X = np.array(X)
        y_pred = self._predict(X, X, type_boundary_handling)
        
        return y_pred

    def _predict(self, Xl, Xu, type_boundary_handling=PROBABILITY_MEASURE):
        """
        Predict class labels for samples in the form of hyperboxes represented 
        by low bounds `Xl` and upper bounds `Xu`.

        Notes
        -----
        In the case there are many winner hyperboxes representing different class labels
        but with the same membership value with respect to the input pattern :math:`X_i` in 
        the form of an hyperbox represented by a lower bound :math:`Xl_i` and an upper 
        bound :math:`Xu_i`, an additional criterion based on the probability generated by number of samples included in 
        winner hyperboxes and membership values or the Manhattan distance between the central point 
        of winner hyperboxes and the input sample is used to find the final winner hyperbox that 
        its class label is used for predicting the class label of the input hyperbox :math:`X_i`.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The data matrix containing the lower bounds of input patterns
            for which we want to predict the targets.
        Xu : array-like of shape (n_samples, n_features)
            The data matrix containing the upper bounds of input patterns 
            for which we want to predict the targets.
        type_boundary_handling : int, optional, default=PROBABILITY_MEASURE (aka 1)
            The way of handling many winner hyperboxes, i.e., PROBABILITY_MEASURE or MANHATTAN_DIS

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        if type_boundary_handling == PROBABILITY_MEASURE:
            y_pred = predict_with_probability(self.V, self.W, self.C, self.N_samples, Xl, Xu, self.gamma)
        else:
            y_pred = predict_with_manhattan(self.V, self.W, self.C, Xl, Xu, self.gamma)
        
        return y_pred

    def get_sample_explanation(self, xl, xu, type_boundary_handling=PROBABILITY_MEASURE):
        """
        Get useful information for explaining the reason behind the predicted result for the input pattern

        Parameters
        ----------
        xl : ndarray of shape (n_feature,)
            Minimum point of the input pattern which needs to be explained.
        xu : ndarray of shape (n_feature,)
            Maximum point of the input pattern which needs to be explained.
        type_boundary_handling : int, optional, default=PROBABILITY_MEASURE (aka 1)
            The way of handling samples located on the boundary.

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
        y_pred = self._predict(xl, xu, type_boundary_handling)[0]
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

    def simple_pruning(self, Xl_val, Xu_val, y_val, acc_threshold=0.5, keep_empty_boxes=False, type_boundary_handling=PROBABILITY_MEASURE):
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
        type_boundary_handling : int, optional, default=PROBABILITY_MEASURE (aka 1)
            The way of handling samples located on the boundary.

        Returns
        -------
        self
            A hyperbox-based model with the low-qualitied hyperboxes pruned.

        """
        n_samples = Xl_val.shape[0]
        rnd = np.random
        rnd.seed(0)
        random.seed(0)
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
                        sum_prod_denum = (mem_val[max_mem_V_id] * self.N_samples[max_mem_V_id]).sum()
                        max_prob = -1
                        pre_id_cls = None
                        for c in cls_same_mem:
                            id_cls = np.nonzero(self.C[max_mem_V_id] == c)[0]
                            sum_pro_num = (mem_val[max_mem_V_id[id_cls]] * self.N_samples[max_mem_V_id[id_cls]]).sum()
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
            C_pruned_excl_empty_boxes = self.C[id_remained_excl_empty_boxes]
            N_samples_excl_empty_boxes = self.N_samples[id_remained_excl_empty_boxes]
            
            W_pruned_incl_empty_boxes = self.W[id_remained_incl_empty_boxes]
            V_pruned_incl_empty_boxes = self.V[id_remained_incl_empty_boxes]
            C_pruned_incl_empty_boxes = self.C[id_remained_incl_empty_boxes]
            N_samples_incl_empty_boxes = self.N_samples[id_remained_incl_empty_boxes]
            
            if type_boundary_handling == PROBABILITY_MEASURE:
                y_val_pred_excl_empty_boxes = predict_with_probability(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, N_samples_excl_empty_boxes, Xl_val, Xu_val, self.gamma)
                y_val_pred_incl_empty_boxes = predict_with_probability(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, N_samples_incl_empty_boxes, Xl_val, Xu_val, self.gamma)
            else:
                y_val_pred_excl_empty_boxes = predict_with_manhattan(V_pruned_excl_empty_boxes, W_pruned_excl_empty_boxes, C_pruned_excl_empty_boxes, Xl_val, Xu_val, self.gamma)
                y_val_pred_incl_empty_boxes = predict_with_manhattan(V_pruned_incl_empty_boxes, W_pruned_incl_empty_boxes, C_pruned_incl_empty_boxes, Xl_val, Xu_val, self.gamma)
            
            if (accuracy_score(y_val, y_val_pred_excl_empty_boxes) >= accuracy_score(y_val, y_val_pred_incl_empty_boxes)):
                self.V = V_pruned_excl_empty_boxes
                self.W = W_pruned_excl_empty_boxes
                self.C = C_pruned_excl_empty_boxes
                self.N_samples = N_samples_excl_empty_boxes
            else:
                self.V = V_pruned_incl_empty_boxes
                self.W = W_pruned_incl_empty_boxes
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

    simi_measure_choice = ['mid', 'long','short']
    asimil_type_choice = ['min', 'max']

    def check_simi_measure(s):
        options = [c for c in simi_measure_choice if s in c]
        if len(options) == 1:
            return options[0]
        else:
            return 'mid'

    def check_asimil_type(s):
        options = [c for c in asimil_type_choice if s in c]
        if len(options) == 1:
            return options[0]
        else:
            return 'max'

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
    optional.add_argument('--min_simil', type=float, default=0.5,
                          help='Mimimum similarity value so that two hyperboxes can be merged (in the range of [0, 1])(default: 0.5)')
    optional.add_argument('--simil_measure', choices=simi_measure_choice, type=check_simi_measure, default='mid',
                          help='Type of similarity measure (default: mid)')   
    optional.add_argument('--asimil_type', choices=asimil_type_choice, type=check_asimil_type, default='max',
                          help='Type of handling asymmetric similarity matrix (default: max)')
    optional.add_argument('--is_draw', type=str2bool, default=False,
                          help='Show the existing hyperboxes during the training process on the screen (default: False)')

    args = parser.parse_args()

    if args.theta <= 0 or args.theta > 1:
        parser.error("--theta has to be in the range of (0, 1]")

    if args.min_simil < 0 or args.min_simil > 1:
        parser.error("--min_simil has to be in the range of [0, 1]")

    if args.gamma <= 0:
        parser.error("--gamma has to be larger than 0")

    gamma = args.gamma
    theta = args.theta
    min_simil = args.min_simil
    simil_measure = args.simil_measure
    asimil_type = args.asimil_type
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

    accel_agglo_gfmm = AccelAgglomerativeLearningGFMM(theta, gamma, min_simil, simil_measure, asimil_type, is_draw)
    accel_agglo_gfmm.fit(Xtr, ytr)
    print('Number of hyperboxes = %d'%accel_agglo_gfmm.get_n_hyperboxes())

    y_pred = accel_agglo_gfmm.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy (using a probability measure for samples on the boundary) = {acc * 100: .2f}%')

    y_pred_2 = accel_agglo_gfmm.predict(Xtest, MANHATTAN_DIS)
    acc_2 = accuracy_score(ytest, y_pred_2)
    print(f'Testing accuracy (using a Manhattan distance for samples on the boundary) = {acc_2 * 100: .2f}%')

    # sample_need_explain = 10
    # y_pred_input_0, mem_val_classes, min_points_classes, max_points_classes = accel_agglo_gfmm.get_sample_explanation(Xtest[sample_need_explain], Xtest[sample_need_explain])
    # accel_agglo_gfmm.show_sample_explanation(Xtest[sample_need_explain], Xtest[sample_need_explain], min_points_classes, max_points_classes, y_pred_input_0, "2D")

    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/syn_num_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]

    # accel_agglo_gfmm.simple_pruning(X_val, X_val, y_val, 0.5, False, PROBABILITY_MEASURE)
    # print('Number of hyperboxes after pruning = %d'%accel_agglo_gfmm.get_n_hyperboxes())
    # accel_agglo_gfmm.draw_hyperbox_and_boundary()

    # y_pred_2 = accel_agglo_gfmm.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy (using a probability measure for samples on the boundary) = {acc_pruned * 100: .2f}%')          
