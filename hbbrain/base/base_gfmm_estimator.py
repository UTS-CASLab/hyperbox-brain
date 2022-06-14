"""
Base class and functions for all general fuzzy min-max neural network estimators.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import random
from hbbrain.base.base_estimator import BaseHyperboxClassifier
from hbbrain.utils.membership_calc import membership_func_gfmm, get_membership_gfmm_all_classes
from hbbrain.utils.dist_metrics import manhattan_distance, manhattan_distance_with_missing_val
from hbbrain.constants import UNLABELED_CLASS, EPSILON_MISSING_VAL


def predict_with_manhattan(V, W, C, Xl, Xu, g=1):
    """
    Predict class labels for samples in `X` represented in the form of invervals `[Xl, Xu]`.
    This is a common function to determine the right class labels for X wrt. a trained hyperbox-based 
    classifier represented by `[V, W, C]`. It uses the winner-takes-all principle to predict 
    class labels for each sample in X by assigning the class label of the sample to the class 
    label of the hyperbox with the maximum membership value to that sample. It will use 
    a Manhattan distance in the case of many hyperboxes with different classes having the 
    same maximum membership value.

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
    Xl : array-like of shape (n_samples, n_features)
        The data matrix contains lower bounds of input patterns for which we want to predict the targets.
    Xu : array-like of shape (n_samples, n_features)
        The data matrix contains upper bounds of input patterns for which we want to predict the targets.
    g : float or array-like of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the membership function in each dimension.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        A vector contains the predictions. In binary and multiclass problems, this 
        is a vector containing `n_samples`. 

    """
    if Xl.ndim == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)
        
    if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
        Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)
        
    is_exist_missing_value = (V > W).any()
    
    #initialization
    yX = Xl.shape[0]
    y_pred = np.full(yX, 0)
    # classifications
    sample_id = 0
    for i in range(yX):
        sample_id += 1
        
        if is_exist_missing_value == False:
            mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], V, W, g) # calculate memberships for all hyperboxes
        else:
            mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], np.minimum(V, W), np.maximum(W, V), g) # calculate memberships for all hyperboxes
            
        bmax = mem_val.max() # get the maximum membership value
        
        if ((Xl[i] < 0).any() == True) or ((Xu[i] > 1).any() == True):
            print(">>> The testing sample %d with the coordinate %s is outside the range [0, 1]. Membership value = %f. The prediction is more likely incorrect." % (sample_id, Xl[i], bmax))
            
        max_mem_V_id = np.nonzero(mem_val == bmax)[0] # get indices of all hyperboxes with the maximum membership values
        
        if len(np.unique(C[max_mem_V_id])) > 1:
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


def predict_with_probability(V, W, C, N_samples, Xl, Xu, g=1):
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
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points of all hyperboxes of a trained hyperbox-based model, 
        in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points of all hyperboxes of a trained hyperbox-based model, 
        in which each row is a maximal point of a hyperbox.
    C : ndarray of shape (n_hyperboxes,)
        An array contains all class lables for all hyperboxes of a trained hyperbox-based model.
    N_samples : ndarray of shape (n_hyperboxes,)
        An array contains number of samples included in each hyperbox of a trained hyperbox-based model.
    Xl : array-like of shape (n_samples, n_features)
        The data matrix contains lower bounds of input patterns for which we want to predict the targets.
    Xu : array-like of shape (n_samples, n_features)
        The data matrix contains upper bounds of input patterns for which we want to predict the targets.
    g : float or array-like of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the membership function in each dimension.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        A vector contains the predictions. In binary and multiclass problems, this 
        is a vector containing `n_samples`.
        
    """
    if Xl.ndim == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)
        
    if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
        Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)
        
    is_exist_missing_value = (V > W).any()
        
    #initialization
    n_samples = Xl.shape[0]
    y_pred = np.full(n_samples, 0)
    sample_id = 0
    # classifications
    for i in range(n_samples):
        sample_id += 1
        
        if is_exist_missing_value == False:
            mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], V, W, g) # calculate memberships for all hyperboxes
        else:
            mem_val = membership_func_gfmm(Xl[i, :], Xu[i, :], np.minimum(V, W), np.maximum(W, V), g) # calculate memberships for all hyperboxes
            
        bmax = mem_val.max() # get the maximum membership value
        
        if ((Xl[i] < 0).any() == True) or ((Xu[i] > 1).any() == True):
            print(">>> The testing sample %d with the coordinate %s is outside the range [0, 1]. Membership value = %f. The prediction is more likely incorrect." % (sample_id, Xl[i], bmax))
            
        max_mem_V_id = np.nonzero(mem_val == bmax)[0] # get indices of all hyperboxes with the maximum membership values
        
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


def convert_format_missing_input_zero_one(Xl, Xu, y=None):
    """
    Convert missing values in the features and labels under the form of NaN values
    to the form used in the algorithms

    Parameters
    ----------
    Xl : array-like of shape (n_samples, n_features)
        A matrix containing lower bound values of features and samples, where `n_samples` 
        is the number of samples and `n_features` is the number of features.
    Xu : array-like of shape (n_samples, n_features)
        A matrix containing upper bound values of features and samples, where `n_samples` 
        is the number of samples and `n_features` is the number of features.
    y : array-like of shape (n_samples,)
        Target vector relative to [Xl, Xu].

    Returns
    -------
    Xl_out : array-like of shape (n_samples, n_features)
        The transformed matrix of the input matrix Xl.
    Xu_out : array-like of shape (n_samples, n_features)
        The transformed matrix of the input matrix Xu.
    y_out : array-like of shape (n_samples, n_features)
        The transformed vector of the input vector y.

    """
    Xl_out = np.where(np.isnan(Xl), 1 + EPSILON_MISSING_VAL, Xl)
    Xu_out = np.where(np.isnan(Xu), -EPSILON_MISSING_VAL, Xu)
    if y is not None:
        y_out = np.where(np.isnan(y), UNLABELED_CLASS, y)
    else:
        y_out = None

    return (Xl_out, Xu_out, y_out)


def is_contain_missing_value(X):
    """
    Check whether an input vector `X` contains any missing values.

    Parameters
    ----------
    X : array-like of shape (n_features,) or (n_samples, n_features)
        A input vector for which we want to check the existence of missing values. 

    Returns
    -------
    bool
        The output value showing whether the input vector `X` contains missing
        values or not.

    """
    if np.isnan(X).sum() > 0:
        return True
    else:
        return False


class BaseGFMMClassifier(BaseHyperboxClassifier):
    """
    Base class for all hyperbox-based estimators in the hyperbox-brain.

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
        BaseHyperboxClassifier.__init__(self, theta=theta, is_draw=is_draw, V=V, W=W, C=C)
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
        y_pred = self._predict(X, X)

        return y_pred

    def _predict(self, Xl, Xu):
        """
        Predict class labels for samples in the form of hyperboxes represented 
        by low bounds `Xl` and upper bounds `Xu`.
        
        .. note::

            In the case there are many winner hyperboxes representing different
            class labels but with the same membership value with respect to the
            input pattern :math:`X_i` in the form of an hyperbox represented by
            a lower bound :math:`Xl_i` and an upper bound :math:`Xu_i`, an
            additional criterion based on the minimum Manhattan distance
            between the central point of input hyperbox :math:`X_i = [Xl_i, Xu_i]`
            and the central points of winner hyperboxes are used to find the
            final winner hyperbox that its class label is used for predicting
            the class label of the input hyperbox :math:`X_i`.
        
        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The data matrix containing the lower bounds of input patterns
            for which we want to predict the targets.
        Xu : array-like of shape (n_samples, n_features)
            The data matrix containing the upper bounds of input patterns 
            for which we want to predict the targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`.

        """
        y_pred = predict_with_manhattan(self.V, self.W, self.C, Xl, Xu, self.gamma)

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
    
    def _predict_proba(self, Xl, Xu):
        """
        Predict class probabilities of the input hyperboxes represented by
        lower bounds Xl and upper bounds Xu.
        
        The predicted class probability is the fraction of the membership value
        of the representative hyperbox of that class and the sum of all
        membership values of all representative hyperboxes of all classes.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The lower bounds of input hyperboxes.
        Xu : array-like of shape (n_samples, n_features)
            The upper bounds of input hyperboxes.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        mem_vals = self._predict_with_membership(Xl, Xu)
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
        mem_vals = self._predict_with_membership(X, X)
        
        return mem_vals
    
    def _predict_with_membership(self, Xl, Xu):
        """
        Predict class membership values of the input hyperboxes represented by
        lower bounds Xl and upper bounds Xu.
        
        The predicted class membership value is the membership value
        of the representative hyperbox of that class.

        Parameters
        ----------
        Xl : array-like of shape (n_samples, n_features)
            The lower bounds of input hyperboxes.
        Xu : array-like of shape (n_samples, n_features)
            The upper bounds of input hyperboxes.

        Returns
        -------
        mem_vals : ndarray of shape (n_samples, n_classes)
            The class membership values of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        if (is_contain_missing_value(Xl) == True) or (is_contain_missing_value(Xu) == True):
            Xl, Xu, _ = convert_format_missing_input_zero_one(Xl, Xu)
        
        mem_vals, _ = get_membership_gfmm_all_classes(Xl, Xu, self.V, self.W, self.C, self.gamma)
        
        return mem_vals