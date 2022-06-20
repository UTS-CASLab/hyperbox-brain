"""
The :mod:`hbbrain.utils.adjust_hyperbox` submodule implements various functions for 
hyperbox adjustment, e.g., hyperbox overlap test, overlap resolving, and hyperbox contraction.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
from copy import deepcopy
from hbbrain.utils.membership_calc import membership_func_gfmm
from hbbrain.utils.matrix_transformation import hashing
from hbbrain.constants import UNLABELED_CLASS, DEFAULT_CATEGORICAL_VALUE


def is_overlap_one_many_hyperboxes_num_data_general(V, W, C, id_box):
    """
    Check overlap between the hyperbox at the position `id_box` and remaning hyperboxes in the current list.

    .. note::

        The current input list of hyperboxes contains all existing hyperboxes
        including the hyperboxes representing the same class as the hyperbox at
        the position `id_box`. Therefore, to perform overlap testing, the list
        of hyperboxes representing the class labels other than the class label
        of the `id_box`-th hyperbox should be first filtered. Finally, the
        overlap test is only conducted on this filtered list.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (minimum points) of all existing hyperboxes in the trained model.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes.
    id_box : int
        id_extended_boxex of the hyperbox to be checked for overlap.

    Returns
    -------
    bool
        Show if the input hyperbox overlaps with any hyperbox in the list of
        hyperboxes representing the classes other than its class label.

    """
    if (V[id_box] > W[id_box]).any() == True:
        return False
    else:
        id_considered_hyperboxes = np.nonzero((W >= V).all(axis = 1))[0] 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(id_considered_hyperboxes) == 0:
            return False
        else:
            class_considered_hyperboxes = C[id_considered_hyperboxes]
            id_diff_label_hyperboxes = id_considered_hyperboxes[(class_considered_hyperboxes != C[id_box]) | (class_considered_hyperboxes == UNLABELED_CLASS)] # get index of hyperbox representing different classes
            if C[id_box] == UNLABELED_CLASS:
                id_exclude_itself = np.nonzero(id_diff_label_hyperboxes == id_box)[0]
                id_diff_label_hyperboxes = np.delete(id_diff_label_hyperboxes, id_exclude_itself)

            if len(id_diff_label_hyperboxes) > 0:
                b = membership_func_gfmm(W[id_box], V[id_box], V[id_diff_label_hyperboxes], W[id_diff_label_hyperboxes], 1)
                id_overlaped_hyperboxes = np.nonzero(b == 1)[0] # id of overlapping hyperboxes
                if len(id_overlaped_hyperboxes) == 0:
                    # No overlap
                    return False
                else:
                    return True
            else:
                return False


def is_overlap_one_many_diff_label_hyperboxes_num_data_general(V, W, V_cmp, W_cmp):
    """
    Check whether an input hyperbox overlaps with any hyperboxes representing different classes with the input hyperbox

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of hyperboxes representing other classes compared to V_cmp.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of hyperboxes representing other classes compared to W_cmp.
    V_cmp : array-like of shape (n_features,)
        Minimum point of the compared hyperbox.
    W_cmp : array-like of shape (n_features,)
        Maximum point of the compared hyperbox.

    Returns
    -------
    bool
        Show if the input hyperbox overlaps with any hyperbox in the list of hyperboxes representing the classes other than the input hyperbox.

    """
    if (V_cmp > W_cmp).any() == True:
        return False
    else:
        id_considered_hyperboxes = np.nonzero((W >= V).all(axis = 1))[0] 	# examine only hyperboxes w/o missing dimensions, meaning that in each dimension upper bound is larger than lowerbound
        
        if len(id_considered_hyperboxes) == 0:
            return False
        else:
            b = membership_func_gfmm(W_cmp, V_cmp, V[id_considered_hyperboxes], W[id_considered_hyperboxes], 1)
            id_overlaped_hyperboxes = np.nonzero(b == 1)[0] # id of overlapping hyperboxes
            if len(id_overlaped_hyperboxes) == 0:
                # No overlap
                return False
            else:
                return True


def is_two_hyperboxes_overlap_num_data_general(Vi, Wi, Vk, Wk):
    """Check if two hyperboxes `Bi` and `Bk` overlap with each other or not.
    This function uses a general formula built from a shortest gap-based similarity
    measure. If this measure returns a value of 1, it means that these two hyperboxes 
    overlap with each other. Otherwise, two hyperboxes `Bi` and `Bk` do not overlap.
    See the references [1]_ and [2]_ for more details.

    Parameters
    ----------
    Vi : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bi`.
    Wi : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bi`.
    Vk : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bk`.
    Wk : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bk`.

    Returns
    -------
    is_overlap : boolean
        Show if two hyperboxes `Bi` and `Bk` overlap or not.
        
    References
    ----------
    For more details regarding the way of checking the overlap between two hyperboxes, 
    please see the following articles:

    .. [1] B. Gabrys, "Agglomerative learning algorithms for general fuzzy min-max 
           neural network," Journal of VLSI signal processing systems for signal, 
           image and video technology, vol. 32, no. 1, pp. 67-82, 2002.
    .. [2] T.T. Khuat and B. Gabrys, "An Online Learning Algorithm for a Neuro-Fuzzy 
           Classifier with Mixed-Attribute Data," ArXiv Preprint, no. arXiv:2009.14670, 2020.

    """
    is_overlap = False
    if ((Vi > Wi).any() == True) or ((Vk > Wk).any() == True):
        # Check if any hyperbox contains missing dimensions. If so, return no overlap
        return is_overlap
    else:
        b = membership_func_gfmm(Wi, Vi, Vk.reshape(1, -1), Wk.reshape(1, -1), 1)
        if b[0] == 1:
            is_overlap = True
        else:
            is_overlap = False
        
    return is_overlap


def overlap_resolving_num_data(Vi, Wi, ci, Vk, Wk, ck, alpha=0.00001):
    """Resolve overlap between two hyperboxes `Bi` and `Bk` with coordinates being 
    numerical features. For more details regarding the way of contracting two
    overlapping hyperboxes, please see the article [1]_:

    Parameters
    ----------
    Vi : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bi`.
    Wi : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bi`.
    ci : int
        Class label of the hyperbox `Bi`.
    Vk : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bk`.
    Wk : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bk`.
    ck : int
        Class label of the hyperbox `Bk`.
    alpha : float, optional, default=0.00001
        A very small value is used to avoid the overlap between two hyperboxes
        after contraction.

    Returns
    -------
    Vi : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bi` after contraction.
    Wi : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bi` after contraction.
    Vk : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bk` after contraction.
    Wk : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bk` after contraction.
        
    References
    ----------
    .. [1] T.T. Khuat and B. Gabrys, "Accelerated learning algorithms of general 
           fuzzy min-max neural network using a novel hyperbox selection rule,"
           Information Sciences, vol. 547, pp. 887-909, 2021.
           
    """
    if (ci == ck) and (ci != UNLABELED_CLASS):
        return (Vi, Wi, Vk, Wk)

    delta_j = 2
    case = -1
    min_j = -1
    for j in range(len(Vi)):
        if (Vi[j] <= Vk[j]) and (Vk[j] < Wi[j]) and (Wi[j] <= Wk[j]):
            if delta_j > Wi[j] - Vk[j]:
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 1
        elif (Vk[j] <= Vi[j]) and (Vi[j] < Wk[j]) and (Wk[j] <= Wi[j]):
            if delta_j > Wk[j] - Vi[j]:
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 2
        elif (Vi[j] < Vk[j]) and (Wk[j] < Wi[j]):
            if (delta_j > (Wk[j] - Vi[j])) and ((Wk[j] - Vi[j]) <= (Wi[j] - Vk[j])):
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 31
            elif delta_j > (Wi[j] - Vk[j]):
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 32
        elif (Vk[j] < Vi[j]) and (Wi[j] < Wk[j]):
            if (delta_j > (Wk[j] - Vi[j])) and ((Wk[j] - Vi[j]) <= (Wi[j] - Vk[j])):
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 41
            elif delta_j > (Wi[j] - Vk[j]):
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 42
        # Here we add four more overlapping cases between two hyperboxes
        elif (Vi[j] <= Wi[j]) and (Wi[j] == Vk[j]) and (Vk[j] < Wk[j]):
            # S1: Vij <= Wij = Vkj < Wkj
            delta_j = 0
            min_j = j
            case = 5
            break
        elif (Vi[j] < Wi[j]) and (Wi[j] == Vk[j]) and (Vk[j] <= Wk[j]):
            # S2: Vij < Wij = Vkj <= Wkj
            delta_j = 0
            min_j = j
            case = 6
            break
        elif (Vk[j] <= Wk[j]) and (Wk[j] == Vi[j]) and (Vi[j] < Wi[j]):
            # S3: Vkj <= Wkj = Vij < Wij
            delta_j = 0
            min_j = j
            case = 7
            break
        elif (Vk[j] < Wk[j]) and (Wk[j] == Vi[j]) and (Vi[j] <= Wi[j]):
            # S4: Vkj < Wkj = Vij <= Wij
            delta_j = 0
            min_j = j
            case = 8
            break

    # Adjust the coordinates of two hyperboxes Bi and Bk according to the overlap cases
    if case == 1:
        tmp = (Wi[min_j] + Vk[min_j]) / 2
        Wi[min_j] = tmp
        Vk[min_j] = tmp  + alpha
    elif case == 2:
        tmp = (Wk[min_j] + Vi[min_j]) / 2
        Vi[min_j] =  tmp + alpha
        Wk[min_j] = tmp
    elif case == 31:
        Vi[min_j] = Wk[min_j] + alpha
    elif case == 32:
        Wi[min_j] = Vk[min_j] - alpha
    elif case == 41:
        Wk[min_j] = Vi[min_j] - alpha
    elif case == 42:
        Vk[min_j] = Wi[min_j] + alpha
    elif case == 5:
        if (Vk[min_j] + alpha) < Wk[min_j]:
            Vk[min_j] = Vk[min_j] + alpha
        else:
            Vk[min_j] = Wk[min_j]
    elif case == 6:
        if (Wi[min_j] - alpha) > Vi[min_j]:
            Wi[min_j] = Wi[min_j] - alpha
        else:
            Wi[min_j] = Vi[min_j]
    elif case == 7:
        if (Vi[min_j] + alpha) < Wi[min_j]:
            Vi[min_j] = Vi[min_j] + alpha
        else:
            Vi[min_j] = Wi[min_j]
    elif case == 8:
        if (Wk[min_j] - alpha) > Vk[min_j]:
            Wk[min_j] = Wk[min_j] - alpha
        else:
            Wk[min_j] = Vk[min_j]

    return (Vi, Wi, Vk, Wk)


def hyperbox_overlap_test_fmnn(V, W, id_extended_box, id_tested_box):
    """
    Check the overlap of two input hyperboxes

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of all existing hyperboxes in the trained model.
    id_extended_box : int
        id_extended_boxex of the extended hyperbox which needs to test for overlap.
    id_tested_box : int
        id_extended_boxex of the hyperbox to test for overlap with the extended hyperbox.

    Returns
    -------
    dim : list with two integer elements
        The first element contains the overlap test case which two hyperboxes overlap with each other.
        The second element contains the corresponding dimension where two hyperboxes overlap with each other.

    """
    dim = np.array([])

    if (V[id_extended_box] > W[id_extended_box]).any() == True:
        # The extended hyperbox contains missing features => No need to test for overlap
        return dim

    if (V[id_tested_box] > W[id_tested_box]).any() == True:
        # The tested hyperbox contains missing features => No need to test for overlap
        return dim

    n_features = W.shape[1]

    condWiWk = W[id_extended_box, :] - W[id_tested_box, :] > 0
    condViVk = V[id_extended_box, :] - V[id_tested_box, :] > 0
    condWkVi = W[id_tested_box, :] - V[id_extended_box, :] > 0
    condWiVk = W[id_extended_box, :] - V[id_tested_box, :] > 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condWiWk & ~condViVk
    c4 = ~condWiWk & condViVk
    c = c1 + c2 + c3 + c4

    ad = c.all()

    if ad == True:
        minimum = 2
        for i in range(n_features):
            if c1[i] == True:
                if minimum > W[id_extended_box, i] - V[id_tested_box, i]:
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([1, i])
            
            elif c2[i] == True:
                if minimum > W[id_tested_box, i] - V[id_extended_box, i]:
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([2, i])
            
            elif c3[i] == True:
                if minimum > (W[id_tested_box, i] - V[id_extended_box,i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([31, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([32, i])
                    
            elif c4[i] == True:
                if minimum > (W[id_tested_box, i] - V[id_extended_box, i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([41, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([42, i])

    return dim


def hyperbox_contraction_fmnn(V, W, case_contraction, id_extended_box, id_tested_box, alpha=0.00001):
    """
    Adjust the coordinates of two hyperboxes for overlap resolving.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of all existing hyperboxes in the trained model.
    case_contraction : a list of two elements
        This is a special struct which is the outcomes of the :func:`hyperbox_overlap_test_fmnn` function to 
        determine the overlap test case and corresponding overlapped dimension.
    id_extended_box : int
        id_extended_boxex of the extended hyperbox which needs to test for overlap.
    id_tested_box : int
        id_extended_boxex of the hyperbox to test for overlap with the extended hyperbox.
    alpha : float, optional, default=0.00001
        A very small value is used to avoid the overlap between two hyperboxes
        after contraction.

    Returns
    -------
    Vout : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes with two hyperboxes adjusted.
    Wout : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of all existing hyperboxes with two hyperboxes adjusted.

    """
    Vout = V.copy()
    Wout = W.copy()
    if case_contraction[0] == 1:
        Wout[id_extended_box, case_contraction[1]] = (Vout[id_tested_box, case_contraction[1]] + Wout[id_extended_box, case_contraction[1]]) / 2
        Vout[id_tested_box, case_contraction[1]] = Wout[id_extended_box, case_contraction[1]] + alpha
    elif case_contraction[0] == 2:
        Vout[id_extended_box, case_contraction[1]] = (Wout[id_tested_box, case_contraction[1]] + Vout[id_extended_box, case_contraction[1]]) / 2
        Wout[id_tested_box, case_contraction[1]] = Vout[id_extended_box, case_contraction[1]] - alpha
    elif case_contraction[0] == 31:
        Vout[id_extended_box, case_contraction[1]] = Wout[id_tested_box, case_contraction[1]] + alpha
    elif case_contraction[0] == 32:
        Wout[id_extended_box, case_contraction[1]] = Vout[id_tested_box, case_contraction[1]] - alpha
    elif case_contraction[0] == 41:
        Wout[id_tested_box, case_contraction[1]] = Vout[id_extended_box, case_contraction[1]] - alpha
    elif case_contraction[0] == 42:
        Vout[id_tested_box, case_contraction[1]] = Wout[id_extended_box, case_contraction[1]] + alpha

    return (Vout, Wout)


def hyperbox_overlap_test_efmnn(V, W, id_extended_box, id_tested_box, X):
    """
    Check the overlap of two input hyperboxes

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of all existing hyperboxes in the trained model.
    id_extended_box : int
        Index of the extended hyperbox which needs to test for overlap.
    id_tested_box : int
        Index of the hyperbox to test for overlap with the extended hyperbox.
    X : array-like of shape (n_features, )
        Current input sample leads to the extension of the existing hyperbox (only be used for contraction case 9)

    Returns
    -------
    dim : list with two integer elements
        The first element contains the overlap test case which two hyperboxes overlap with each other.
        The second element contains the corresponding dimension where two hyperboxes overlap with each other.
    """
    dim = np.array([]);
    n_features = V.shape[1]

    condWiWk = W[id_extended_box, :] - W[id_tested_box, :] > 0
    condViVk = V[id_extended_box, :] - V[id_tested_box, :] > 0
    condWkVi = W[id_tested_box, :] - V[id_extended_box, :] > 0
    condWiVk = W[id_extended_box, :] - V[id_tested_box, :] > 0

    condEqViVk = V[id_extended_box, :] - V[id_tested_box, :] == 0
    condEqWiWk = W[id_extended_box, :] - W[id_tested_box, :] == 0

    c1 = ~condWiWk & ~condViVk & condWiVk
    c2 = condWiWk & condViVk & condWkVi
    c3 = condEqViVk & condWiVk & ~condWiWk
    c4 = ~condViVk & condWiVk & condEqWiWk
    c5 = condEqViVk & condWkVi & condWiWk
    c6 = condViVk & condWkVi & condEqWiWk
    c7 = ~condViVk & condWiWk
    c8 = condViVk & ~condWiWk
    c9 = condEqViVk & ~condViVk & condEqWiWk

    c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9

    ad = c.all()

    if ad == True:
        minimum = 2
        for i in range(n_features):
            if c1[i] == True and c3[i] == False and c4[i] == False:
                if minimum > W[id_extended_box, i] - V[id_tested_box, i]:
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([1, i])
            
            elif c2[i] == True:
                if minimum > W[id_tested_box, i] - V[id_extended_box, i]:
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([2, i])
            
            elif c3[i] == True and c9[i] == False:
                if minimum > (W[id_tested_box, i] - V[id_extended_box,i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([3, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([3, i])

            elif c4[i] == True and c9[i] == False:
                if minimum > (W[id_tested_box, i] - V[id_extended_box, i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([4, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([4, i])
                
            elif c5[i] == True:
                if minimum > (W[id_tested_box, i] - V[id_extended_box,i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([5, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([5, i])

            elif c6[i] == True:
                if minimum > (W[id_tested_box, i] - V[id_extended_box,i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([6, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([6, i])

            elif c7[i] == True:
                if minimum > (W[id_tested_box, i] - V[id_extended_box,i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([71, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([72, i])

            elif c8[i] == True:
                if minimum > (W[id_tested_box, i] - V[id_extended_box,i]) and (W[id_tested_box, i] - V[id_extended_box, i]) < (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box, i]
                    dim = np.array([81, i])
                elif minimum > (W[id_extended_box, i] - V[id_tested_box, i]):
                    minimum = W[id_extended_box, i] - V[id_tested_box, i]
                    dim = np.array([82, i])

            elif c9[i] == True:
                if minimum > (W[id_tested_box, i] - V[id_extended_box,i]):
                    minimum = W[id_tested_box, i] - V[id_extended_box,i]

                    if W[id_extended_box, i] == X[i]: # maximum point of the hyperbox is expanded
                        dim = np.array([91, i])
                    else: # minimum point of the hyperbox is expanded
                        dim = np.array([92, i])

    return dim


def hyperbox_contraction_efmnn(V, W, case_contraction, id_extended_box, id_tested_box, alpha=0.00001):
    """
    Adjust the coordinates of two hyperboxes for overlap resolving corresponding to nine overlap test cases.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of all existing hyperboxes in the trained model.
    case_contraction : a list of two elements
        This is a special struct which is the outcomes of the hyperbox_overlap_test_efmnn function to 
        determine the overlap test case and corresponding overlapped dimension.
    id_extended_box : int
        id_extended_boxex of the extended hyperbox which needs to test for overlap.
    id_tested_box : int
        id_extended_boxex of the hyperbox to test for overlap with the extended hyperbox.
    alpha : float
        A very small value is used to avoid the overlap between two hyperboxes
        after contraction.

    Returns
    -------
    Vout : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes with two hyperboxes adjusted.
    Wout : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of all existing hyperboxes with two hyperboxes adjusted.

    """
    Vout = V.copy()
    Wout = W.copy()
    if case_contraction[0] == 1 or case_contraction[0] == 91:
        Wout[id_extended_box, case_contraction[1]] = (Vout[id_tested_box, case_contraction[1]] + Wout[id_extended_box, case_contraction[1]]) / 2
        Vout[id_tested_box, case_contraction[1]] = Wout[id_extended_box, case_contraction[1]] + alpha
    elif case_contraction[0] == 2 or case_contraction[0] == 92:
        Vout[id_extended_box, case_contraction[1]] = (Wout[id_tested_box, case_contraction[1]] + Vout[id_extended_box, case_contraction[1]]) / 2
        Wout[id_tested_box, case_contraction[1]] = Vout[id_extended_box, case_contraction[1]] - alpha
    elif case_contraction[0] == 3 or case_contraction[0] == 82:
        Vout[id_tested_box, case_contraction[1]] = Wout[id_extended_box, case_contraction[1]] + alpha
    elif case_contraction[0] == 4 or case_contraction[0] == 72:
        Wout[id_extended_box, case_contraction[1]] = Vout[id_tested_box, case_contraction[1]] - alpha
    elif case_contraction[0] == 5 or case_contraction[0] == 71:
        Vout[id_extended_box, case_contraction[1]] = Wout[id_tested_box, case_contraction[1]] + alpha
    elif case_contraction[0] == 6 or case_contraction[0] == 81:
        Wout[id_tested_box, case_contraction[1]] = Vout[id_extended_box, case_contraction[1]] - alpha

    return (Vout, Wout)


def is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp, find_dim_min_overlap=True):
    """
    Check whether there is any overlapping region between the hyperbox represented by 
    minimum point `V_cmp` and maximum point `W_cmp` and any hyperbox in the existing list of 
    hyperboxes belonging to other classes. The detailed information of this
    procedure can be found in [1]_.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of hyperboxes representing other classes compared to V_cmp.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (maximum points) of hyperboxes representing other classes compared to W_cmp.
    V_cmp : array-like of shape (n_features,)
        Minimum point of the compared hyperbox.
    W_cmp : array-like of shape (n_features,)
        Maximum point of the compared hyperbox.
    find_dim_min_overlap : boolean, optional, default=True
        If True, then find the dimension causing the minimum overlap between the hyperbox [V_cmp, W_cmp] and any hyperboxes in the list of 
        hyperboxes represented by [V, W]. Otherwise, only test whether there is any existing overlap zone.

    Returns
    -------
    if find_dim_min_overlap == False:
        return False - no overlap,  True - overlap
    else:
        return:
            + is_overlap: False - no overlap,  True - overlap
            + hyperbox_ids_overlap: indices of hyperboxes overlap with [V_cmp, W_cmp] - numpy array
            + min_overlap_dimension: dimension with minimum overlap value > 0 corresponding to hyperboxes with id located in hyperbox_id_overlap 
            
            if is_overlap == False:
                hyperbox_ids_overlap = min_overlap_dimension = None
                
    References
    ----------
    .. [1] O. N. Al-Sayaydeh, M. F. Mohammed, E. Alhroob, H. Tao, and C. P. Lim,
           "A refined fuzzy min–max neural network with new learning procedures
           for pattern classification," IEEE Transactions on Fuzzy Systems,
           vol. 28, no. 10, pp. 2480-2494, 2019.
    
    """
    if (V is None) or (len(V) == 0):
        return False

    if (V_cmp > W_cmp).any() == True:
        return False
    else:
        n_samples = V.shape[0]
        V_cmp_tile = np.repeat([V_cmp], n_samples, axis=0)
        W_cmp_tile = np.repeat([W_cmp], n_samples, axis=0)
        
        overlap_mat = np.minimum(W_cmp_tile, W) - np.maximum(V_cmp_tile, V)

        overlap_hyperbox_vec = (overlap_mat >= 0).all(axis=1)

        is_overlap = overlap_hyperbox_vec.any()

        if find_dim_min_overlap == False:
            return is_overlap
        else:
            if is_overlap == False:
                return (is_overlap, None, None)
            else:
                # Find the dimension with min overlap values (>0) of hyperboxes overlap with [V_cmp, W_cmp]
                hyperbox_ids_overlap = np.nonzero(overlap_hyperbox_vec)[0]

                V_cmp_tile = np.repeat([V_cmp], len(hyperbox_ids_overlap), axis=0)
                W_cmp_tile = np.repeat([W_cmp], len(hyperbox_ids_overlap), axis=0)

                overlap_value_mat =  np.minimum((W[hyperbox_ids_overlap] - V_cmp_tile), (W_cmp_tile - V[hyperbox_ids_overlap])).astype(np.float64)

                # Find indices of rows in matrix with all values being 0
                all_zero_row_id = ~overlap_value_mat.any(axis=1)

                for index, val in enumerate(all_zero_row_id):
                    if val == False:
                        overlap_value_mat[index] = np.where(overlap_value_mat[index] == 0, np.nan, overlap_value_mat[index])

                min_overlap_dimension = np.nanargmin(overlap_value_mat, axis=1)

                return (is_overlap, hyperbox_ids_overlap, min_overlap_dimension)


def hyperbox_contraction_rfmnn(V, W, C, ids_parent_box, id_child_box, overlap_dim, scale=0.001):
    """
    Adjusting or splitting min-max points of overlaping clusters in the refined
    fuzzy min-max neural network classifier. The detailed information of this
    procedure can be found in [1]_.

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (minimum points) of all existing hyperboxes in the trained model.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes.
    ids_parent_box : array-like of shape (n_parent_hyperboxes,)
        List of indices of parent hyperboxes fully containing the child hyperbox.
    id_child_box : int
        Index of a child hyperbox fully contained in parent hyperboxes
    overlap_dim : array-like of shape (n_parent_hyperboxes,)
        The overlapped dimensions between parent hyperboxes and the child hyperbox need to make contraction.
    scale : float, optional, default=0.001
        A buffer value is used to avoid overlap on the edges between hyperboxes after contraction

    Returns
    -------
    Vout : array-like of shape (n_hyperboxes, n_features)
        Lower bounds (minimum points) of all existing hyperboxes after doing contraction.
    Wout : array-like of shape (n_hyperboxes, n_features)
        Upper bounds (minimum points) of all existing hyperboxes after doing contraction.
    Cout : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes after doing contraction.
        
    References
    ----------
    .. [1] O. N. Al-Sayaydeh, M. F. Mohammed, E. Alhroob, H. Tao, and C. P. Lim,
           "A refined fuzzy min–max neural network with new learning procedures
           for pattern classification," IEEE Transactions on Fuzzy Systems,
           vol. 28, no. 10, pp. 2480-2494, 2019.

    """
    Vout = V.copy()
    Wout = W.copy()
    Cout = C.copy()

    for index, it in enumerate(ids_parent_box):
        if Vout[it, overlap_dim[index]] < Vout[id_child_box, overlap_dim[index]] and Wout[id_child_box, overlap_dim[index]] < Wout[it, overlap_dim[index]]:
            Wj2 = Wout[it].copy()
            Wout[it, overlap_dim[index]] = Vout[id_child_box, overlap_dim[index]] - scale
            Vj2 = Vout[it].copy()
            Vj2[overlap_dim[index]] = Wout[id_child_box, overlap_dim[index]] + scale
            
            Vout = np.concatenate((Vout, Vj2.reshape(1, -1)), axis = 0)
            Wout = np.concatenate((Wout, Wj2.reshape(1, -1)), axis = 0)
            Cout = np.concatenate((Cout, [C[it]]))       

    return (Vout, Wout, Cout)


def is_overlap_cat_features_one_by_one(E1, F1, E2, F2):
    """
    Check whether all categorical features of two input hyperboxes repsented by
    lower bounds `E1`, `E2` and upper bounds `F1`, `F2` overlap with each
    other.

    Parameters
    ----------
    E1 : array-like of shape (n_cat_features, )
        Lower bound for categorical features of the first hyperbox.
    F1 : array-like of shape (n_cat_features, )
        Upper bound for categorical features of the first hyperbox.
    E2 : array-like of shape (n_cat_features, )
        Lower bound for categorical features of the second hyperbox.
    F2 : array-like of shape (n_cat_features, )
        Upper bound for categorical features of the second hyperbox.

    Returns
    -------
    bool
        True if all categorical features in bounds overlap with each other.
        Otherwise, return False.

    """
    if E1.ndim == 1:
        n_cat_features = len(E1)
    else:
        n_cat_features = E1.shape[1]
        E1 = E1.flatten()
        E2 = E2.flatten()
        F1 = F1.flatten()
        F2 = F2.flatten()

    for i in range(n_cat_features):
        if (E1[i] != E2[i]) and (E1[i] != F2[i]) and (F1[i] != E2[i]) and (F1[i] != F2[i]):
            return False
        else:
            if (E1[i] != E2[i]) and (F1[i] == DEFAULT_CATEGORICAL_VALUE) and (F2[i] == DEFAULT_CATEGORICAL_VALUE):
                return False
    # two hyperboxes with categorical features overlap when all dims overlap
    return True


def is_overlap_cat_features_one_vs_many(E1, F1, E, F, tested_box_ids=[]):
    """
    Check for overlap in categorical features between an input hyperbox and
    a list of existing hyperboxes.

    Parameters
    ----------
    E1 : array-like of shape (n_cat_features, )
        Lower bound for categorical features of the hyperbox which needs to
        check for overlap.
    F1 : array-like of shape (n_cat_features, )
        Upper bound for categorical features of the hyperbox which needs to
        check for overlap.
    E : array-like of shape (n_hyperboxes, n_cat_features)
        Lower bounds for categorical features of all existing hyperboxes.
    F : array-like of shape (n_hyperboxes, n_cat_features)
        Upper bounds for categorical features of all existing hyperboxes.
    tested_box_ids : a list of int, optional, default=[]
        The indices of existing hyperboxes with which the input hyperbox needs
        to check overlap.

    Returns
    -------
    bool
        Return True if the categorical features of the input hyperbox overlap
        with any existing hyperboxes that are checked for.

    """
    if len(tested_box_ids) > 0:
        for i in tested_box_ids:
            if is_overlap_cat_features_one_by_one(E1, F1, E[i], F[i]) == True:
                return True

    return False


def hyperbox_overlap_test_freq_cat_gfmm(E, F, id_extended_box, id_tested_box, X_cat, similarity_of_cat_vals, cat_overlap_resolved_hyperbox_id):
    """
    Test overlap in categorical features between two input hyperboxes.

    Parameters
    ----------
    E : array-like of shape (n_hyperboxes, n_cat_features)
        Lower bounds for categorical features of all existing hyperboxes.
    F : array-like of shape (n_hyperboxes, n_cat_features)
        Upper bounds for categorical features of all existing hyperboxes.
    id_extended_box : int
        Index of the extended hyperbox which needs to test for overlap.
    id_tested_box : int
        Index of the hyperbox to test for overlap with the extended hyperbox.
    X_cat : array-like of shape (n_samples, n_cat_features)
        Categorical features of all training data.
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical
        values for each categorical feature index. Each element in this array
        is an dictionary with keys being a hashed value of two categorical
        values and values of this dictionary being a similarity value.
    cat_overlap_resolved_hyperbox_id : a list of int
        Indices of hyperboxes overlapping with the extended hyperbox but
        the overlapping regions among them were resolved by changing values
        on only categorical features. When replacing an overlap area by other
        values, we are not allowed to create the overlapping regions with
        the hyperboxes having indices stored in this list.

    Returns
    -------
    dim : a list of two element in the form of [dimension, replaced values]
        If this list is empty, there is no overlapping area among categorical
        features in two hyperboxes. Otherwise, this list shows the categorical
        dimension where the overlap occurs. If the first element in this list
        (dimension) gets the value of -1, it means that there is an overlapping
        region but we cannot find a suitable value to replace for any dimension
        to resolve the overlap in the categorical features. The second element
        in this list contains two new values for lower and upper bounds in the
        categorical dimension shown in `dimension`. If the second value of this
        list is None, then no change happens in the bounds in that categorical
        dimension.

    """
    dim = []
    is_cat_overlap = is_overlap_cat_features_one_by_one(E[id_extended_box], F[id_extended_box], E[id_tested_box], F[id_tested_box])

    if is_cat_overlap == True:
        # resolving overlap by contracting the one dimesion of the extended hyperbox
        min_change = 2
        dimension = -1
        replaced_vals = [None, None]
        for i in range(E.shape[1]):
            unq_val_fe_i = np.unique(X_cat[:, i])
            min_change_i = 2
            newVal_lower = None
            newVal_upper = None
            is_changed_lower = False
            is_changed_upper = False
            is_overlap_lower = False
            is_overlap_upper = False
            cur_dist = similarity_of_cat_vals[i][hashing(E[id_extended_box, i], F[id_extended_box, i])]
            if (E[id_extended_box, i] == E[id_tested_box, i]) or (E[id_extended_box, i] == F[id_tested_box, i]):
                # replace the value at E[id_extended_box, i]
                # get the other values different from E[id_extended_box, i]
                is_overlap_lower = True
                replaceable_vals = unq_val_fe_i[unq_val_fe_i != E[id_extended_box, i]]
                if len(replaceable_vals) > 0:
                    # find the value such that similarity(replaceable_val, F[id_extended_box, i]) < similarity(E[id_extended_box, i], F[id_extended_box, i]) and the change is the minimum
                    # Note the replaced value is also different from E[id_tested_box, i] and F[id_tested_box, i]
                    for val in replaceable_vals:
                        if (val != E[id_tested_box, i]) and (val != F[id_tested_box, i]):
                            new_dist = similarity_of_cat_vals[i][hashing(val, F[id_extended_box, i])]
                            if (min_change_i > (cur_dist - new_dist)) and (cur_dist >= new_dist):
                                tmp_E_id_extended_box = deepcopy(E[id_extended_box])
                                tmp_E_id_extended_box[i] = val
                                if is_overlap_cat_features_one_vs_many(tmp_E_id_extended_box, F[id_extended_box], E, F, cat_overlap_resolved_hyperbox_id) == False:
                                    min_change_i = cur_dist - new_dist
                                    newVal_lower = val
                                    is_changed_lower = True

            if (F[id_extended_box, i] != DEFAULT_CATEGORICAL_VALUE) and ((F[id_extended_box, i] == E[id_tested_box, i]) or (F[id_extended_box, i] == F[id_tested_box, i]) ) :
                # replace the value at F[id_extended_box, i]
                is_overlap_upper = True
                if (is_overlap_lower == False) or ((is_overlap_lower == True) and (is_changed_lower == True)):
                    replaceable_vals = unq_val_fe_i[unq_val_fe_i != F[id_extended_box, i]]
                    if len(replaceable_vals) > 0:
                        # find the value such that similarity(E[id_extended_box, i], replaceable_val) < similarity(E[id_extended_box, i], F[id_extended_box, i]) and the change is the minimum
                        # Note the replaced value is also different from E[id_tested_box, i] and F[id_tested_box, i]
                        min_change_i = 2
                        for val in replaceable_vals:
                            if (val != E[id_tested_box, i]) and (val != F[id_tested_box, i]):
                                if is_changed_lower == True:
                                    new_dist = similarity_of_cat_vals[i][hashing(newVal_lower, val)]
                                else:
                                    new_dist = similarity_of_cat_vals[i][hashing(E[id_extended_box, i], val)]
                                if (min_change_i > (cur_dist - new_dist)) and (cur_dist >= new_dist):
                                    tmp_F_id_extended_box = deepcopy(F[id_extended_box])
                                    tmp_F_id_extended_box[i] = val
                                    tmp_E_id_extended_box = deepcopy(E[id_extended_box])
                                    if is_changed_lower == True:
                                        tmp_E_id_extended_box[i] = newVal_lower
                                    if is_overlap_cat_features_one_vs_many(tmp_E_id_extended_box, tmp_F_id_extended_box, E, F, cat_overlap_resolved_hyperbox_id) == False:
                                        min_change_i = cur_dist - new_dist
                                        newVal_upper = val
                                        is_changed_upper = True

            is_changed = True  
            if (is_overlap_lower == True) and (is_changed_lower == False):
                is_changed = False
            if (is_overlap_upper == True) and (is_changed_upper == False):
                is_changed = False
            
            if (is_changed == True) and (min_change_i < min_change):
                min_change = min_change_i
                dimension = i
                replaced_vals[0] = newVal_lower
                replaced_vals[1] = newVal_upper

        dim = [dimension, replaced_vals]

    return dim


def hyperbox_contraction_freq_cat_gfmm(Ei, Fi, case_contraction):
    """
    Perform hyperbox contraction in categorical features for a given hyperbox.

    Parameters
    ----------
    Ei : array-like of shape (n_cat_features,)
        Lower bounds for categorical features of the hyperbox which need to do
        contraction.
    Fi : array-like of shape (n_cat_features,)
        Upper bounds for categorical features of the hyperbox which need to do
        contraction.
    case_contraction : a list of two elements
        This is a special struct which is the outcomes of the
        :func:`hyperbox_overlap_test_freq_cat_gfmm` function to determine
        the overlap test case and corresponding overlapped dimension for
        categorical features.

    Returns
    -------
    E_out : array-like of shape (n_cat_features, )
        Lower bounds for categorical features of the hyperboxes contracted.
    F_out : array-like of shape (n_cat_features, )
        Upper bounds for categorical features of the hyperboxes contracted.

    """
    E_out = Ei.copy()
    F_out = Fi.copy()

    if len(case_contraction) > 0 and case_contraction[0] != -1:
        if case_contraction[1][0] is not None:
            E_out[case_contraction[0]] = case_contraction[1][0]
        if case_contraction[1][1] is not None:
            F_out[case_contraction[0]] = case_contraction[1][1]

    return (E_out, F_out)


def is_overlap_one_many_diff_label_hyperboxes_mixed_data_general(V, W, D, N_samples, V_cmp, W_cmp, D_cmp, N_samples_cmp):
    """
    Check whether an input hyperbox overlaps with any hyperboxes representing
    different classes with the input hyperbox

    Parameters
    ----------
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        Lower continuous bounds (minimum points) of hyperboxes representing
        other classes compared to V_cmp.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        Upper continuous bounds (maximum points) of hyperboxes representing
        other classes compared to W_cmp.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores a special structure for categorical features of all
        hyperboxes representing other classes compared to D_cmp. Each element
        in `D` stores a set of symbolic values with their cardinalities for
        the j-th categorical dimension of a given hyperbox.
    N_samples : array-like of shape (n_hyperboxes, )
        A vector save the number of samples contained in each hyperbox stored
        in the lists of V, W, D
    V_cmp : array-like of shape (n_continuous_features,)
        Minimum point of the compared hyperbox.
    W_cmp : array-like of shape (n_continuous_features,)
        Maximum point of the compared hyperbox.
    D_cmp : array-like of shape (n_cat_features,)
        Categorical bound of the compared hyperbox. It contains a set of
        symbolic values with their cardinalities for the j-th categorical
        dimension of the compared hyperbox.
    N_samples_cmp : int
        A scalar storing the number of hyperboxes included in the hyperbox
        represented by [V_cmp, W_cmp, D_cmp]

    Returns
    -------
    bool
        Show if the input hyperbox overlaps with any hyperbox in the list of
        hyperboxes representing the classes other than the input hyperbox.

    """
    # Note: if the compared hyperbox does not overlap with any other hyperboxes
    # on the continuous dims, we do not need to check the categorical dims
    # The checking for the categorical dims is performed for only overlapped
    # hyperboxes on the numerical dims
    if V_cmp is not None:
        if (V is None) or (len(V) == 0):
            return False

        if (V_cmp > W_cmp).any() == True:
            return False
        else:
            # examine only hyperboxes w/o missing dimensions, meaning that in
            # each dimension upper bound is larger than lower bound
            id_considered_hyperboxes = np.nonzero((W >= V).all(axis=1))[0]

            if len(id_considered_hyperboxes) == 0:
                return False
            else:
                b = membership_func_gfmm(W_cmp, V_cmp, V[id_considered_hyperboxes], W[id_considered_hyperboxes], 1)
                # find indices of the overlapping hyperboxes
                id_overlaped_hyperboxes = np.nonzero(b == 1)[0]
                if len(id_overlaped_hyperboxes) == 0:
                    # No overlap
                    return False

                if D_cmp is None:
                    return True
    else:
        id_overlaped_hyperboxes = list(range(D.shape[0]))

    D = D[id_overlaped_hyperboxes, :]
    N_samples = N_samples[id_overlaped_hyperboxes]
    n_cat_features = D.shape[1]

    for ind, hyperbox in enumerate(D):
        is_overlap = np.ones(n_cat_features, dtype=bool)
        for j in range(n_cat_features):
            inter_cat_val = np.intersect1d(list(hyperbox[j].keys()), list(D_cmp[j].keys()))
            if len(inter_cat_val) > 0:
                diff_all = True
                for it in inter_cat_val:
                    if hyperbox[j][it] / N_samples[ind] == D_cmp[j][it] / N_samples_cmp:
                        # overlap on dimension j
                        diff_all = False
                        break

                if diff_all == True:
                    # No categorical value between two cat features having the
                    # same probability => Two hyperboxes do not overlap
                    is_overlap[j] = False
                    break
            else:
                # All categorical values on the dim j are different between
                # two hyperboxes => Two hyperboxes do not overlap
                is_overlap[j] = False
                break

        if is_overlap.all() == True:
            # All categorical dims overlap between two hyperboxes
            return True

    # No hyperbox overlaps with the considered hyperbox on the categorical features
    return False


def is_two_hyperboxes_overlap_num_data_free_range_general(Vi, Wi, Vk, Wk):
    """Check if two hyperboxes `Bi` and `Bk` overlap with each other or not.
    This function uses a general formula concerning the determination of
    an hyperbox within the overlaping region. If this hyperbox exists for all
    dimensions, two hyperboxes `Bi` and `Bk` overlap, else no overlap occurs.

    Parameters
    ----------
    Vi : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bi`.
    Wi : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bi`.
    Vk : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bk`.
    Wk : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bk`.

    Returns
    -------
    is_overlap : boolean
        Show if two hyperboxes `Bi` and `Bk` overlap or not.

    """
    overlap_vec = np.minimum(Wi, Wk) - np.maximum(Vi, Vk)
    is_overlap = (overlap_vec >= 0).all()

    return is_overlap


def overlap_resolving_num_data_free_range(Vi, Wi, ci, Vk, Wk, ck, alpha=0.00001):
    """Resolve overlap between two hyperboxes `Bi` and `Bk` with coordinates
    being numerical features with unlimited ranges. For more details regarding
    the way of contracting two overlapping hyperboxes, please see [1]_.

    Parameters
    ----------
    Vi : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bi`.
    Wi : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bi`.
    ci : int
        Class label of the hyperbox `Bi`.
    Vk : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bk`.
    Wk : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bk`.
    ck : int
        Class label of the hyperbox `Bk`.
    alpha : float
        A very small value is used to avoid the overlap between two hyperboxes
        after contraction.

    Returns
    -------
    Vi : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bi` after contraction.
    Wi : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bi` after contraction.
    Vk : array-like of shape (n_features,)
        Minimum coordinate of the hyperbox `Bk` after contraction.
    Wk : array-like of shape (n_features,)
        Maximum coordinate of the hyperbox `Bk` after contraction.
        
    References
    ----------
    .. [1] T.T. Khuat and B. Gabrys, "Accelerated learning algorithms of general 
           fuzzy min-max neural network using a novel hyperbox selection rule,"
           Information Sciences, vol. 547, pp. 887-909, 2021.
           
    """
    if (ci == ck) and (ci != UNLABELED_CLASS):
        return (Vi, Wi, Vk, Wk)

    delta_j = np.finfo(np.float).max
    case = -1
    min_j = -1
    for j in range(len(Vi)):
        if (Vi[j] <= Vk[j]) and (Vk[j] < Wi[j]) and (Wi[j] <= Wk[j]):
            if delta_j > Wi[j] - Vk[j]:
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 1
        elif (Vk[j] <= Vi[j]) and (Vi[j] < Wk[j]) and (Wk[j] <= Wi[j]):
            if delta_j > Wk[j] - Vi[j]:
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 2
        elif (Vi[j] < Vk[j]) and (Wk[j] < Wi[j]):
            if (delta_j > (Wk[j] - Vi[j])) and ((Wk[j] - Vi[j]) <= (Wi[j] - Vk[j])):
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 31
            elif delta_j > (Wi[j] - Vk[j]):
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 32
        elif (Vk[j] < Vi[j]) and (Wi[j] < Wk[j]):
            if (delta_j > (Wk[j] - Vi[j])) and ((Wk[j] - Vi[j]) <= (Wi[j] - Vk[j])):
                delta_j = Wk[j] - Vi[j]
                min_j = j
                case = 41
            elif delta_j > (Wi[j] - Vk[j]):
                delta_j = Wi[j] - Vk[j]
                min_j = j
                case = 42
        # Here we add four more overlapping cases between two hyperboxes
        elif (Vi[j] <= Wi[j]) and (Wi[j] == Vk[j]) and (Vk[j] < Wk[j]):
            # S1: Vij <= Wij = Vkj < Wkj
            delta_j = 0
            min_j = j
            case = 5
            break
        elif (Vi[j] < Wi[j]) and (Wi[j] == Vk[j]) and (Vk[j] <= Wk[j]):
            # S2: Vij < Wij = Vkj <= Wkj
            delta_j = 0
            min_j = j
            case = 6
            break
        elif (Vk[j] <= Wk[j]) and (Wk[j] == Vi[j]) and (Vi[j] < Wi[j]):
            # S3: Vkj <= Wkj = Vij < Wij
            delta_j = 0
            min_j = j
            case = 7
            break
        elif (Vk[j] < Wk[j]) and (Wk[j] == Vi[j]) and (Vi[j] <= Wi[j]):
            # S4: Vkj < Wkj = Vij <= Wij
            delta_j = 0
            min_j = j
            case = 8
            break

    # Adjust the coordinates of two hyperboxes Bi and Bk according to the overlap cases
    if case == 1:
        tmp = (Wi[min_j] + Vk[min_j]) / 2
        Wi[min_j] = tmp
        Vk[min_j] = tmp  + alpha
    elif case == 2:
        tmp = (Wk[min_j] + Vi[min_j]) / 2
        Vi[min_j] =  tmp + alpha
        Wk[min_j] = tmp
    elif case == 31:
        Vi[min_j] = Wk[min_j] + alpha
    elif case == 32:
        Wi[min_j] = Vk[min_j] - alpha
    elif case == 41:
        Wk[min_j] = Vi[min_j] - alpha
    elif case == 42:
        Vk[min_j] = Wi[min_j] + alpha
    elif case == 5:
        if (Vk[min_j] + alpha) < Wk[min_j]:
            Vk[min_j] = Vk[min_j] + alpha
        else:
            Vk[min_j] = Wk[min_j]
    elif case == 6:
        if (Wi[min_j] - alpha) > Vi[min_j]:
            Wi[min_j] = Wi[min_j] - alpha
        else:
            Wi[min_j] = Vi[min_j]
    elif case == 7:
        if (Vi[min_j] + alpha) < Wi[min_j]:
            Vi[min_j] = Vi[min_j] + alpha
        else:
            Vi[min_j] = Wi[min_j]
    elif case == 8:
        if (Wk[min_j] - alpha) > Vk[min_j]:
            Wk[min_j] = Wk[min_j] - alpha
        else:
            Wk[min_j] = Vk[min_j]

    return (Vi, Wi, Vk, Wk)
