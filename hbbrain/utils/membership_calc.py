"""
Contain all functions supporting for computing fuzzy membership values in
various ways.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
from copy import deepcopy
from hbbrain.utils.matrix_transformation import hashing_mat
from hbbrain.constants import DEFAULT_CATEGORICAL_VALUE


def _ramp_func(z, g):
    """A ramp threshold function for fuzzy membership calculation.

    Notes
    -----
        f = 1,          if z*g > 1\n
        f = z*g,        if 0 <= z*g <= 1\n
        f = 0,          if z*g < 0\n

    Parameters
    ----------
    z : array-like of shape (n_hyperboxes, n_features)
        A vector stores `n_hyperboxes` values, where `n_hyperboxes` is the
        number of hyperboxes used to compute membership values.
    g : float or array-like of shape (n_features,)
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    f : array-like of shape (n_hyperboxes, n_features)
        The outputs of the ramp function.

    """
    if np.size(g) > 1:  # the parameter g is a vector, not a scalar
        p = z * (np.ones((z.shape[0], 1)) * g)
    else:
        p = z * g

    f = ((p >= 0) * (p <= 1)).astype(np.float64) * p + (p > 1).astype(np.float64)

    return f


def n_cat_features_containing_bit_one(v):
    """This function is to count number of categorical features in v in which
    there is at least one bit 1
    """
    features = [np.any(i) for i in v]

    return np.sum(features)


def bitwise_membership(x_cat, D):
    """Compute membership values between categorical features in the input
    pattern `X_cat` and all categorical features of existing hyperboxes stored
    in `D`.

    Parameters
    ----------
    x_cat : array-like of shape (n_cat_features, )
        Categorical features of an input pattern. Each feature is represented
        by an array of one-hot encoded values.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all bounds of categorical features for all existing
        hyperboxes, in which each row stores categorical features of a
        hyperbox.

    Returns
    -------
    mem_val : array-like of shape (n_hyperboxes, ).
        An array stores the degrees of membership from the input pattern to
        all existing hyperboxes which are computed based on categorical
        features.
    """
    if D.ndim == 1:
        D = deepcopy(D).reshape(-1, 1)

    mem_bit_and = np.bitwise_and(D, x_cat)
    n_cat_features = D.shape[1]

    # Count number of bit 1 in each cat feature. For each cat feature,
    # if there is at least one True in the one-hot-vector, then return 1
    mem_val = [n_cat_features_containing_bit_one(i) / n_cat_features
               for i in mem_bit_and]

    mem_val = np.array(mem_val)

    return mem_val


def f_sim_freq_cat_features(x_cat, E, similarity_of_cat_vals):
    """
    Compute similarity values in each categorical dimension between a
    input categorical features `x_cat` and each element in the current list of
    categorical bounds of existing hyperboxes.

    Parameters
    ----------
    x_cat : array-like of shape (n_cat_features, )
        Categorical features of an input pattern.
    E : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all bounds of categorical features for all existing
        hyperboxes, in which each row stores categorical features of a
        hyperbox.
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical
        values for each categorical feature index. Each element in this array
        is an dictionary with keys being a hashed value of two categorical
        values and values of this dictionary being a similarity value.

    Returns
    -------
    sim_vals : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores similarity values in each categorical dimension
        between the input categorical features and all bounds of categorical
        features of existing hyperboxes.

    """
    X_cat_mat = np.ones((E.shape[0], 1)) * x_cat
    val_mat = hashing_mat(X_cat_mat, E)

    sim_vals = np.array([similarity_of_cat_vals[idj].get(j, 0) for i in val_mat for idj, j in enumerate(i)]).reshape(val_mat.shape)

    return sim_vals


def membership_function_freq_cat(x_cat, E, F, similarity_of_cat_vals):
    """
    Compute membership degrees between input categorical features and all
    lower and upper bounds of categorical features of existing hyperboxes.

    Parameters
    ----------
    x_cat : array-like of shape (n_cat_features, )
        Categorical features of an input pattern.
    E : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all lower bounds of categorical features for all
        existing hyperboxes, in which each row stores a lower categorical
        features bound for a hyperbox.
    F : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all upper bounds of categorical features for all
        existing hyperboxes, in which each row stores a upper categorical
        features bound for a hyperbox.
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical
        values for each categorical feature index. Each element in this array
        is an dictionary with keys being a hashed value of two categorical
        values and values of this dictionary being a similarity value.

    Returns
    -------
    b : array-like of shape (n_hyperboxes, ).
        An array stores the degrees of membership from the input pattern to
        all existing hyperboxes which are computed based on categorical
        features.

    """
    violMin = 1 - f_sim_freq_cat_features(x_cat, E, similarity_of_cat_vals)
    violMax = 1 - f_sim_freq_cat_features(x_cat, F, similarity_of_cat_vals)
    # Note: If one of two values with respect to E and F is 1 => return 1
    # Note: If F gets default values => its mem = 1, then the mem value depends on the E
    # If x_e = F => return 1, and in this case violMin = 1 to mem = 1
    violMax[(violMin == 1) & (E != DEFAULT_CATEGORICAL_VALUE)] = 1
    violMin[(violMax == 1) & (F != DEFAULT_CATEGORICAL_VALUE)] = 1

    b = np.minimum(violMax, violMin).min(axis=1)

    return b


def membership_func_gfmm(xl, xu, V, W, g=1):
    """Compute fuzzy membership values between an input pattern and a list of
    existing hyperboxes of a general fuzzy min-max neural network.

    For more details regarding how to calculate fuzzy membership values, please
    refer to the publications [1]_ and [2]_.

    .. note::

        This function provides the degrees of membership b of an input pattern
        `x` (in form of upper bound `xu` and lower bound `xl`) with respect to
        the existing hyperboxes described by minimal points `V` and maximal
        points `W`. The sensitivity parameter `g` regulates how fast the
        membership values decrease when an input pattern is separeted from
        hyperbox core.

    Parameters
    ----------
    xl : array-like of shape (n_features,)
        Lower bound of an input pattern.
    xu : array-like of shape (n_features,)
        Upper bound of an input pattern.
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points of all existing hyperboxes,
        in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points of all existing hyperboxes,
        in which each row is a maximal point of a hyperbox.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    b : array-like of shape (n_hyperboxes,)
        Degrees of membership of the input pattern X=[Xl, Xu] corresponding to
        each hyperbox in the current list of existing hyperboxes.

    References
    ----------
    .. [1] Gabrys, B., & Bargiela, A. (2000). General fuzzy min-max neural
           network for clustering and classification. IEEE transactions on
           neural networks, 11(3), 769-783.
    .. [2] Khuat, T. T., & Gabrys, B. (2021). Accelerated learning algorithms
           of general fuzzy min-max neural network using a novel hyperbox
           selection rule. Information Sciences, 547, 887-909.

    """
    yW = W.shape[0]
    ones_mat = np.ones((yW, 1))
    viol_max = 1 - _ramp_func(ones_mat * xu - W, g)
    viol_min = 1 - _ramp_func(V - ones_mat * xl, g)

    b = np.minimum(viol_max, viol_min).min(axis=1)

    return b


def asym_similarity_val_one_many_hyperboxes(xl, xu, V, W, g=1, asimil_type='max'):
    """
    Calculate the asymetrical similarity value of a specific hyperbox
    (lower bound - xl, upper bound - xu) and hyperboxes having lower and upper
    bounds stored in two matrix V and W respectively

    Parameters
    ----------
    xl : array-like of shape (n_features,)
        Lower bound of an input hyperbox.
    xu : array-like of shape (n_features,)
        Upper bound of an input hyperbox.
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points of all existing hyperboxes,
        in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points of all existing hyperboxes,
        in which each row is a maximal point of a hyperbox.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.
    asimil_type : {'max', 'min'}, optional, default='max'
        Type of handling asymmetric similarity matrix.

    Returns
    -------
    b : array-like of shape (n_hyperboxes,)
        Similarity values of the specific hyperbox with all hyperboxes having
        lower and upper bounds in V and W.
    """
    num_hyperboxes = W.shape[0]

    Vk = np.tile(xl, [num_hyperboxes, 1])
    Wk = np.tile(xu, [num_hyperboxes, 1])

    viol_max1 = 1 - _ramp_func(Wk - W, g)
    viol_min1 = 1 - _ramp_func(V - Vk, g)

    viol_max2 = 1 - _ramp_func(W - Wk, g)
    viol_min2 = 1 - _ramp_func(Vk - V, g)

    b1 = np.minimum(viol_max1, viol_min1).min(axis=1)
    b2 = np.minimum(viol_max2, viol_min2).min(axis=1)

    if asimil_type == 'max':
        b = np.maximum(b1, b2)
    else:
        b = np.minimum(b1, b2)

    return b


def get_membership_gfmm_all_classes(Xl, Xu, V, W, C, g=1):
    """
    Return membership values (according to the membership function of the GFMM
    classifiers) with respect to all class labels between the input patterns
    stored in two lower and upper bound input matrices `Xl` and `Xu` and
    existing hyperboxes represented by two matrices of minimum and maximum
    points `V` and `W` together with corresponding class labels in vector `C`.

    Parameters
    ----------
    Xl : array-like of shape (n_samples, n_features) or (n_features, )
        Lower bounds of input samples.
    Xu : array-like of shape (n_samples, n_features) or (n_features, )
        Upper bounds of input samples.
    V : array-like of shape (n_hyperboxes, n_features)
        Minimum points of the existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Maximum points of the existing hyperboxes in the trained model.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes corresponding to the values
        stored in `V` and `W`.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    mem_vals_matrix : array-like of shape (n_samples, n_classes)
        Membership values with regard to all class labels for each input
        sample. Each row is a vector of membership values. Each column
        represents an index of a class label sorted in an ascending order
        of class labels.
    hyperbox_ids_matrix : array-like of shape (n_samples, n_classes)
        Storing the indices of hyperboxes corresponding to membership values
        for classes.
    """
    if len(Xl.shape) == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)

    n_samples = Xl.shape[0]
    class_vals = np.unique(C)
    n_classes = len(class_vals)
    mem_vals_matrix = np.zeros((n_samples, n_classes), dtype=float)
    hyperbox_ids_matrix = np.zeros((n_samples, n_classes), dtype=int)
    # Get membership values for each sample
    is_exist_missing_value = (V > W).any()
    for i in range(n_samples):
        # calculate memberships for all hyperboxes
        if is_exist_missing_value == False:
            mem_vals = membership_func_gfmm(Xl[i, :], Xu[i, :], V, W, g)
        else:
            mem_vals = membership_func_gfmm(Xl[i, :], Xu[i, :], np.minimum(V, W), np.maximum(W, V), g)

        class_c_mem = np.zeros(n_classes)
        class_c_hyperbox_id = np.zeros(n_classes)
        for _id, c in enumerate(class_vals):
            # Find all hyperboxes showing the same class as c
            id_c = np.nonzero(C == c)[0]
            id_c_max_mem = mem_vals[id_c].argmax()
            # Get maximum membership values among hyperboxes with the same
            # class as c
            class_c_mem[_id] = mem_vals[id_c[id_c_max_mem]]
            class_c_hyperbox_id[_id] = id_c[id_c_max_mem]

        mem_vals_matrix[i] = class_c_mem
        hyperbox_ids_matrix[i] = class_c_hyperbox_id

    return (mem_vals_matrix, hyperbox_ids_matrix)


def _min_func(x, g):
    """
    Minimum function is used to compute membership values of Simpson fuzzy
    min-max neural network and its improved versions

    .. math:: f = g * min(1, x)

    Parameters
    ----------
    x : array-like of shape (n_hyperboxes, n_features)
        A vector stores `n_hyperboxes` values, where `n_hyperboxes` is the
        number of hyperboxes used to compute membership values.
    g : float or array-like of shape (n_features,)
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    f : array-like of shape (n_hyperboxes, n_features)
        The outputs of the ramp function.

    """
    n_hyperboxes, n_features = x.shape

    if np.size(g) > 1:
        f = (np.ones((n_hyperboxes, 1)) * g) * np.minimum(np.ones((n_hyperboxes, n_features)), x)
    else:
        f = g * np.minimum(np.ones((n_hyperboxes, n_features)), x)

    return f


def membership_func_fmnn(x, V, W, g=1):
    """Compute fuzzy membership values between an input pattern and a list of
    existing hyperboxes of a fuzzy min-max neural network and its improved
    versions.

    For more details regarding how to calculate fuzzy membership values, please
    refer to the publication [1]_.

    .. note::

        This function provides the degrees of membership b of an input pattern
        `x` with respect to the existing hyperboxes described by minimal points
        `V` and maximal points `W`. The sensitivity parameter `g` regulates how
        fast the membership values decrease when an input pattern is separeted
        from hyperbox core.

    Parameters
    ----------
    x : array-like of shape (n_features,)
        An input pattern.
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points of all existing hyperboxes,
        in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points of all existing hyperboxes,
        in which each row is a maximal point of a hyperbox.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    b : array-like of shape (n_hyperboxes,)
        Degrees of membership of the input pattern X=[Xl, Xu] corresponding to
        each hyperbox in the current list of existing hyperboxes.

    References
    ----------
    .. [1] Simpson, P. (1992). Fuzzy min—max neural networks—Part 1:
           Classiﬁcation. IEEE transactions on neural networks, 3(5), 776-786.

    """
    n_hyperboxes, n_features = V.shape
    X_mat = np.ones((n_hyperboxes, 1)) * x
    zeros_mat = np.zeros((n_hyperboxes, n_features))

    viol_max1 = np.maximum(zeros_mat, 1 - np.maximum(zeros_mat, _min_func(X_mat - W, g)))
    viol_max2 = np.maximum(zeros_mat, 1 - np.maximum(zeros_mat, _min_func(V - X_mat, g)))

    viol_mat = viol_max1 + viol_max2

    b = np.sum(viol_mat, axis=1) / (2 * n_features)

    return b


def get_membership_fmnn_all_classes(X, V, W, C, g=1):
    """
    Return membership values (according to the membership function of the FMNN
    classifiers) with respect to all class labels between the input patterns
    stored in the matrix `X` and existing hyperboxes represented by two
    matrices of minimum and maximum points `V` and `W` together with the
    corresponding class labels in the vector `C`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or (n_features, )
        Input samples.
    V : array-like of shape (n_hyperboxes, n_features)
        Minimum points of the existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Maximum points of the existing hyperboxes in the trained model.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes corresponding to the values
        stored in `V` and `W`.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    mem_vals_matrix : array-like of shape (n_samples, n_classes)
        Membership values with respect to all class labels for each input
        sample. Each row is a vector of membership values. Each column
        represents an index of a class label sorted in an ascending order of
        class labels.
    hyperbox_ids_matrix : array-like of shape (n_samples, n_classes)
        Storing the indices of hyperboxes corresponding to membership values
        for classes.
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    n_samples = X.shape[0]
    class_vals = np.unique(C)
    n_classes = len(class_vals)
    mem_vals_matrix = np.zeros((n_samples, n_classes), dtype=float)
    hyperbox_ids_matrix = np.zeros((n_samples, n_classes), dtype=int)
    # Get membership values for each sample
    for i in range(n_samples):
        # calculate memberships for all hyperboxes
        mem_vals = membership_func_fmnn(X[i, :], V, W, g)
        class_c_mem = np.zeros(n_classes)
        class_c_hyperbox_id = np.zeros(n_classes)
        for _id, c in enumerate(class_vals):
            # Find all hyperboxes showing the same class as c
            id_c = np.nonzero(C == c)[0]
            id_c_max_mem = mem_vals[id_c].argmax()
            # Get maximum membership values among hyperboxes with the same
            # class as c
            class_c_mem[_id] = mem_vals[id_c[id_c_max_mem]]
            class_c_hyperbox_id[_id] = id_c[id_c_max_mem]

        mem_vals_matrix[i] = class_c_mem
        hyperbox_ids_matrix[i] = class_c_hyperbox_id

    return (mem_vals_matrix, hyperbox_ids_matrix)


def membership_func_onehot_gfmm(xl, xu, xd, V, W, D, g=1):
    """
    Compute membership values between an input pattern of which continuous
    features are represented by the lower bound `xl` and the upper bound `xu`
    while categorical features are presented by the bound `xd` and all existing
    hyperboxes with lower and upper bounds stored in `V` and `W` and the
    categorical bound stored in `D`.

    Parameters
    ----------
    xl : array-like of shape (n_continuous_features,)
        Lower bound of continous features of an input pattern.
    xu : array-like of shape (n_continuous_features,)
        Upper bound of continous features of an input pattern.
    xd : array-like of shape (n_cat_features,)
        Categorical features of an input pattern.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all minimal points of continous features for all
        existing hyperboxes, in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        A matrix stores all maximal points of continuous features for all
        existing hyperboxes, in which each row is a maximal point of a hyperbox.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all bounds of categorical features for all existing
        hyperboxes, in which each row contains the bound of a hyperbox.
    g : float or ndarray of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in continuous dimensions.

    Returns
    -------
    b : array-like of shape (n_hyperboxes,)
        Degrees of membership of the input pattern `x=[xl, xu, xd]`
        corresponding to each hyperbox in the current list of existing
        hyperboxes.

    """
    if (xl is not None) and (xd is not None):
        b_con = np.array(membership_func_gfmm(xl, xu, V, W, g))
        b_cat = bitwise_membership(xd, D)

        b = (b_con + b_cat) / 2
    else:
        if xl is not None:
            b = np.array(membership_func_gfmm(xl, xu, V, W, g))
        else:
            b = bitwise_membership(xd, D)

    return b


def get_membership_onehot_gfmm_all_classes(Xl, Xu, Xd, V, W, D, C, g=1):
    """
    Return membership values (according to the membership function of the GFMM
    classifiers) with respect to all class labels between the input patterns
    stored in two lower and upper bound input matrices `Xl` and `Xu` and
    existing hyperboxes represented by two matrices of minimum and maximum
    points `V` and `W` together with corresponding class labels in vector `C`.

    Parameters
    ----------
    Xl : array-like of shape (n_samples, n_continuous_features) or (n_continuous_features, )
        Lower bounds of continuous features of all input samples.
        If None, there are no continous features.
    Xu : array-like of shape (n_samples, n_continuous_features) or (n_continuous_features, )
        Lower bounds of continuous features of all input samples.
        If None, there are no continous features.
    Xd : array-like of shape (n_samples, n_cat_features) or (n_cat_features, )
        Bounds of categorical features of all input patterns.
        If None, there are no categorical features.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        Minimum points of all continuous features of the existing hyperboxes
        in the trained model. If None, there are no continous features.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        Maximum points of all continuous features of the existing hyperboxes
        in the trained model. If None, there are no continous features.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        Bounds of all categorical features of the existing hyperboxes in the
        trained model. If None, there are no categorical features.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes corresponding to the values
        stored in `V`, `W`, and `D`.
    g : float or ndarray of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continous dimension.

    Returns
    -------
    mem_vals_matrix : array-like of shape (n_samples, n_classes)
        Membership values with regard to all class labels for each input
        sample. Each row is a vector of membership values. Each column
        represents an index of a class label sorted in an ascending order
        of class labels.
    hyperbox_ids_matrix : array-like of shape (n_samples, n_classes)
        Storing the indices of hyperboxes corresponding to membership values
        for classes.
    """
    if Xl is not None and len(Xl.shape) == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)
    if Xd is not None and Xd.ndim == 1:
        Xd = Xd.reshape(1, -1)

    if Xl is not None:
        n_samples = Xl.shape[0]
    else:
        n_samples = Xd.shape[0]

    if V is not None:
        is_exist_missing_value = (V > W).any()
    else:
        is_exist_missing_value = False

    class_vals = np.unique(C)
    n_classes = len(class_vals)
    mem_vals_matrix = np.zeros((n_samples, n_classes), dtype=float)
    hyperbox_ids_matrix = np.zeros((n_samples, n_classes), dtype=int)
    # Get membership values for each sample
    for i in range(n_samples):
        # calculate memberships for all hyperboxes
        if Xl is not None and Xd is not None:
            if is_exist_missing_value == False:
                mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], Xd[i], V, W, D, g)
            else:
                mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], Xd[i], np.minimum(V, W), np.maximum(W, V), D, g)
        elif Xl is not None:
            if is_exist_missing_value == False:
                mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], None, V, W, D, g)
            else:
                mem_vals = membership_func_onehot_gfmm(Xl[i], Xu[i], None, np.minimum(V, W), np.maximum(W, V), D, g)
        else:
            mem_vals = membership_func_onehot_gfmm(None, None, Xd[i], V, W, D, g)

        class_c_mem = np.zeros(n_classes)
        class_c_hyperbox_id = np.zeros(n_classes)
        for _id, c in enumerate(class_vals):
            # Find all hyperboxes showing the same class as c
            id_c = np.nonzero(C == c)[0]
            id_c_max_mem = mem_vals[id_c].argmax()
            # Get maximum membership values among hyperboxes with the same
            # class as c
            class_c_mem[_id] = mem_vals[id_c[id_c_max_mem]]
            class_c_hyperbox_id[_id] = id_c[id_c_max_mem]

        mem_vals_matrix[i] = class_c_mem
        hyperbox_ids_matrix[i] = class_c_hyperbox_id

    return (mem_vals_matrix, hyperbox_ids_matrix)


def membership_func_freq_cat_gfmm(xl, xu, x_cat, V, W, E, F, similarity_of_cat_vals, g=1):
    """
    Compute the membership values between an input pattern with respect to all
    hyperboxes (including continous and categorical features). The membership
    values for categorical features is computed based on the occurrence
    frequency values of different class labels with regards to each categorical
    values in each categorical feature.
    
    For more details regarding how to calculate fuzzy membership values, please
    refer to the publications [1]_ and [2]_.

    Parameters
    ----------
    xl : array-like of shape (n_continuous_features, )
        Lower bounds of input continuous features of the input pattern.
    xu : array-like of shape (n_continuous_features, )
        Upper bounds of input continuous features of the input pattern.
    x_cat : array-like of shape (n_cat_features, )
        Categorical features of an input pattern.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        Minimum points of continuous features of the existing hyperboxes
        in the trained model.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        Maximum points of continuous features of the existing hyperboxes
        in the trained model.
    E : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all lower bounds of categorical features for all
        existing hyperboxes, in which each row stores a lower categorical
        features bound for a hyperbox.
    F : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all upper bounds of categorical features for all
        existing hyperboxes, in which each row stores a upper categorical
        features bound for a hyperbox.
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical
        values for each categorical feature index. Each element in this array
        is an dictionary with keys being a hashed value of two categorical
        values and values of this dictionary being a similarity value.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous dimension.

    Returns
    -------
    b : array-like of shape (n_hyperboxes,)
        Degrees of membership of the input pattern X=[Xl, Xu] corresponding to
        each hyperbox in the current list of existing hyperboxes.

    References
    ----------
    .. [1] T.T. Khuat, B. Gabrys, "An in-depth comparison of methods handling
           mixed-attribute data for general fuzzy min–max neural network",
           Neurocomputing, vol. 464, pp. 175-202, 2021.
    .. [2] P.R. Castillo, J. Cardenosa, "Fuzzy min–max neural networks for
           categorical data: application to missing data imputation", Neural
           Computing and Applications, vol. 21, pp. 1349–1362, 2012.

    """
    if (xl is not None) and (x_cat is not None):
        b_con = np.array(membership_func_gfmm(xl, xu, V, W, g))
        b_cat = np.array(membership_function_freq_cat(x_cat, E, F, similarity_of_cat_vals))

        b = np.minimum(b_con, b_cat)
    else:
        if (xl is not None):
            b = np.array(membership_func_gfmm(xl, xu, V, W, g))
        else:
            b = np.array(membership_function_freq_cat(x_cat, E, F, similarity_of_cat_vals))

    return b


def get_membership_freq_cat_gfmm_all_classes(Xl, Xu, X_cat, V, W, E, F, C, similarity_of_cat_vals, g=1):
    """
    Return membership values (according to the membership function of the GFMM
    classifiers) with respect to all class labels between the input patterns
    stored in two lower and upper bound matrices for input continuous features 
    `Xl` and `Xu` and two lower and upper bound matrices for input categorical
    features and existing hyperboxes represented by four matrices of minimum
    and maximum points for continuous features `V` and `W` and lower and upper
    bounds for categorical features `E` and `F` together with corresponding
    class labels in vector `C`.

    Parameters
    ----------
    Xl : array-like of shape (n_samples, n_continuous_features) or (n_continuous_features, )
        Lower bounds of continuous features of all input samples.
        If None, there are no continous features.
    Xu : array-like of shape (n_samples, n_continuous_features) or (n_continuous_features, )
        Lower bounds of continuous features of all input samples.
        If None, there are no continous features.
    X_cat : array-like of shape (n_samples, n_cat_features) or (n_cat_features, )
        Categorical features of all input patterns. If None, there are no
        categorical features.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        Minimum points of all continuous features of the existing hyperboxes
        in the trained model. If None, there are no continous features.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        Maximum points of all continuous features of the existing hyperboxes
        in the trained model. If None, there are no continous features.
    E : array-like of shape (n_hyperboxes, n_cat_features)
        Lower bounds of all categorical features of the existing hyperboxes in
        the trained model. If None, there are no categorical features.
    F : array-like of shape (n_hyperboxes, n_cat_features)
        Upper bounds of all categorical features of the existing hyperboxes in
        the trained model. If None, there are no categorical features.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes corresponding to the values
        stored in `V`, `W`, and `E`, `F`.
    similarity_of_cat_vals : array-like of shape (n_cat_features,)
        An array stores all similarity values among all pairs of categorical
        values for each categorical feature index. Each element in this array
        is an dictionary with keys being a hashed value of two categorical
        values and values of this dictionary being a similarity value.
    g : float or ndarray of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continous dimension.

    Returns
    -------
    mem_vals_matrix : array-like of shape (n_samples, n_classes)
        Membership values with regard to all class labels for each input
        sample. Each row is a vector of membership values. Each column
        represents an index of a class label sorted in an ascending order
        of class labels.
    hyperbox_ids_matrix : array-like of shape (n_samples, n_classes)
        Storing the indices of hyperboxes corresponding to membership values
        for classes.
    """
    if Xl is not None and len(Xl.shape) == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)
    if X_cat is not None and X_cat.ndim == 1:
        X_cat = X_cat.reshape(1, -1)

    if Xl is not None:
        n_samples = Xl.shape[0]
    else:
        n_samples = X_cat.shape[0]

    class_vals = np.unique(C)
    n_classes = len(class_vals)
    mem_vals_matrix = np.zeros((n_samples, n_classes), dtype=float)
    hyperbox_ids_matrix = np.zeros((n_samples, n_classes), dtype=int)

    if V is not None:
        is_exist_missing_continous_value = (V > W).any()
    else:
        is_exist_missing_continous_value = False

    # Get membership values for each sample
    for i in range(n_samples):
        # calculate memberships for all hyperboxes
        if Xl is not None and X_cat is not None:
            if not is_exist_missing_continous_value:
                mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], X_cat[i], V, W, E, F, similarity_of_cat_vals, g)
            else:
                mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], X_cat[i], np.minimum(V, W), np.maximum(W, V), E, F, similarity_of_cat_vals, g)
        elif Xl is not None:
            if not is_exist_missing_continous_value:
                mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], None, V, W, E, F, similarity_of_cat_vals, g)
            else:
                mem_vals = membership_func_freq_cat_gfmm(Xl[i], Xu[i], None, np.minimum(V, W), np.maximum(W, V), E, F, similarity_of_cat_vals, g)
        else:
            mem_vals = membership_func_freq_cat_gfmm(None, None, X_cat[i], V, W, E, F, similarity_of_cat_vals, g)
        class_c_mem = np.zeros(n_classes)
        class_c_hyperbox_id = np.zeros(n_classes)
        for _id, c in enumerate(class_vals):
            # Find all hyperboxes showing the same class as c
            id_c = np.nonzero(C == c)[0]
            id_c_max_mem = mem_vals[id_c].argmax()
            # Get maximum membership values among hyperboxes with the same
            # class as c
            class_c_mem[_id] = mem_vals[id_c[id_c_max_mem]]
            class_c_hyperbox_id[_id] = id_c[id_c_max_mem]

        mem_vals_matrix[i] = class_c_mem
        hyperbox_ids_matrix[i] = class_c_hyperbox_id

    return (mem_vals_matrix, hyperbox_ids_matrix)


def membership_cat_feature_eiol_gfmm(x_cat, D):
    """
    Compute membership degrees between input categorical features and all
    bounds of categorical features of existing hyperboxes for the extended
    improved online learning algorithm of general fuzzy min-max neural network.

    Parameters
    ----------
    x_cat : array-like of shape (n_cat_features, )
        Categorical features of an input pattern.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        A matrix stores all bounds of categorical features for all existing
        hyperboxes, in which each row stores a categorical
        features bound for a hyperbox. Each element :math:`d_{ij} \in D` is a
        set of symbolic values with their cardinalities for the j-th
        categorical dimension of the hyperbox :math:`B_i`. For example,
        :math:`d_{i1} = \{apple : 5, orange : 1\}` means that the first
        categorical feature of the hyperbox :math:`B_i` contains 5 values of
        apple and 1 value of orange.

    Returns
    -------
    b : array-like of shape (n_hyperboxes, ).
        An array stores the degrees of membership from the input pattern to
        all existing hyperboxes which are computed based on categorical
        features.

    """
    n_cat_features = len(x_cat)
    b = np.zeros(D.shape[0])

    for i, hyperbox in enumerate(D):
        s = 0
        n_samples = sum(hyperbox[0].values())
        for j in range(n_cat_features):
            if x_cat[j] in hyperbox[j]:
                s = s + hyperbox[j][x_cat[j]] / n_samples

        b[i] = s / n_cat_features

    return b


def membership_func_extended_iol_gfmm(xl, xu, x_cat, V, W, D, g=1, alpha = 0.5):
    """
    Compute fuzzy membership values between an input pattern and a list of
    existing hyperboxes of a general fuzzy min-max neural network with
    mixed-attribute data.

    .. note::

        This function provides the degrees of membership `b` of an input pattern
        `x` (in form of upper bound `xu` and lower bound `xl` for continuous
        features and categorical features `x_cat`) with respect to the existing
        hyperboxes represented by minimal points `V` and maximal points `W` for
        continuous features and the bound `D` for categorical features. The
        sensitivity parameter `g` regulates how fast the membership values
        decrease when an input continuous pattern is separeted from hyperbox
        core. The parameter `alpha` is the trade-off factor between impacts of
        continuous features and categorical features on the output of membership
        values. Each element :math:`d_{ij} \in D` is a set of symbolic values
        with their cardinalities for the j-th categorical dimension of the
        hyperbox :math:`B_i`. For example, :math:`d_{i1} = \{apple : 5, orange : 1\}`
        means that the first categorical feature of the hyperbox :math:`B_i`
        contains 5 values of apple and 1 value of orange.

    Parameters
    ----------
    xl : array-like of shape (n_continuous_features,)
        Lower bound of continous features of an input pattern.
    xu : array-like of shape (n_continuous_features,)
        Upper bound of continous features of an input pattern.
    x_cat : array-like of shape (n_cat_features, )
        Categorical features of an input pattern.
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
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continous dimension.
    alpha : float, optional, default=0.5
        The trade-off weighting factor between the impacts of categorical
        features and numerical features on the outputs of membership values.

    Returns
    -------
    b : array-like of shape (n_hyperboxes,)
        Degrees of membership of the input pattern `x=[xl, xu, x_cat]`
        corresponding to each hyperbox in the current list of existing
        hyperboxes.

    """
    if (xl is not None) and (x_cat is not None):
        b_numeric = np.array(membership_func_gfmm(xl, xu, V, W, g))
        b_cat = membership_cat_feature_eiol_gfmm(x_cat, D)

        b = alpha * b_numeric + (1 - alpha) * b_cat
    else:
        if xl is not None:
            b = np.array(membership_func_gfmm(xl, xu, V, W, g))
        else:
            b = membership_cat_feature_eiol_gfmm(x_cat, D)

    return b


def get_membership_extended_iol_gfmm_all_classes(Xl, Xu, X_cat, V, W, D, C, g=1, alpha=0.5):
    """
    Return membership values (according to the membership function of the GFMM
    classifiers) with respect to all class labels between the continuoues input
    patterns stored in two lower and upper bound input matrices `Xl` and `Xu`
    while categorical input patterns stored in the matrix `Xd` and
    existing hyperboxes represented by two matrices of minimum and maximum
    points `V` and `W` for continuous features and the matrix of categorical
    features `D` together with corresponding class labels in vector `C`.

    Parameters
    ----------
    Xl : array-like of shape (n_samples, n_continuous_features) or (n_continuous_features, )
        Lower bounds of input samples.
    Xu : array-like of shape (n_samples, n_continuous_features) or (n_continuous_features, )
        Upper bounds of input samples.
    X_cat : array-like of shape (n_samples, n_cat_features) or (n_cat_features, )
        Categorical bounds of input samples.
    V : array-like of shape (n_hyperboxes, n_continuous_features)
        Minimum points of the existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_continuous_features)
        Maximum points of the existing hyperboxes in the trained model.
    D : array-like of shape (n_hyperboxes, n_cat_features)
        Categorical bound of the existing hyperboxes in the trained model.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes corresponding to the values
        stored in `V`, `W`, and `D`.
    g : float or ndarray of shape (n_continuous_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each continuous dimension.
    alpha : float, optional, default=0.5
        The trade-off weighting factor between the impacts of categorical
        features and numerical features on the outputs of membership values.

    Returns
    -------
    mem_vals_matrix : array-like of shape (n_samples, n_classes)
        Membership values with regard to all class labels for each input
        sample. Each row is a vector of membership values. Each column
        represents an index of a class label sorted in an ascending order
        of class labels.
    hyperbox_ids_matrix : array-like of shape (n_samples, n_classes)
        Storing the indices of hyperboxes corresponding to membership values
        for classes.
    """
    if Xl is not None and Xl.ndim == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)
    if X_cat is not None and X_cat.ndim == 1:
        X_cat = X_cat.reshape(1, -1)

    if Xl is not None:
        n_samples = Xl.shape[0]
    else:
        n_samples = X_cat.shape[0]

    if V is not None:
        is_exist_missing_continous_value = (V > W).any()
    else:
        is_exist_missing_continous_value = False

    class_vals = np.unique(C)
    n_classes = len(class_vals)
    mem_vals_matrix = np.zeros((n_samples, n_classes), dtype=float)
    hyperbox_ids_matrix = np.zeros((n_samples, n_classes), dtype=int)
    # Get membership values for each sample
    for i in range(n_samples):
        # calculate memberships for all hyperboxes
        if Xl is not None and X_cat is not None:
            if is_exist_missing_continous_value == False:
                mem_vals = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], V, W, D, g, alpha)
            else:
                mem_vals = membership_func_extended_iol_gfmm(Xl[i], Xu[i], X_cat[i], np.minimum(V, W), np.maximum(W, V), D, g, alpha)
        elif Xl is not None:
            if is_exist_missing_continous_value == False:
                mem_vals = membership_func_extended_iol_gfmm(Xl[i], Xu[i], None, V, W, D, g, alpha)
            else:
                mem_vals = membership_func_extended_iol_gfmm(Xl[i], Xu[i], None, np.minimum(V, W), np.maximum(W, V), D, g, alpha)
        else:
            mem_vals = membership_func_extended_iol_gfmm(None, None, X_cat[i], V, W, D, g, alpha)
        class_c_mem = np.zeros(n_classes)
        class_c_hyperbox_id = np.zeros(n_classes)
        for _id, c in enumerate(class_vals):
            # Find all hyperboxes showing the same class as c
            id_c = np.nonzero(C == c)[0]
            id_c_max_mem = mem_vals[id_c].argmax()
            # Get maximum membership values among hyperboxes with the same
            # class as c
            class_c_mem[_id] = mem_vals[id_c[id_c_max_mem]]
            class_c_hyperbox_id[_id] = id_c[id_c_max_mem]

        mem_vals_matrix[i] = class_c_mem
        hyperbox_ids_matrix[i] = class_c_hyperbox_id

    return (mem_vals_matrix, hyperbox_ids_matrix)


def membership_func_free_range_gfmm(xl, xu, V, W, g=1):
    """Compute fuzzy membership values between an input pattern and a list of
    existing hyperboxes of a general fuzzy min-max neural network. This membership
    function does not require the coordinates located in the range of [0, 1].

    .. note::

        This function provides the degrees of membership b of an input pattern
        `x` (in form of upper bound `xu` and lower bound `xl`) with respect to
        the existing hyperboxes described by minimal points `V` and maximal
        points `W`. The sensitivity parameter `g` regulates how fast the
        membership values decrease when an input pattern is separeted from
        hyperbox core.

    Parameters
    ----------
    xl : array-like of shape (n_features,)
        Lower bound of an input pattern.
    xu : array-like of shape (n_features,)
        Upper bound of an input pattern.
    V : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all minimal points of all existing hyperboxes,
        in which each row is a minimal point of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features)
        A matrix stores all maximal points of all existing hyperboxes,
        in which each row is a maximal point of a hyperbox.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    b : array-like of shape (n_hyperboxes,)
        Degrees of membership of the input pattern X=[Xl, Xu] corresponding to
        each hyperbox in the current list of existing hyperboxes.

    """
    yW = W.shape[0]
    ones_mat = np.ones((yW, 1))
    viol_max = 1 - _ramp_func((ones_mat * xu - W) / (W - V + 1), 1/g)
    viol_min = 1 - _ramp_func((V - ones_mat * xl) / (W - V + 1), 1/g)

    b = np.minimum(viol_max, viol_min).min(axis=1)

    return b


def get_membership_free_range_gfmm_all_classes(Xl, Xu, V, W, C, g=1):
    """
    Return membership values (according to the membership function of the GFMM
    classifiers with unbounded range) with respect to all class labels between
    the input patterns stored in two lower and upper bound input matrices `Xl`
    and `Xu` and existing hyperboxes represented by two matrices of minimum and
    maximum points `V` and `W` together with corresponding class labels in
    vector `C`.

    Parameters
    ----------
    Xl : array-like of shape (n_samples, n_features) or (n_features, )
        Lower bounds of input samples.
    Xu : array-like of shape (n_samples, n_features) or (n_features, )
        Upper bounds of input samples.
    V : array-like of shape (n_hyperboxes, n_features)
        Minimum points of the existing hyperboxes in the trained model.
    W : array-like of shape (n_hyperboxes, n_features)
        Maximum points of the existing hyperboxes in the trained model.
    C : array-like of shape (n_hyperboxes,)
        Class labels of all existing hyperboxes corresponding to the values
        stored in `V` and `W`.
    g : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the
        membership function in each dimension.

    Returns
    -------
    mem_vals_matrix : array-like of shape (n_samples, n_classes)
        Membership values with regard to all class labels for each input
        sample. Each row is a vector of membership values. Each column
        represents an index of a class label sorted in an ascending order
        of class labels.
    hyperbox_ids_matrix : array-like of shape (n_samples, n_classes)
        Storing the indices of hyperboxes corresponding to membership values
        for classes.
    """
    if len(Xl.shape) == 1:
        Xl = Xl.reshape(1, -1)
        Xu = Xu.reshape(1, -1)

    n_samples = Xl.shape[0]
    class_vals = np.unique(C)
    n_classes = len(class_vals)
    mem_vals_matrix = np.zeros((n_samples, n_classes), dtype=float)
    hyperbox_ids_matrix = np.zeros((n_samples, n_classes), dtype=int)
    # Get membership values for each sample
    for i in range(n_samples):
        # calculate memberships for all hyperboxes
        mem_vals = membership_func_free_range_gfmm(Xl[i, :], Xu[i, :], V, W, g)
        class_c_mem = np.zeros(n_classes)
        class_c_hyperbox_id = np.zeros(n_classes)
        for _id, c in enumerate(class_vals):
            # Find all hyperboxes showing the same class as c
            id_c = np.nonzero(C == c)[0]
            id_c_max_mem = mem_vals[id_c].argmax()
            # Get maximum membership values among hyperboxes with the same
            # class as c
            class_c_mem[_id] = mem_vals[id_c[id_c_max_mem]]
            class_c_hyperbox_id[_id] = id_c[id_c_max_mem]

        mem_vals_matrix[i] = class_c_mem
        hyperbox_ids_matrix[i] = class_c_hyperbox_id

    return (mem_vals_matrix, hyperbox_ids_matrix)
