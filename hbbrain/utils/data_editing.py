"""
The :mod:`hbbrain.utils.data_editing` submodule implements various functions
for data editing using general fuzzy min-max neural network.

For more details regarding various data editing approaches, please
refer to the publication [1]_.

References
----------
.. [1] Gabrys, B. (2001). Data editing for neuro-fuzzy classifiers. In
       Proceedings of the Fourth International ICSC Symposia on Soft Computing
       and Intelligent Systems for Industry.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import clone
from hbbrain.utils.membership_calc import membership_func_gfmm
from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM


def data_editing_leave_one_out(X, y, gfmm_estimator=None, k_neighbors=5, n_iters=100, n_last_iters=5, seed=0):
    """
    Data editing procedure based on k-nearest neighbors and a leave-one-out
    scheme.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A matrix contains a data set on which a data editing method is
        performed to eliminate redundant samples.
    y : array-like of shape (n_samples,)
        Target vector relative to X.
    gfmm_estimator : object, default=None
        The general fuzzy min-max neural network estimator is used to fit on
        the input dataset. If None, then the base estimator is a
        :class:`~hbbrain.numerical_data.incremental_learner.onln_gfmm.OnlineGFMM`.
    k_neighbors : int, optional, default=5
        Number of nearest neighbours is used to make prediction for each
        validation sample.
    n_iters : int, optional, default=100
        Number of iterations is used to perform repeated leave-one-out
        validation during data editing procedure.
    n_last_iters : int, optional, default=5
        The consecutive number of iterations have resulted in no samples being
        eliminated from the training data set to terminate the algorithm.
    seed : None, int or instance of RandomState, optional, default=0
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    X_out : array-like of shape (n_samples, n_features)
        Data set after doing data editing.
    y_out : array-like of shape (n_samples,)
        Target vector relative to X_out after doing data editing.

    """
    if gfmm_estimator is None:
        gfmm_estimator = OnlineGFMM(0.1)

    X = np.array(X)
    random_state = check_random_state(seed)
    X_out = np.copy(X)
    y_out = np.copy(y)
    classes = np.unique(y)
    n_classes = len(classes)
    mapping_class_index = {}
    for i, val in enumerate(classes):
        mapping_class_index[val] = i

    last_iters = 0

    for t in range(n_iters):
        # Randomly shuffle the input data set
        n_samples = X_out.shape[0]
        index = np.arange(n_samples)
        random_state.shuffle(index)
        X_out = X_out[index]
        y_out = y_out[index]

        flag = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            # remove sample i from the input data set
            flag[i] = False
            X_tr = X_out[flag]
            y_tr = y_out[flag]
            x_val = X_out[i]
            y_val = y_out[i]

            # Fit a gfmm classifier
            classifier = clone(gfmm_estimator)
            classifier.fit(X_tr, y_tr)

            # Make prediction based on k-nearest neighbors rule with respect to
            # hyperboxes of a trained GFMM model
            mem_vals = membership_func_gfmm(x_val, x_val, classifier.V, classifier.W, classifier.gamma)
            # Find k_neighbors maximum values from a trained gfmm model with
            # respect to the validation sample
            id_sorted_mem = np.argsort(mem_vals)[::-1]
            votes = np.zeros(n_classes, dtype=int)
            cur_neighbors = k_neighbors if k_neighbors <= len(id_sorted_mem) else len(id_sorted_mem)
            for j in range(cur_neighbors):
                y_pred = classifier.C[id_sorted_mem[j]]
                votes[mapping_class_index[y_pred]] += 1

            y_pred_voting = classes.take(np.argmax(votes))

            if y_pred_voting == y_val:
                flag[i] = True

        X_out = X_out[flag]
        y_out = y_out[flag]

        if (flag == True).all():
            last_iters += 1
            if last_iters >= n_last_iters or len(flag) == 1:
                break
        else:
            last_iters = 0

    return (X_out, y_out)


def data_editing_two_fold_cv(X, y, gfmm_estimator=None, n_iters=100, n_last_iters=5, seed=0):
    """
    Data editing procedure based on repeated two fold cross-validation scheme.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A matrix contains a data set on which a data editing method is
        performed to eliminate redundant samples.
    y : array-like of shape (n_samples,)
        Target vector relative to X.
    gfmm_estimator : object, default=None
        The general fuzzy min-max neural network estimator is used to fit on
        the input dataset. If None, then the base estimator is a
        :class:`~hbbrain.numerical_data.incremental_learner.onln_gfmm.OnlineGFMM`.
    n_iters : int, optional, default=100
        Number of iterations is used to perform repeated two-fold
        cross-validation during data editing procedure.
    n_last_iters : int, optional, default=5
        The consecutive number of iterations have resulted in no samples being
        marked as misclassification from the training data set to terminate
        the algorithm.
    seed : None, int or instance of RandomState, optional, default=0
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    X_out : array-like of shape (n_samples, n_features)
        Data set after doing data editing.
    y_out : array-like of shape (n_samples,)
        Target vector relative to X_out after doing data editing.

    """
    if gfmm_estimator is None:
        gfmm_estimator = OnlineGFMM(0.1)

    X = np.array(X)
    random_state = check_random_state(seed)
    last_iters = 0
    n_samples = X.shape[0]
    flag = np.ones(n_samples, dtype=bool)

    last_iters = 0

    for t in range(n_iters):
        # Randomly shuffle the input data set
        index = np.arange(n_samples)
        random_state.shuffle(index)
        pivot = int(n_samples/2)
        index_1 = index[:pivot]
        X_tr_1 = X[index_1]
        y_tr_1 = y[index_1]
        index_2 = index[pivot:]
        X_tr_2 = X[index_2]
        y_tr_2 = y[index_2]

        is_misclassification = False
        gfmm_estimator_1 = clone(gfmm_estimator)
        gfmm_estimator_2 = clone(gfmm_estimator)
        # Training model on the first fold
        gfmm_estimator_1.fit(X_tr_1, y_tr_1)
        # Test the trained model on the second fold
        y_pred_2 = gfmm_estimator_1.predict(X_tr_2)
        for j in range(len(y_pred_2)):
            if y_pred_2[j] != y_tr_2[j]:
                flag[index_2[j]] = False
                is_misclassification = True

        # Training model on the second fold
        gfmm_estimator_2.fit(X_tr_2, y_tr_2)
        # Test the trained model on the first fold
        y_pred_1 = gfmm_estimator_2.predict(X_tr_1)
        for j in range(len(y_pred_1)):
            if y_pred_1[j] != y_tr_1[j]:
                flag[index_1[j]] = False
                is_misclassification = True

        if is_misclassification == False:
            # No sample is marked as misclassification
            last_iters += 1
            if last_iters >= n_last_iters:
                break

    X_out = X[flag]
    y_out = y[flag]

    return (X_out, y_out)


def data_editing_two_fold_cv_with_probability(X, y, gfmm_estimator=None, n_iters=100, min_remained_prob=0.5, seed=0):
    """
    Data editing procedure based on repeated two fold cross-validation scheme
    and a probability of every single point in the original training data set
    to be classified correctly during the multiple cross-validation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A matrix contains a data set on which a data editing method is
        performed to eliminate redundant samples.
    y : array-like of shape (n_samples,)
        Target vector relative to X.
    gfmm_estimator : object, default=None
        The general fuzzy min-max neural network estimator is used to fit on
        the input dataset. If None, then the base estimator is a
        :class:`~hbbrain.numerical_data.incremental_learner.onln_gfmm.OnlineGFMM`.
    n_iters : int, optional, default=100
        Number of iterations is used to perform repeated two-fold
        cross-validation during data editing procedure.
    min_remained_prob : float, optional, default=0.5
        Minimum probability value so that a sample is kept.
    seed : None, int or instance of RandomState, optional, default=0
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    X_out : array-like of shape (n_samples, n_features)
        Data set after doing data editing.
    y_out : array-like of shape (n_samples,)
        Target vector relative to X_out after doing data editing.

    """
    if min_remained_prob < 0 or min_remained_prob > 1:
        raise ValueError('The value of min_remained_prob must be in the range of [0, 1].')

    if gfmm_estimator is None:
        gfmm_estimator = OnlineGFMM(0.1)

    X = np.array(X)
    random_state = check_random_state(seed)
    n_samples = X.shape[0]
    n_times_kept = np.zeros(n_samples)

    for t in range(n_iters):
        # Randomly shuffle the input data set
        index = np.arange(n_samples)
        random_state.shuffle(index)
        pivot = int(n_samples/2)
        index_1 = index[:pivot]
        X_tr_1 = X[index_1]
        y_tr_1 = y[index_1]
        index_2 = index[pivot:]
        X_tr_2 = X[index_2]
        y_tr_2 = y[index_2]

        gfmm_estimator_1 = clone(gfmm_estimator)
        gfmm_estimator_2 = clone(gfmm_estimator)
        # Training model on the first fold
        gfmm_estimator_1.fit(X_tr_1, y_tr_1)
        # Test the trained model on the second fold
        y_pred_2 = gfmm_estimator_1.predict(X_tr_2)
        for j in range(len(y_pred_2)):
            if y_pred_2[j] == y_tr_2[j]:
                n_times_kept[index_2[j]] += 1

        # Training model on the second fold
        gfmm_estimator_2.fit(X_tr_2, y_tr_2)
        # Test the trained model on the first fold
        y_pred_1 = gfmm_estimator_2.predict(X_tr_1)
        for j in range(len(y_pred_1)):
            if y_pred_1[j] == y_tr_1[j]:
                n_times_kept[index_1[j]] += 1

    probs = n_times_kept / n_iters

    flag_remained_samples = probs >= min_remained_prob

    X_out = X[flag_remained_samples]
    y_out = y[flag_remained_samples]

    return (X_out, y_out)
