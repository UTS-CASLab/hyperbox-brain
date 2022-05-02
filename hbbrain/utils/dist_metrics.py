"""
The :mod:`hbbrain.utils.dist_metrics` submodule implements various functions 
to compute distance-based metrics.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
from hbbrain.constants import EPSILON_MISSING_VAL


def manhattan_distance(X, Y):
    """Compute Manhattan distance between two points X and Y

    Parameters
    ----------
    X : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the coordinates of the first point.
    Y : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the coordinates of the second point.

    Returns
    -------
    d : float or ndarray of shape (n_samples,)
        A scalar value or a vector stores the resulting Manhattan distance values.

    """
    if X.ndim > 1:
        return (np.abs(X - Y)).sum(1)
    else:
        return (np.abs(X - Y)).sum()


def manhattan_distance_with_missing_val(X1, X2, Y1, Y2):
    """
    Compute Manhattan distance between the central points of X1, X2 and Y1, Y2.

    .. note::

        `X1`, `X2`, `Y1`, `Y2` can contain missing values. In that case,
        `X1j=1+EPSILON_MISSING_VAL > X2j=-EPSILON_MISSING_VAL` and
        `Y1j=1+EPSILON_MISSING_VAL > Y2j=-EPSILON_MISSING_VAL`. The Manhattan
        distance is only computed for the dimensions without missing values.

    Parameters
    ----------
    X1 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the lower bounds of the first point.
    X2 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the upper bounds of the first point.
    Y1 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the lower bounds of the second point.
    Y2 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the upper bounds of the second point.

    Returns
    -------
    result : ndarray of shape (n_samples,)
        A vector stores the resulting Manhattan distance values.

    """
    if X1.ndim == 1:
        n_samples = Y1.shape[0]
        one_sample = True
        id_non_missing_X1 = np.nonzero(X1 != 1 + EPSILON_MISSING_VAL)[0]
        id_non_missing_X2 = np.nonzero(X2 != -EPSILON_MISSING_VAL)[0]
        id_non_missing_1 = np.intersect1d(id_non_missing_X1, id_non_missing_X2)
    else:
        one_sample = False
        n_samples = X1.shape[0]
        
    result = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        if one_sample == False:
            id_non_missing_X1 = np.nonzero(X1[i] != 1 + EPSILON_MISSING_VAL)[0]
            id_non_missing_X2 = np.nonzero(X2[i] != -EPSILON_MISSING_VAL)[0]
            id_non_missing_1 = np.intersect1d(id_non_missing_X1, id_non_missing_X2)
            
        if n_samples > 1:
            id_non_missing_Y1 = np.nonzero(Y1[i] != 1 + EPSILON_MISSING_VAL)[0]
            id_non_missing_Y2 = np.nonzero(Y2[i] != -EPSILON_MISSING_VAL)[0]
            id_non_missing_2 = np.intersect1d(id_non_missing_Y1, id_non_missing_Y2)
            id_non_missing = np.intersect1d(id_non_missing_1, id_non_missing_2)
            
            if len(id_non_missing) > 0:
                Y_sel = (Y1[i, id_non_missing] + Y2[i, id_non_missing])/2
        else:
            id_non_missing_Y1 = np.nonzero(Y1 != 1 + EPSILON_MISSING_VAL)[0]
            id_non_missing_Y2 = np.nonzero(Y2 != -EPSILON_MISSING_VAL)[0]
            id_non_missing_2 = np.intersect1d(id_non_missing_Y1, id_non_missing_Y2)
            id_non_missing = np.intersect1d(id_non_missing_1, id_non_missing_2)
            
            if len(id_non_missing) > 0:
                Y_sel = (Y1[id_non_missing] + Y2[id_non_missing])/2
        
        if len(id_non_missing) > 0:
            if one_sample == True:
                X_sel = (X1[id_non_missing] + X2[id_non_missing]) / 2
            else:
                X_sel = (X1[i, id_non_missing] + X2[i, id_non_missing]) / 2
            result[i] = np.abs(X_sel - Y_sel).sum()
        else:
            result[i] = np.iinfo(np.int32).max
        
    return result


def rfmnn_distance(X, V, W):
    """
    Compute the distance from the input pattern to the list of existing hyperboxes 
    represented by minimum points `V` and maximum points `W`.

    Parameters
    ----------
    X : ndarray of shape (n_features,) or (n_hyperboxes, n_features)
        Vector or matrix contains the coordinates of the input pattern.
    V : ndarray of shape (n_hyperboxes, n_features)
        Lower bounds of all existing hyperboxes.
    W : ndarray of shape (n_hyperboxes, n_features)
        Upper bounds of all existing hyperboxes.

    Returns
    -------
    dist : ndarray of shape (n_hyperboxes,)
        The distance values from the input pattern to all existing hyperboxes.

    """
    if V.ndim > 1:
        n_samples, n_features = V.shape
    else:
        n_samples = 1
        n_features = len(V)
        
    if n_samples > 1 and X.ndim == 1:
        X = np.ones((n_samples, 1)) * X
        
    if V.ndim > 1:
        return (np.abs(X - V) + np.abs(X - W)).sum(1) / (2 * n_features)
    else:
        return (np.abs(X - V) + np.abs(X - W)).sum() / (2 * n_features)


def manhattan_distance_with_missing_val_free_range(X1, X2, Y1, Y2, MIN_RANGE, MAX_RANGE):
    """
    Compute Manhattan distance between the central points of X1, X2 and Y1, Y2.
    The coordinates are not limited by ranges.

    .. note::

        `X1`, `X2`, `Y1`, `Y2` can contain missing values. In that case,
        `X1j=MAX_RANGE > X2j=MIN_RANGE` and `Y1j=MAX_RANGE > Y2j=MIN_RANGE`.
        The Manhattan distance is only computed for the dimensions without
        missing values.

    Parameters
    ----------
    X1 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the lower bounds of the first point.
    X2 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the upper bounds of the first point.
    Y1 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the lower bounds of the second point.
    Y2 : ndarray of shape (n_features,) or (n_samples, n_features)
        Vector or matrix contains the upper bounds of the second point.
    MIN_RANGE : float
        The minimum value of floating numbers for missing features.
    MAX_RANGE : float
        The maximum values of floating numbers for missing features.

    Returns
    -------
    result : ndarray of shape (n_samples,)
        A vector stores the resulting Manhattan distance values.

    """
    if X1.ndim == 1:
        n_samples = Y1.shape[0]
        one_sample = True
        id_non_missing_X1 = np.nonzero(X1 != MAX_RANGE)[0]
        id_non_missing_X2 = np.nonzero(X2 != MIN_RANGE)[0]
        id_non_missing_1 = np.intersect1d(id_non_missing_X1, id_non_missing_X2)
    else:
        one_sample = False
        n_samples = X1.shape[0]
        
    result = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        if one_sample == False:
            id_non_missing_X1 = np.nonzero(X1[i] != MAX_RANGE)[0]
            id_non_missing_X2 = np.nonzero(X2[i] != MIN_RANGE)[0]
            id_non_missing_1 = np.intersect1d(id_non_missing_X1, id_non_missing_X2)
            
        if n_samples > 1:
            id_non_missing_Y1 = np.nonzero(Y1[i] != MAX_RANGE)[0]
            id_non_missing_Y2 = np.nonzero(Y2[i] != MIN_RANGE)[0]
            id_non_missing_2 = np.intersect1d(id_non_missing_Y1, id_non_missing_Y2)
            id_non_missing = np.intersect1d(id_non_missing_1, id_non_missing_2)
            
            if len(id_non_missing) > 0:
                Y_sel = (Y1[i, id_non_missing] + Y2[i, id_non_missing])/2
        else:
            id_non_missing_Y1 = np.nonzero(Y1 != MAX_RANGE)[0]
            id_non_missing_Y2 = np.nonzero(Y2 != MIN_RANGE)[0]
            id_non_missing_2 = np.intersect1d(id_non_missing_Y1, id_non_missing_Y2)
            id_non_missing = np.intersect1d(id_non_missing_1, id_non_missing_2)
            
            if len(id_non_missing) > 0:
                Y_sel = (Y1[id_non_missing] + Y2[id_non_missing])/2
        
        if len(id_non_missing) > 0:
            if one_sample == True:
                X_sel = (X1[id_non_missing] + X2[id_non_missing]) / 2
            else:
                X_sel = (X1[i, id_non_missing] + X2[i, id_non_missing]) / 2
            result[i] = np.abs(X_sel - Y_sel).sum()
        else:
            result[i] = np.iinfo(np.int32).max
        
    return result
