"""
The :mod:`hbbrain.utils.matrix_transformation` submodule implements various
functions for matrix transformation measures.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np


def split_matrix(A, asimil_type='max', is_sort=True):
    """
    Split an input matrix `A` into a maxtrix with three columns:

    - The first column contains the row indices of `A`
    - The second column contains the column indices of `A`
    - The third column contains the values corresponding to each row and column

    Parameters
    ----------
    A : ndarray of shape (n_samples, n_features)
        Input matrix needs to be split.
    asimil_type : str, optional, default='max'
        Use the minimum or maximum values of :math:`a_{ij}` or :math:`a_{ij}`
        for the third column if the matrix `A` is assymetric. Get a value of
        'max' or 'min'.
    is_sort : boolean, optional, default=True
        Sort the values of the third column in a descending order or not.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The outcome of the input matrix `A` after transformation.

    """
    # get min/max memberships from triu and tril of the memberhsip matrix which
    # might not be symmetric
    if asimil_type == 'min':
        # rotate tril to align it with triu for min (max) operation
        trans_A = np.minimum(np.flipud(np.rot90(np.tril(A, -1))), np.triu(A, 1))
    else:
        trans_A = np.maximum(np.flipud(np.rot90(np.tril(A, -1))), np.triu(A, 1))

    id_rows, id_columns = np.nonzero(trans_A)
    values = trans_A[id_rows, id_columns]

    if is_sort == True:
        id_sorted_trans_A = np.argsort(values)[::-1]
        sorted_trans_A = values[id_sorted_trans_A]
        X = np.concatenate((id_rows[id_sorted_trans_A][:, np.newaxis], id_columns[id_sorted_trans_A][:, np.newaxis], sorted_trans_A[:, np.newaxis]), axis=1)
    else:
        X = np.concatenate((id_rows[:, np.newaxis], id_columns[:, np.newaxis], values[:, np.newaxis]), axis=1)

    return X


def hashing(a, b):
    """
    Transform a pair of positive integer numbers into a unique number and
    this value is have commutation ability.

    Parameters
    ----------
    a : positive int
        The first value.
    b : positive int
        The second value.

    Returns
    -------
    c : postivie int
        A unique transformed value from the two input values.

    """
    c = max(a, b) * (max(a, b) + 1) / 2 + min(a, b)
    return c


def hashing_mat(A, B):
    """
    Transform each pair of items in two matrices A and B to a unique number

    Parameters
    ----------
    A : array-like of shape (n_samples, n_features)
        The first matrix.
    B : array-like of shape (n_samples, n_features)
        The second matrix.

    Returns
    -------
    C : array-like of shape (n_samples, n_features)
        A matrix that each element is a combination of two corresponding
        elements in two input matrices at the same position.

    """
    C = np.maximum(A, B) * (np.maximum(A, B) + 1) / 2 + np.minimum(A, B)
    return C
