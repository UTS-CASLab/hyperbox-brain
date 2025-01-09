# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0


import numpy as np

from hbbrain.utils.matrix_transformation import (
    split_matrix,
    hashing,
    hashing_mat
)

sim_matrix = np.array([[1, 0.3, 0.4, 0.6, 0.1],
                       [0.2, 1, 0.3, 0.4, 0.6],
                       [0.5, 0.1, 1, 0.6, 0.1],
                       [0.6, 0.2, 0.3, 1, 0.7],
                       [0.2, 0.8, 0.4, 0.4, 1]])


def test_split_matrix_non_sort_min():
    split_max = split_matrix(sim_matrix, asimil_type='min', is_sort=False)
    expected_res = np.array([[0, 1, 0.2],
                             [0, 2, 0.4],
                             [0, 3, 0.6],
                             [0, 4, 0.1],
                             [1, 2, 0.1],
                             [1, 3, 0.2],
                             [1, 4, 0.6],
                             [2, 3, 0.3],
                             [2, 4, 0.1],
                             [3, 4, 0.4]])
    np.testing.assert_array_equal(split_max, expected_res)


def test_split_matrix_non_sort_max():
    split_max = split_matrix(sim_matrix, asimil_type='max', is_sort=False)
    expected_res = np.array([[0, 1, 0.3],
                             [0, 2, 0.5],
                             [0, 3, 0.6],
                             [0, 4, 0.2],
                             [1, 2, 0.3],
                             [1, 3, 0.4],
                             [1, 4, 0.8],
                             [2, 3, 0.6],
                             [2, 4, 0.4],
                             [3, 4, 0.7]])
    np.testing.assert_array_equal(split_max, expected_res)


def test_split_matrix_sorted_min():
    split_max = split_matrix(sim_matrix, asimil_type='min', is_sort=True)
    expected_res = np.array([[1 , 4 , 0.6], 
                             [0 , 3 , 0.6], 
                             [3 , 4 , 0.4], 
                             [0 , 2 , 0.4], 
                             [2 , 3 , 0.3], 
                             [1 , 3 , 0.2], 
                             [0 , 1 , 0.2], 
                             [2 , 4 , 0.1], 
                             [1 , 2 , 0.1], 
                             [0 , 4 , 0.1]])
    expected_res_1 = np.array([[0. , 3. , 0.6],
                             [1. , 4. , 0.6],
                             [0. , 2. , 0.4],
                             [3. , 4. , 0.4],
                             [2. , 3. , 0.3],
                             [0. , 1. , 0.2],
                             [1. , 3. , 0.2],
                             [2. , 4. , 0.1],
                             [1. , 2. , 0.1],
                             [0. , 4. , 0.1]])
    assert (np.array_equal(split_max, expected_res) or np.array_equal(split_max, expected_res_1))


def test_split_matrix_sorted_max():
    split_max = split_matrix(sim_matrix, asimil_type='max', is_sort=True)
    expected_res = np.array([[1 , 4 , 0.8], 
                             [3 , 4 , 0.7],
                             [2 , 3 , 0.6],
                             [0 , 3 , 0.6],
                             [0 , 2 , 0.5],
                             [2 , 4 , 0.4],
                             [1 , 3 , 0.4],
                             [1 , 2 , 0.3],
                             [0 , 1 , 0.3],
                             [0 , 4 , 0.2]])
    expected_res_1 = np.array([[1 , 4 , 0.8], 
                             [3 , 4 , 0.7],
                             [0 , 3 , 0.6],
                             [2 , 3 , 0.6],
                             [0 , 2 , 0.5],
                             [2 , 4 , 0.4],
                             [1 , 3 , 0.4],
                             [1 , 2 , 0.3],
                             [0 , 1 , 0.3],
                             [0 , 4 , 0.2]])
                             
    assert (np.array_equal(split_max, expected_res) or np.array_equal(split_max, expected_res_1))


def test_hashing():
    a = 10
    b = 20
    res = hashing(a, b)
    assert res == 220
    res_2 = hashing(b, a)
    assert res_2 == res


def test_hashing_mat():
    A = np.array([[10, 15],
                  [100, 200]])
    B = np.array([[20, 5],
                  [30, 50]])
    res = hashing_mat(A, B)
    expected_res = np.array([[220, 125],
                             [5080, 20150]])
    assert (res == expected_res).all()
    res_2 = hashing_mat(B, A)
    assert (res_2 == res).all()
