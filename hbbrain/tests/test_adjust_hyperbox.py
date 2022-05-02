# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
from hbbrain.utils.adjust_hyperbox import (
    is_overlap_one_many_hyperboxes_num_data_general,
    is_overlap_one_many_diff_label_hyperboxes_num_data_general,
    is_two_hyperboxes_overlap_num_data_general,
    overlap_resolving_num_data,
    hyperbox_overlap_test_fmnn,
    hyperbox_contraction_fmnn,
    hyperbox_overlap_test_efmnn,
    hyperbox_contraction_efmnn,
    is_overlap_diff_labels_num_data_rfmnn,
    hyperbox_contraction_rfmnn,
    is_overlap_cat_features_one_by_one,
    is_overlap_cat_features_one_vs_many,
    hyperbox_overlap_test_freq_cat_gfmm,
    hyperbox_contraction_freq_cat_gfmm,
    is_overlap_one_many_diff_label_hyperboxes_mixed_data_general
)


def test_is_overlap_one_many_hyperboxes_num_data_general():
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2],
                  [0.4, 0.5]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4],
                  [0.8, 0.7]])
    C = np.array([1, 1, 2])
    # test overlap of B1: expected Yes
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V, W, C, 0)
    assert is_overlap == True
    # test overlap of B2: expected No
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V, W, C, 1)
    assert is_overlap == False
    # test overlap of B3: expected Yes
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V, W, C, 2)
    assert is_overlap == True
    # Testing the fully included case
    C1 = np.array([1, 2, 2])
    # test overlap of B1: expected Yes
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V, W, C1, 0)
    assert is_overlap == True
    # test overlap of B2: expected Yes
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V, W, C1, 1)
    assert is_overlap == True
    # test overlap of B3: expected Yes
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V, W, C1, 2)
    assert is_overlap == True
    # test overlap in case of only one point overlapping
    V1 = np.array([[0.2, 0.1],
                  [0.3, 0.2],
                  [0.5, 0.6]])
    # test overlap of B3: expected Yes
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V1, W, C, 2)
    assert is_overlap == True
    # test no overlap
    V2 = np.array([[0.2, 0.1],
                  [0.3, 0.2],
                  [0.5001, 0.6]])
    # test overlap of B3: expected Noe
    is_overlap = is_overlap_one_many_hyperboxes_num_data_general(V2, W, C, 2)
    assert is_overlap == False


def test_is_overlap_one_many_diff_label_hyperboxes_num_data_general():
    # test no overlap case
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])

    V_cmp = np.array([0.5001, 0.6])
    W_cmp = np.array([0.8, 0.7])
    
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_num_data_general(V, W, V_cmp, W_cmp)
    assert is_overlap == False
    # test fully covering other boxes case
    V_cmp_1 = np.array([0.05, 0.06])
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_num_data_general(V, W, V_cmp_1, W_cmp)
    assert is_overlap == True
    # test fully covered by other boxes case
    V_cmp = np.array([0.21, 0.11])
    W_cmp = np.array([0.29, 0.19])
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_num_data_general(V, W, V_cmp, W_cmp)
    assert is_overlap == True
    # test the overlap on an edge
    V_cmp = np.array([0.5, 0.55])
    W_cmp = np.array([0.8, 0.9])
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_num_data_general(V, W, V_cmp, W_cmp)
    assert is_overlap == True
    # test the overlap by an area
    V_cmp = np.array([0.45, 0.5])
    W_cmp = np.array([0.7, 0.7])
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_num_data_general(V, W, V_cmp, W_cmp)
    assert is_overlap == True


def test_is_two_hyperboxes_overlap_num_data_general():
    # no overlap case
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.6, 0.3])
    W2 = np.array([0.7, 0.8])
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V1, W1, V2, W2)
    assert is_overlap == False
    # overlap by an area
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.3, 0.3])
    W2 = np.array([0.7, 0.8])
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V1, W1, V2, W2)
    assert is_overlap == True
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V2, W2, V1, W1)
    assert is_overlap == True
    # overlap by fully containing
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.3, 0.3])
    W2 = np.array([0.4, 0.4])
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V1, W1, V2, W2)
    assert is_overlap == True
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V2, W2, V1, W1)
    assert is_overlap == True
    # overlap by one line
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.5, 0.1])
    W2 = np.array([0.5, 0.8])
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V1, W1, V2, W2)
    assert is_overlap == True
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V2, W2, V1, W1)
    assert is_overlap == True
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.1, 0.1])
    W2 = np.array([0.2, 0.4])
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V1, W1, V2, W2)
    assert is_overlap == True
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V2, W2, V1, W1)
    assert is_overlap == True
    # overlap by one point
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.5, 0.6])
    W2 = np.array([0.7, 0.8])
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V1, W1, V2, W2)
    assert is_overlap == True
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V2, W2, V1, W1)
    assert is_overlap == True
    # fully overlap between 2 hyperboxes
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.2, 0.2])
    W2 = np.array([0.5, 0.6])
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V1, W1, V2, W2)
    assert is_overlap == True
    is_overlap = is_two_hyperboxes_overlap_num_data_general(V2, W2, V1, W1)
    assert is_overlap == True


def test_overlap_resolving_num_data_same_class():
    # two hyperboxes with the same class label
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.3, 0.3])
    W2 = np.array([0.7, 0.8])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 1)
    assert (V1 == V1_out).all()
    assert (W1 == W1_out).all()
    assert (V2 == V2_out).all()
    assert (W2 == W2_out).all()


def test_overlap_resolving_num_data_case_1():
    # test case 1 - contraction
    alpha = 0.0001
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.3, 0.3])
    W2 = np.array([0.7, 0.8])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    assert (V1 == V1_out).all()
    assert (W2 == W2_out).all()
    expected_W1_out = np.array([0.4, 0.6])
    expected_V2_out = np.array([0.4 + alpha, 0.3])
    assert (W1_out == expected_W1_out).all()
    assert (V2_out == expected_V2_out).all()


def test_overlap_resolving_num_data_case_2():
    # test case 2 - contraction
    alpha = 0.0001
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.3, 0.3])
    W2 = np.array([0.7, 0.8])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V2, W2, 1, V1, W1, 2, alpha)
    assert (V1 == V2_out).all()
    assert (W2 == W1_out).all()
    expected_V1_out = np.array([0.4 + alpha, 0.3])
    expected_W2_out = np.array([0.4, 0.6])
    assert (V1_out == expected_V1_out).all()
    assert (W2_out == expected_W2_out).all()


def test_overlap_resolving_num_data_case_3():
    # test case 31 - contraction
    alpha = 0.0001
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.3, 0.3])
    W2 = np.array([0.35, 0.4])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    expected_V1_out = np.array([0.35 + alpha, 0.2])
    assert (expected_V1_out == V1_out).all()
    assert (W1 == W1_out).all()
    assert (W2 == W2_out).all()
    assert (V2 == V2_out).all()
    # test case 32 - contraction
    V1 = np.array([0.2, 0.2])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.35, 0.3])
    W2 = np.array([0.4, 0.4])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    expected_W1_out = np.array([0.35 - alpha, 0.6])
    assert (expected_W1_out == W1_out).all()
    assert (V1 == V1_out).all()
    assert (W2 == W2_out).all()
    assert (V2 == V2_out).all()


def test_overlap_resolving_num_data_case_4():
    # test case 41 - contraction
    alpha = 0.0001
    V1 = np.array([0.3, 0.3])
    W1 = np.array([0.35, 0.4])
    V2 = np.array([0.2, 0.2])
    W2 = np.array([0.5, 0.6])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    expected_V2_out = np.array([0.35 + alpha, 0.2])
    assert (expected_V2_out == V2_out).all()
    assert (W2 == W2_out).all()
    assert (W1 == W1_out).all()
    assert (V1 == V1_out).all()
    # test case 42 - contraction
    V1 = np.array([0.35, 0.3])
    W1 = np.array([0.4, 0.4])
    V2 = np.array([0.2, 0.2])
    W2 = np.array([0.5, 0.6])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    expected_W2_out = np.array([0.35 - alpha, 0.6]) 
    assert (expected_W2_out == W2_out).all()
    assert (V2 == V2_out).all()
    assert (V1 == V1_out).all()
    assert (W1 == W1_out).all()


def test_overlap_resolving_num_data_case_5():
    # test case 5 - contraction
    alpha = 0.0001
    V1 = np.array([0.3, 0.3])
    W1 = np.array([0.3, 0.4])
    V2 = np.array([0.3, 0.35])
    W2 = np.array([0.6, 0.8])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    assert (W1 == W1_out).all()
    assert (V1 == V1_out).all()
    expected_V2_out = np.array([0.3 + alpha, 0.35])
    assert (expected_V2_out == V2_out).all()
    assert (W2 == W2_out).all()


def test_overlap_resolving_num_data_case_6():
    # test case 6 - contraction
    alpha = 0.0001
    V1 = np.array([0.2, 0.1])
    W1 = np.array([0.5, 0.6])
    V2 = np.array([0.5, 0.4])
    W2 = np.array([0.5, 0.8])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    assert (V1 == V1_out).all()
    expected_W1_out = np.array([0.5 - alpha, 0.6])
    assert (expected_W1_out == W1_out).all()
    assert (V2 == V2_out).all()
    assert (W2 == W2_out).all()


def test_overlap_resolving_num_data_case_7():
    # test case 7 - contraction
    alpha = 0.0001
    V1 = np.array([0.3, 0.35])
    W1 = np.array([0.6, 0.8])
    V2 = np.array([0.3, 0.3])
    W2 = np.array([0.3, 0.4])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    expected_V1_out = np.array([0.3 + alpha, 0.35])
    assert (expected_V1_out == V1_out).all()
    assert (W1 == W1_out).all()
    assert (V2 == V2_out).all()
    assert (W2 == W2_out).all()


def test_overlap_resolving_num_data_case_8():
    # test case 8 - contraction
    alpha = 0.0001
    V1 = np.array([0.5, 0.4])
    W1 = np.array([0.5, 0.8])
    V2 = np.array([0.2, 0.1])
    W2 = np.array([0.5, 0.6])
    V1_out, W1_out, V2_out, W2_out = overlap_resolving_num_data(V1, W1, 1, V2, W2, 2, alpha)
    assert (V1 == V1_out).all()
    assert (W1 == W1_out).all()
    expected_W2_out = np.array([0.5 - alpha, 0.6])
    assert (expected_W2_out == W2_out).all()
    assert (V2 == V2_out).all()


def test_hyperbox_overlap_test_fmnn():
    # no overlap case
    V = np.array([[0.2, 0.2],
                  [0.6, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim.size == 0
    # overlap by an area
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim[0] == 1
    assert dim[1] == 0
    dim = hyperbox_overlap_test_fmnn(V, W, 1, 0)
    assert dim[0] == 2
    assert dim[1] == 0
    # overlap by fully containing
    V = np.array([[0.2, 0.2],
                  [0.35, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim[0] == 32
    assert dim[1] == 0
    dim = hyperbox_overlap_test_fmnn(V, W, 1, 0)
    assert dim[0] == 41
    assert dim[1] == 0
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.35, 0.4]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim[0] == 31
    assert dim[1] == 0
    dim = hyperbox_overlap_test_fmnn(V, W, 1, 0)
    assert dim[0] == 42
    assert dim[1] == 0
    # overlap by one line
    V = np.array([[0.2, 0.2],
                  [0.5, 0.1]])
    W = np.array([[0.5, 0.6],
                  [0.5, 0.8]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim.size == 0
    dim = hyperbox_overlap_test_fmnn(V, W, 1, 0)
    assert dim[0] == 41
    assert dim[1] == 0
    V = np.array([[0.2, 0.2],
                  [0.1, 0.1]])
    W = np.array([[0.5, 0.6],
                  [0.2, 0.4]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim.size == 0
    dim = hyperbox_overlap_test_fmnn(V, W, 1, 0)
    assert dim.size == 0
    # overlap by one point
    V = np.array([[0.2, 0.2],
                  [0.5, 0.6]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim.size == 0
    dim = hyperbox_overlap_test_fmnn(V, W, 1, 0)
    assert dim.size == 0
    # fully overlap between 2 hyperboxes
    V = np.array([[0.2, 0.2],
                  [0.2, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.5, 0.6]])
    dim = hyperbox_overlap_test_fmnn(V, W, 0, 1)
    assert dim[0] == 1
    assert dim[1] == 0
    dim = hyperbox_overlap_test_fmnn(V, W, 1, 0)
    assert dim[0] == 1
    assert dim[1] == 0


def test_hyperbox_contraction_fmnn_case_1():
    alpha = 0.0001
    # Case 1 - Contraction
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    V_out, W_out = hyperbox_contraction_fmnn(V, W, np.array([1, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.4 + alpha, 0.3]])
    expected_W_out = np.array([[0.4, 0.6],
                               [0.7, 0.8]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_fmnn_case_2():
    alpha = 0.0001
    # Case 2 - Contraction
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    V_out, W_out = hyperbox_contraction_fmnn(V, W, np.array([2, 0]), 1, 0, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.4, 0.3]])
    expected_W_out = np.array([[0.4 - alpha, 0.6],
                               [0.7, 0.8]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_fmnn_case_3():
    alpha = 0.0001
    # Case 31 - Contraction
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.35, 0.4]])
    V_out, W_out = hyperbox_contraction_fmnn(V, W, np.array([31, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.35 + alpha, 0.2],
                               [0.3, 0.3]])
    expected_W_out = np.array([[0.5, 0.6],
                               [0.35, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()
    # Case 32 - Contraction
    V = np.array([[0.2, 0.2],
                  [0.35, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    V_out, W_out = hyperbox_contraction_fmnn(V, W, np.array([32, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.35, 0.3]])
    expected_W_out = np.array([[0.35 - alpha, 0.6],
                               [0.4, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_fmnn_case_4():
    alpha = 0.0001
    # Case 41 - Contraction
    V = np.array([[0.2, 0.2],
                  [0.35, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    V_out, W_out = hyperbox_contraction_fmnn(V, W, np.array([41, 0]), 1, 0, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.35, 0.3]])
    expected_W_out = np.array([[0.35 - alpha, 0.6],
                               [0.4, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()
    # Case 42 - Contraction
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.35, 0.4]])
    V_out, W_out = hyperbox_contraction_fmnn(V, W, np.array([42, 0]), 1, 0, alpha)
    expected_V_out = np.array([[0.35+ alpha, 0.2],
                               [0.3, 0.3]])
    expected_W_out = np.array([[0.5, 0.6],
                               [0.35, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_overlap_test_efmnn():
    # no overlap case
    V = np.array([[0.2, 0.2],
                  [0.6, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.5, 0.6]))
    assert dim.size == 0
    # case 1 and 2
    # overlap by an area
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.5, 0.6]))
    assert dim[0] == 1
    assert dim[1] == 0
    dim = hyperbox_overlap_test_efmnn(V, W, 1, 0, np.array([0.7, 0.8]))
    assert dim[0] == 2
    assert dim[1] == 0
    # case 3
    V = np.array([[0.2, 0.2],
                  [0.2, 0.2]])
    W = np.array([[0.7, 0.6],
                  [0.8, 0.7]])
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.7, 0.6]))
    assert dim[0] == 3
    assert dim[1] == 1
    # case 4
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.5, 1]])
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.5, 0.6]))
    assert dim[0] == 4
    assert dim[1] == 0
    # case 5
    V = np.array([[0.2, 0.2],
                  [0.2, 0.2]])
    W = np.array([[0.8, 0.7],
                  [0.7, 0.6]])
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.8, 0.7]))
    assert dim[0] == 5
    assert dim[1] == 1
    # case 6
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.5, 1]])
    dim = hyperbox_overlap_test_efmnn(V, W, 1, 0, np.array([0.5, 1]))
    assert dim[0] == 6
    assert dim[1] == 0
    # case 72 and 81
    # overlap by fully containing
    V = np.array([[0.2, 0.2],
                  [0.35, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.5, 0.6]))
    assert dim[0] == 72
    assert dim[1] == 0
    dim = hyperbox_overlap_test_efmnn(V, W, 1, 0, np.array([0.4, 0.4]))
    assert dim[0] == 81
    assert dim[1] == 0
    # Case 71 and 82
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.35, 0.4]])
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.5, 0.6]))
    assert dim[0] == 71
    assert dim[1] == 0
    dim = hyperbox_overlap_test_efmnn(V, W, 1, 0, np.array([0.35, 0.4]))
    assert dim[0] == 82
    assert dim[1] == 0
    # Case 9
    V = np.array([[0.2, 0.2],
                  [0.2, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.5, 1]])
    # Case 91
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.5, 0.6]))
    assert dim[0] == 91
    assert dim[1] == 0
    # case 92
    dim = hyperbox_overlap_test_efmnn(V, W, 0, 1, np.array([0.2, 0.2]))
    assert dim[0] == 92
    assert dim[1] == 0


def test_hyperbox_contraction_efmnn_case_1():
    alpha = 0.0001
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([1, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.4 + alpha, 0.3]])
    expected_W_out = np.array([[0.4, 0.6],
                               [0.7, 0.8]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_2():
    alpha = 0.0001
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.7, 0.8]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([2, 0]), 1, 0, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.4, 0.3]])
    expected_W_out = np.array([[0.4 - alpha, 0.6],
                               [0.7, 0.8]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_3():
    alpha = 0.0001
    V = np.array([[0.2, 0.2],
                  [0.2, 0.2]])
    W = np.array([[0.7, 0.6],
                  [0.8, 0.7]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([3, 1]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.2, 0.6 + alpha]])
    expected_W_out = np.array([[0.7, 0.6],
                               [0.8, 0.7]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_4():
    alpha = 0.0001
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.5, 1]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([4, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.3, 0.3]])
    expected_W_out = np.array([[0.3 - alpha, 0.6],
                               [0.5, 1]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_5():
    alpha = 0.0001
    V = np.array([[0.2, 0.2],
                  [0.2, 0.2]])
    W = np.array([[0.8, 0.7],
                  [0.7, 0.6]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([5, 1]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.6 + alpha],
                               [0.2, 0.2]])
    expected_W_out = np.array([[0.8, 0.7],
                               [0.7, 0.6]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_6():
    alpha = 0.0001
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.5, 1]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([6, 0]), 1, 0, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.3, 0.3]])
    expected_W_out = np.array([[0.3 - alpha, 0.6],
                               [0.5, 1]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_7():
    alpha = 0.0001
    # Case 71
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.35, 0.4]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([71, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.35 + alpha, 0.2],
                               [0.3, 0.3]])
    expected_W_out = np.array([[0.5, 0.6],
                               [0.35, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()
    # Case 72
    V = np.array([[0.2, 0.2],
                  [0.35, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([72, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.35, 0.3]])
    expected_W_out = np.array([[0.35 - alpha, 0.6],
                               [0.4, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_8():
    alpha = 0.0001
    # Case 81
    V = np.array([[0.2, 0.2],
                  [0.35, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([81, 0]), 1, 0, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.35, 0.3]])
    expected_W_out = np.array([[0.35 - alpha, 0.6],
                               [0.4, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()
    # case 82
    V = np.array([[0.2, 0.2],
                  [0.3, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.35, 0.4]])
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([82, 0]), 1, 0, alpha)
    expected_V_out = np.array([[0.35 + alpha, 0.2],
                               [0.3, 0.3]])
    expected_W_out = np.array([[0.5, 0.6],
                               [0.35, 0.4]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_hyperbox_contraction_efmnn_case_9():
    alpha = 0.0001
    V = np.array([[0.2, 0.2],
                  [0.2, 0.3]])
    W = np.array([[0.5, 0.6],
                  [0.5, 1]])
    # Case 91
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([91, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.2, 0.2],
                               [0.35 + alpha, 0.3]])
    expected_W_out = np.array([[0.35, 0.6],
                               [0.5, 1]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()
    # Case 92
    V_out, W_out = hyperbox_contraction_efmnn(V, W, np.array([92, 0]), 0, 1, alpha)
    expected_V_out = np.array([[0.35, 0.2],
                               [0.2, 0.3]])
    expected_W_out = np.array([[0.5, 0.6],
                               [0.35 - alpha, 1]])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()


def test_is_overlap_diff_labels_num_data_rfmnn_check_overlap_only():
    find_dim_min_overlap = False
    # test no overlap case
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])

    V_cmp = np.array([0.5001, 0.6])
    W_cmp = np.array([0.8, 0.7])
    
    is_overlap = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp, find_dim_min_overlap)
    assert is_overlap == False
    # test fully covering other boxes case
    V_cmp_1 = np.array([0.05, 0.06])
    is_overlap = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp_1, W_cmp, find_dim_min_overlap)
    assert is_overlap == True
    # test fully covered by other boxes case
    V_cmp = np.array([0.21, 0.11])
    W_cmp = np.array([0.29, 0.19])
    is_overlap = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp, find_dim_min_overlap)
    assert is_overlap == True
    # test the overlap on an edge
    V_cmp = np.array([0.5, 0.55])
    W_cmp = np.array([0.8, 0.9])
    is_overlap = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp, find_dim_min_overlap)
    assert is_overlap == True


def test_is_overlap_diff_labels_num_data_rfmnn_find_overlap_dim():
    # test no overlap case
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])

    V_cmp = np.array([0.5001, 0.6])
    W_cmp = np.array([0.8, 0.7])
    
    dim = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp)
    assert dim[0] == False
    assert dim[1] is None
    assert dim[2] is None
    # test fully covering other boxes case
    V_cmp_1 = np.array([0.05, 0.06])
    dim = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp_1, W_cmp)
    assert dim[0] == True
    assert (dim[1] == np.array([0, 1])).all()
    assert (dim[2] == np.array([0, 1])).all()
    # test fully covered by other boxes case
    V_cmp = np.array([0.21, 0.11])
    W_cmp = np.array([0.29, 0.19])
    dim = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp)
    assert dim[0] == True
    assert (dim[1] == np.array([0])).all()
    assert (dim[2] == np.array([0])).all()
    # test the overlap on an edge
    V_cmp = np.array([0.5, 0.55])
    W_cmp = np.array([0.8, 0.9])
    dim = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp)
    assert dim[0] == True
    assert (dim[1] == np.array([0])).all()
    assert (dim[2] == np.array([1])).all()
    # test the overlap by an area
    V_cmp = np.array([0.45, 0.5])
    W_cmp = np.array([0.7, 0.7])
    dim = is_overlap_diff_labels_num_data_rfmnn(V, W, V_cmp, W_cmp)
    assert dim[0] == True
    assert (dim[1] == np.array([0])).all()
    assert (dim[2] == np.array([0])).all()


def test_hyperbox_contraction_rfmnn_many_parent_hyperboxes():
    scale = 0.001
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2],
                  [0.32, 0.25]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4],
                  [0.32, 0.25]])
    # test two parent hyperboxes include one input
    V_out, W_out, C_out = hyperbox_contraction_rfmnn(V, W, np.array([1, 1, 2]), np.array([0, 1]), 2, np.array([0, 0]), scale)
    expected_V_out = np.array([[0.2, 0.1],
                               [0.3, 0.2],
                               [0.32, 0.25],
                               [0.32 + scale, 0.1],
                               [0.32 + scale, 0.2]])
    expected_W_out = np.array([[0.32 - scale, 0.6],
                               [0.32 - scale, 0.4],
                               [0.32, 0.25],
                               [0.5, 0.6],
                               [0.4, 0.4]])
    expected_C_out = np.array([1, 1, 2, 1, 1])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()
    assert (C_out == expected_C_out).all()


def test_hyperbox_contraction_rfmnn_one_parent_hyperboxes():
    scale = 0.001
    # test only one parent hyperbox includes one input
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2],
                  [0.25, 0.15]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4],
                  [0.25, 0.15]])
    V_out, W_out, C_out = hyperbox_contraction_rfmnn(V, W, np.array([1, 1, 2]), np.array([0]), 2, np.array([0]), scale)
    expected_V_out = np.array([[0.2, 0.1],
                               [0.3, 0.2],
                               [0.25, 0.15],
                               [0.25 + scale, 0.1]])
    expected_W_out = np.array([[0.25 - scale, 0.6],
                               [0.4, 0.4],
                               [0.25, 0.15],
                               [0.5, 0.6]])
    expected_C_out = np.array([1, 1, 2, 1])
    assert (V_out == expected_V_out).all()
    assert (W_out == expected_W_out).all()
    assert (C_out == expected_C_out).all()


def test_is_overlap_cat_features_one_by_one_no_overlap():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E2 = np.array([2, 3])
    F2 = np.array([2, 6])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == False


def test_is_overlap_cat_features_one_by_one_default_val():
    DEFAULT_CATEGORICAL_VALUE = 100000
    E1 = np.array([1, 2])
    F1 = np.array([2, DEFAULT_CATEGORICAL_VALUE])
    E2 = np.array([2, 3])
    F2 = np.array([2, DEFAULT_CATEGORICAL_VALUE])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == False


def test_is_overlap_cat_features_one_by_one_same_one_lower():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E2 = np.array([1, 3])
    F2 = np.array([2, 6])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == False


def test_is_overlap_cat_features_one_by_one_same_lower():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E2 = np.array([1, 2])
    F2 = np.array([2, 6])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == True


def test_is_overlap_cat_features_one_by_one_same_lower1_upper2_one_dim():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E2 = np.array([2, 3])
    F2 = np.array([1, 4])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == False


def test_is_overlap_cat_features_one_by_one_same_lower1_upper2():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E2 = np.array([2, 3])
    F2 = np.array([1, 2])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == True
    

def test_is_overlap_cat_features_one_by_one_same_lower2_upper1():
    E1 = np.array([1, 2])
    F1 = np.array([2, 3])
    E2 = np.array([2, 3])
    F2 = np.array([1, 8])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == True
    
    
def test_is_overlap_cat_features_one_by_one_same_upper1_upper2():
    E1 = np.array([1, 2])
    F1 = np.array([4, 8])
    E2 = np.array([2, 3])
    F2 = np.array([4, 8])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == True
    

def test_is_overlap_cat_features_one_by_one_same_lower_upper():
    E1 = np.array([1, 2])
    F1 = np.array([4, 3])
    E2 = np.array([2, 4])
    F2 = np.array([4, 2])
    is_overlap = is_overlap_cat_features_one_by_one(E1, F1, E2, F2)
    assert is_overlap == True


def test_is_overlap_cat_features_one_vs_many_no_overlap():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E = np.array([[2, 3],
                  [5, 6]])
    F = np.array([[2, 6],
                  [3, 7]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == False


def test_is_overlap_cat_features_one_vs_many_no_overlap_default_val():
    DEFAULT_CATEGORICAL_VALUE = 100000
    E1 = np.array([1, 2])
    F1 = np.array([2, DEFAULT_CATEGORICAL_VALUE])
    E = np.array([[2, 3],
                  [5, 6]])
    F = np.array([[2, DEFAULT_CATEGORICAL_VALUE],
                  [2, DEFAULT_CATEGORICAL_VALUE]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == False


def test_is_overlap_cat_features_one_vs_many_same_one_lower():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E = np.array([[1, 3],
                  [1, 7]])
    F = np.array([[2, 6],
                  [3, 7]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == False


def test_is_overlap_cat_features_one_vs_many_same_lower():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E = np.array([[1, 2],
                  [2, 7]])
    F = np.array([[2, 6],
                  [2, 8]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == True


def test_is_overlap_cat_features_one_vs_many_same_one_lower_same_one_lower_upper():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E = np.array([[1, 5],
                  [2, 7]])
    F = np.array([[0, 0],
                  [2, 8]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == True
    

def test_is_overlap_cat_features_one_vs_many_same_lower1_upper2_one_dim():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E = np.array([[2, 3],
                  [8, 9]])
    F = np.array([[1, 4],
                  [5, 7]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == False


def test_is_overlap_cat_features_one_vs_many_same_lower1_upper2():
    E1 = np.array([1, 2])
    F1 = np.array([2, 5])
    E = np.array([[2, 4],
                  [2, 6]])
    F = np.array([[1, 2],
                  [4, 7]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == True


def test_is_overlap_cat_features_one_vs_many_same_lower2_upper1():
    E1 = np.array([1, 2])
    F1 = np.array([2, 3])
    E = np.array([[2, 3],
                  [7, 8]])
    F = np.array([[1, 8],
                  [4, 5]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == True


def test_is_overlap_cat_features_one_vs_many_same_upper1_upper2():
    E1 = np.array([1, 2])
    F1 = np.array([4, 8])
    E = np.array([[2, 3],
                  [3, 4]])
    F = np.array([[4, 8],
                  [4, 1]])
    is_overlap = is_overlap_cat_features_one_vs_many(E1, F1, E, F, np.array([0, 1]))
    assert is_overlap == True


def test_hyperbox_overlap_test_freq_cat_gfmm_no_overlap():
    similarity_of_cat_vals = np.array([{5000050001.0: 0.0, 2.0: 0.0, 4.0: 1.0, 7.0: 0.625, 5000050002.0: 0.0, 5.0: 0.0, 8.0: 0.375, 5000050003.0: 0.0, 9.0: 0.0},
                                       {5000050001.0: 0.0, 2.0: 0.0, 4.0: 0.888888888888889, 7.0: 1.0, 5000050002.0: 0.0, 5.0: 0.0, 8.0: 0.11111111111111113, 5000050003.0: 0.0, 9.0: 0.0}], dtype=object)
    E = np.array([[1, 2],
                  [2, 3]])
    F = np.array([[2, 0],
                  [2, 1]])
    X_cat = np.array([[1, 2],
                      [2, 3],
                      [2, 0],
                      [1, 1],
                      [2, 3]])
    dim = hyperbox_overlap_test_freq_cat_gfmm(E, F, 0, 1, X_cat, similarity_of_cat_vals, np.array([1]))
    assert len(dim) == 0


def test_hyperbox_overlap_test_freq_cat_gfmm_overlap():
    similarity_of_cat_vals = np.array([{2.0: 0.0, 4.0: 1.0, 7.0: 0.625, 5000050002.0: 0.0, 5.0: 0.0, 8.0: 0.375, 5000050003.0: 0.0, 9.0: 0.0},
                                       {5000050001.0: 0.0, 0.0: 0, 1.0:0.3, 2.0: 0.0, 3.0:0.4, 4.0: 0.888888888888889, 5.0:0.1, 6.0:0.2, 7.0: 1.0, 8.0: 0.11111111111111113, 9.0: 0.3}], dtype=object)
    E = np.array([[1, 1],
                  [2, 3]])
    F = np.array([[2, 0],
                  [2, 1]])
    X_cat = np.array([[1, 2],
                      [2, 3],
                      [2, 0],
                      [1, 1],
                      [2, 3]])
    dim = hyperbox_overlap_test_freq_cat_gfmm(E, F, 0, 1, X_cat, similarity_of_cat_vals, np.array([1]))
    assert len(dim) == 2
    assert dim[0] == 1
    assert dim[1][0] == 0
    assert dim[1][1] is None


def test_hyperbox_contraction_freq_cat_gfmm():
    E = np.array([2, 3])
    F = np.array([2, 1])
    case_contraction = [1, [0, None]]
    E_out, F_out = hyperbox_contraction_freq_cat_gfmm(E, F, case_contraction)
    expected_E_out = np.array([2, 0])
    expected_F_out = np.array([2, 1])
    assert (E_out == expected_E_out).all()
    assert (F_out == expected_F_out).all()


def test_is_overlap_one_many_diff_label_hyperboxes_mixed_data_general_no_overlap():
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    D = np.array([[{1: 5, 2:3}, {0:4, 3:4}],
                  [{1: 3, 2:9}, {0:7, 3:5}]], dtype=object)
    N_samples = np.array([8, 12])
    V_cmp = np.array([0.51, 0.2])
    W_cmp = np.array([0.7, 0.8])
    D_cmp = np.array([{1:3, 2:1}, {0:2, 3:2}])
    N_samples_cmp = 4
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_mixed_data_general(V, W, D, N_samples, V_cmp, W_cmp, D_cmp, N_samples_cmp)
    assert is_overlap == False
    

def test_is_overlap_one_many_diff_label_hyperboxes_mixed_data_general_overlap_con_features_only():
    # Overlap on continuous features only => expected False
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    D = np.array([[{1: 5, 2:3}, {0:4, 3:4}],
                  [{1: 3, 2:9}, {0:7, 3:5}]], dtype=object)
    N_samples = np.array([8, 12])
    V_cmp = np.array([0.4, 0.2])
    W_cmp = np.array([0.7, 0.8])
    D_cmp = np.array([{0:3, 3:1}, {1:2, 2:2}])
    N_samples_cmp = 4
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_mixed_data_general(V, W, D, N_samples, V_cmp, W_cmp, D_cmp, N_samples_cmp)
    assert is_overlap == False


def test_is_overlap_one_many_diff_label_hyperboxes_mixed_data_general_overlap_cat_features_only():
    # Overlap on categorical features only => expected False
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    D = np.array([[{1: 5, 2:3}, {0:4, 3:4}],
                  [{1: 3, 2:9}, {0:7, 3:5}]], dtype=object)
    N_samples = np.array([8, 12])
    V_cmp = np.array([0.51, 0.2])
    W_cmp = np.array([0.7, 0.8])
    D_cmp = np.array([{0:5, 2:3}, {1:4, 3:4}])
    N_samples_cmp = 8
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_mixed_data_general(V, W, D, N_samples, V_cmp, W_cmp, D_cmp, N_samples_cmp)
    assert is_overlap == False
    
def test_is_overlap_one_many_diff_label_hyperboxes_mixed_data_general_overlap_con_and_cat_features():
    # Overlap on both continuous and categorical features => expected True
    V = np.array([[0.2, 0.1],
                  [0.3, 0.2]])
    W = np.array([[0.5, 0.6],
                  [0.4, 0.4]])
    D = np.array([[{1: 5, 2:3}, {0:4, 3:4}],
                  [{1: 3, 2:9}, {0:7, 3:5}]], dtype=object)
    N_samples = np.array([8, 12])
    V_cmp = np.array([0.4, 0.2])
    W_cmp = np.array([0.7, 0.8])
    D_cmp = np.array([{0:10, 2:6}, {1:8, 3:8}])
    N_samples_cmp = 16
    is_overlap = is_overlap_one_many_diff_label_hyperboxes_mixed_data_general(V, W, D, N_samples, V_cmp, W_cmp, D_cmp, N_samples_cmp)
    assert is_overlap == True
    