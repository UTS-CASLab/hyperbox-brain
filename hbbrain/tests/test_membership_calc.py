# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0


import numpy as np

from hbbrain.utils.membership_calc import (
    membership_func_gfmm,
    get_membership_gfmm_all_classes,
    membership_func_fmnn,
    get_membership_fmnn_all_classes,
    asym_similarity_val_one_many_hyperboxes,
    membership_func_onehot_gfmm,
    get_membership_onehot_gfmm_all_classes,
    membership_func_freq_cat_gfmm,
    get_membership_freq_cat_gfmm_all_classes,
    membership_cat_feature_eiol_gfmm,
    membership_func_extended_iol_gfmm,
    get_membership_extended_iol_gfmm_all_classes
)
from hbbrain.utils.matrix_transformation import hashing
# Initialise data to perform test functions
# Input training data
X = np.array([[0.4, 0.3],
              [0.1, 0.2]])
Xd_one_hot = np.array([[np.array([True, False]), np.array([True, False, True])],
                       [np.array([False, True]), np.array([False, True, False])]], dtype=object)
# training classes
y = np.array([1, 2, 1, 2])

# Init existing hyperboxes
V = np.array([[0.2, 0.2],
              [0.45, 0.3]])
W = np.array([[0.45, 0.5],
              [0.6, 0.6]])
D_onehot = np.array([[np.array([False, True]), np.array([True, True, True])],
                     [np.array([True, True]), np.array([False, True, True])]], dtype=object)

X_cat = np.array([[3, 1],
                  [1, 2]])

E = np.array([[2, 1],
              [3, 2]])

F = np.array([[2, 2],
              [2, 1]])

C = np.array([1, 2])

similarity_of_cat_vals = np.array([{5000050001.0: 0.0, 2.0: 0.0, 4.0: 1.0, 7.0: 0.625, 5000050002.0: 0.0, 5.0: 0.0, 8.0: 0.375, 5000050003.0: 0.0, 9.0: 0.0},
                                   {5000050001.0: 0.0, 2.0: 0.0, 4.0: 0.888888888888889, 7.0: 1.0, 5000050002.0: 0.0, 5.0: 0.0, 8.0: 0.11111111111111113, 5000050003.0: 0.0, 9.0: 0.0}], dtype=object)

# Data for EIOL-GFMM
D_eiol = np.array([[{1:5, 2:2, 3:1}, {1:3, 2:4, 3:1}], [{1:2, 2:6, 3:8}, {1:4, 2:7, 3:5}]], dtype=object)

def test_membership_func_gfmm():
    # test the case of including an input sample in a hyperbox
    c1 = membership_func_gfmm(X[0], X[0], V, W, 1)
    assert c1[0] == 1
    assert c1[1] == 0.95
    # test the case of not including an input sample in a hyperbox
    c2 = membership_func_gfmm(X[1], X[1], V, W, 1)
    assert c2[0] == 0.9
    assert c2[1] == 0.65


def test_get_membership_gfmm_all_classes():
    vals = get_membership_gfmm_all_classes(X, X, V, W, C)
    assert (vals[0] == np.array([[1, 0.95], [0.9, 0.65]])).all()
    assert (vals[1] == np.array([[0, 1], [0, 1]])).all()


def test_membership_func_fmnn():
    # test the case of including an input sample in a hyperbox
    c1 = membership_func_fmnn(X[0], V, W, 1)
    assert c1[0] == 1
    assert c1[1] == 0.9875
    # test the case of not including an input sample in a hyperbox
    c2 = membership_func_fmnn(X[1], V, W, 1)
    assert c2[0] == 0.975
    assert c2[1] == 0.8875


def test_get_membership_fmnn_all_classes():
    vals = get_membership_fmnn_all_classes(X, V, W, C)
    assert (vals[0] == np.array([[1, 0.9875], [0.975, 0.8875]])).all()
    assert (vals[1] == np.array([[0, 1], [0, 1]])).all()


def test_asym_similarity_val_one_many_hyperboxes():
    val1 = asym_similarity_val_one_many_hyperboxes(X[0], X[0], V, W, 1, 'max')
    val2 = asym_similarity_val_one_many_hyperboxes(X[0], X[0], V, W, 1, 'min')
    assert (val1 != val2).all()
    assert (val1 == np.array([1, 0.95])).all()
    assert (val2 == np.array([0.8, 0.7])).all()


def test_membership_func_onehot_gfmm():
    # no categorical features
    mem = membership_func_onehot_gfmm(X[0], X[0], None, V, W, None, 1)
    assert mem[0] == 1
    assert mem[1] == 0.95
    # test no continuous features
    mem = membership_func_onehot_gfmm(None, None, Xd_one_hot[0], None, None, D_onehot)
    assert mem[0] == 0.5
    assert mem[1] == 1
    # test the input pattern including both categorical and continuous features
    mem = membership_func_onehot_gfmm(X[0], X[0], Xd_one_hot[0], V, W, D_onehot)
    assert mem[0] == 0.75
    assert mem[1] == 0.975


def test_get_membership_onehot_gfmm_all_classes():
    # no categorical features
    mem = get_membership_onehot_gfmm_all_classes(X[0], X[0], None, V, W, None, C, 1)
    assert (mem[0] == np.array([[1, 0.95]])).all()
    assert (mem[1] == np.array([[0, 1]])).all()
    # test no continuous features
    mem = get_membership_onehot_gfmm_all_classes(None, None, Xd_one_hot[0], None, None, D_onehot, C)
    assert (mem[0] == np.array([[0.5, 1]])).all()
    assert (mem[1] == np.array([[0, 1]])).all()
    # test the input pattern including both categorical and continuous features
    mem = get_membership_onehot_gfmm_all_classes(X[0], X[0], Xd_one_hot[0], V, W, D_onehot, C)
    assert (mem[0] == np.array([[0.75, 0.975]])).all()
    assert (mem[1] == np.array([[0, 1]])).all()


def test_membership_func_freq_cat_gfmm():
    # no categorical features
    mem = membership_func_freq_cat_gfmm(X[0], X[0], None, V, W, None, None, similarity_of_cat_vals)
    assert (mem == np.array([1, 0.95])).all()
    # test no continuous features
    mem = membership_func_freq_cat_gfmm(None, None, X_cat[0], None, None, E, F, similarity_of_cat_vals)
    assert (mem == np.array([0.625, 1])).all()
    # test the input pattern including both categorical and continuous features
    mem = membership_func_freq_cat_gfmm(X[0], X[0], X_cat[0], V, W, E, F, similarity_of_cat_vals)
    assert (mem == np.array([0.625, 0.95])).all()


def test_get_membership_freq_cat_gfmm_all_classes():
    # no categorical features
    vals = get_membership_freq_cat_gfmm_all_classes(X[0], X[0], None, V, W, None, None, C, similarity_of_cat_vals)
    assert (vals[0] == np.array([[1, 0.95]])).all()
    assert (vals[1] == np.array([[0, 1]])).all()
    # test no continuous features
    vals = get_membership_freq_cat_gfmm_all_classes(None, None, X_cat[0], None, None, E, F, C, similarity_of_cat_vals)
    assert (vals[0] == np.array([[0.625, 1]])).all()
    assert (vals[1] == np.array([[0, 1]])).all()
    # test the input pattern including both categorical and continuous features
    vals = get_membership_freq_cat_gfmm_all_classes(X[0], X[0], X_cat[0], V, W, E, F, C, similarity_of_cat_vals)
    assert (vals[0] == np.array([[0.625, 0.95]])).all()
    assert (vals[1] == np.array([[0, 1]])).all()


def test_membership_cat_feature_eiol_gfmm():
    mem = membership_cat_feature_eiol_gfmm(X_cat[0], D_eiol)
    assert (mem == np.array([0.25, 0.375])).all()
    mem = membership_cat_feature_eiol_gfmm(X_cat[1], D_eiol)
    assert (mem == np.array([0.5625, 0.28125])).all()


def test_membership_func_extended_iol_gfmm():
    # no categorical features
    mem = membership_func_extended_iol_gfmm(X[0], X[0], None, V, W, None, g=1, alpha = 0.6)
    assert (mem == np.array([1, 0.95])).all()
    # test no continuous features
    mem = membership_func_extended_iol_gfmm(None, None, X_cat[0], None, None, D_eiol, g=1, alpha = 0.6)
    assert (mem == np.array([0.25, 0.375])).all()
    # test the input pattern including both categorical and continuous features
    mem = membership_func_extended_iol_gfmm(X[0], X[0], X_cat[0], V, W, D_eiol, g=1, alpha = 0.6)
    assert mem[0] == 0.7
    assert mem[1] == 0.72
    mem = membership_func_extended_iol_gfmm(X[1], X[1], X_cat[1], V, W, D_eiol, g=1, alpha = 0.6)
    assert mem[0] == 0.765
    assert mem[1] == 0.5025000000000001
    

def test_get_membership_extended_iol_gfmm_all_classes():
    # no categorical features
    vals = get_membership_extended_iol_gfmm_all_classes(X, X, None, V, W, None, C, g=1, alpha = 0.6)
    assert (vals[0] == np.array([[1, 0.95], [0.9, 0.65]])).all()
    assert (vals[1] == np.array([[0, 1], [0, 1]])).all()
    # test no continuous features
    vals = get_membership_extended_iol_gfmm_all_classes(None, None, X_cat, None, None, D_eiol, C, g=1, alpha = 0.6)
    assert (vals[0] == np.array([[0.25, 0.375], [0.5625, 0.28125]])).all()
    assert (vals[1] == np.array([[0, 1], [0, 1]])).all()
    # test the input pattern including both categorical and continuous features
    vals = get_membership_extended_iol_gfmm_all_classes(X, X, X_cat, V, W, D_eiol, C, g=1, alpha = 0.6)
    assert (vals[0] == np.array([[0.7, 0.72], [0.765, 0.5025000000000001]])).all()
    assert (vals[1] == np.array([[0, 1], [0, 1]])).all()
