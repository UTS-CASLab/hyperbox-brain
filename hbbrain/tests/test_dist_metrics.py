# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0


import numpy as np
from hbbrain.utils.dist_metrics import (
    manhattan_distance,
    rfmnn_distance
)


def test_manhattan_distance_two_point():
    x = np.array([0.2, 0.1])
    y = np.array([0.5, 0.6])
    res = manhattan_distance(x, y)
    assert res == 0.8


def test_manhattan_distance_two_matrices():
    X = np.array([[0.2, 0.1],
                  [0.8, 0.6]])
    Y = np.array([[0.5, 0.6],
                  [0.1, 0.1]])
    res = manhattan_distance(X, Y)
    assert res[0] == 0.8
    assert np.round(res[1], 1) == 1.2


def test_rfmnn_distance_one_hyperbox():
    x = np.array([0.2, 0.1])
    v = np.array([0.4, 0.5])
    w = np.array([0.8, 0.7])
    res = rfmnn_distance(x, v, w)
    assert res == 0.45


def test_rfmnn_distance_one_input_two_hyperboxes():
    x = np.array([0.2, 0.1])
    V = np.array([[0.4, 0.5],
                  [0.3, 0.4]])
    W = np.array([[0.8, 0.7],
                  [0.5, 0.6]])
    res = rfmnn_distance(x, V, W)
    assert res[0] == 0.45
    assert res[1] == 0.3


def test_rfmnn_distance_two_inputs_two_hyperboxes():
    X = np.array([[0.2, 0.1],
                  [0.1, 0.25]])
    V = np.array([[0.4, 0.5],
                  [0.3, 0.4]])
    W = np.array([[0.8, 0.7],
                  [0.5, 0.6]])
    res = rfmnn_distance(X, V, W)
    assert res[0] == 0.45
    assert res[1] == 0.275
