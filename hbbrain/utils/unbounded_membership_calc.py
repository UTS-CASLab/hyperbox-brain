# Import Necessary Libraries
import pandas as pd
import numpy as np

# Some utils
def impute_missing_val_x (x):
    return impute_missing_val_x(x,x)

def impute_missing_val_x (xl, xu):
    xl = xl.astype(float)
    xu = xu.astype(float)
    missing_mask = np.isnan(xl) | np.isnan(xu)
    xl[missing_mask] = np.inf
    xu[missing_mask] = -np.inf
    return xl, xu

def impute_missing_val_term (term):
    return np.where(np.isinf(term), 0.0, term)

def preprocess (xl, xu , V, W):
    V = V.astype(float)
    W = W.astype(float)
    xl, xu = impute_missing_val_x (xl,xu)
    xl = np.broadcast_to(xl, V.shape)
    xu = np.broadcast_to(xu, V.shape)
    return xl, xu, V, W

def calculate_term_middle_distances (xl, xu, V, W):
    # h2 -> h1
    # Middle path
    term1 = impute_missing_val_term (xu - W ) # w2 - w1
    term2 = impute_missing_val_term (V - xl) # v1 - v2

    # Longest path
    term3 = impute_missing_val_term (xu-V) # w2 - v1
    term4 = impute_missing_val_term (W - xl) # w1 - v2

    middle_dist = np.maximum(np.maximum(term1, term2), 0.0)
    long_dist = np.maximum(np.maximum(term3, term4), 0.0)
    min_dist = np.minimum(middle_dist, long_dist)

    return min_dist # n_hyperboxes x n_features

# First function utilising the sum squares of the 2 longest distance
def unbounded_gfmm_membership_func1(xl, xu, V, W, alpha=1.0):
    # h2 -> h1
    xl, xu, V,W = preprocess (xl, xu, V , W)

    # Shortest
    term1 = impute_missing_val_term(V - xu)
    term2 = impute_missing_val_term(xl - W)
    # Longest
    term3 = impute_missing_val_term(W - xl)
    term4 = impute_missing_val_term(xu - V)

    short_dist = np.maximum(np.maximum(term1,term2), 0.0)
    long_dist = np.maximum(np.maximum(term3,term4), 0.0)


    d = np.sqrt(np.sum(short_dist**2 + long_dist**2, axis = 1))
    return np.exp(-alpha * d**2)

def unbounded_gfmm_membership_func2 (xl, xu, V, W, alpha = 1.0):
    # xl, xu -> h2
    xl, xu, V, W = preprocess (xl, xu, V , W)
    min_dist = calculate_term_middle_distances(xl, xu, V , W)
    d = np.sqrt(np.sum(min_dist **2, axis = 1))
    return np.exp(-alpha * d**2)

def unbounded_gfmm_membership_func3 (xl, xu, V, W, alpha = 1.0):
    # xl, xu -> h2
    xl, xu, V, W = preprocess (xl, xu, V , W)
    min_dist = calculate_term_middle_distances(xl, xu, V , W)
    d = np.sum(min_dist , axis = 1)
    return np.exp(-alpha * d)

def unbounded_gfmm_membership_func4 (xl, xu, V, W, alpha = 1.0):
    # xl, xu -> h2
    xl, xu, V, W = preprocess (xl, xu, V , W)
    min_dist = calculate_term_middle_distances(xl, xu, V , W)
    return np.min(np.exp(-alpha * min_dist), axis = 1)


# V1 = np.array([[None]])
# W1 = np.array([[0.2]])

# V2 = np.array([[0.3]])
# W2 = np.array([[0.6]])

# print (unbounded_gfmm_membership_func1(V1, W1, V2, W2))
# print (unbounded_gfmm_membership_func2(V1, W1, V2, W2))
# print (unbounded_gfmm_memebership_func3(V1, W1, V2, W2))
# print (unbounded_gfmm_membership_func4 (V1, W1, V2, W2))

# xl = np.array([[0.1]])
# xu = np.array([[0.4]])

# V = np.array([[0.2]])
# W = np.array([[0.6]])

# print (unbounded_gfmm_membership_func1(xl, xu , V, W))
# print (unbounded_gfmm_membership_func2(xl, xu , V, W))
# print (unbounded_gfmm_memebership_func3(xl, xu , V, W))
# print (unbounded_gfmm_membership_func4 (xl, xu , V, W))