"""
The :mod:`hbbrain.mixed_data` module implements a variety of learning
algorithms using hyperbox representations for mixed-attribute data.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

from hbbrain.mixed_data.onehot_onln_gfmm import (
    one_hot_encoding_cat_feature,
    predict_onehot_cat_feature_manhanttan,
    impute_missing_value_cat_feature,
    OneHotOnlineGFMM
)
from hbbrain.mixed_data.freq_cat_onln_gfmm import (
    ordinal_encode_categorical_features,
    compute_similarity_among_categorical_values,
    predict_freq_cat_feature_manhanttan,
    FreqCatOnlineGFMM
)
from hbbrain.mixed_data.eiol_gfmm import (
    predict_with_manhattan_mixed_data,
    predict_with_probability_mixed_data,
    ExtendedImprovedOnlineGFMM
)

__all__ = [
    "one_hot_encoding_cat_feature",
    "predict_onehot_cat_feature_manhanttan",
    "impute_missing_value_cat_feature",
    "OneHotOnlineGFMM",
    "ordinal_encode_categorical_features",
    "compute_similarity_among_categorical_values",
    "predict_freq_cat_feature_manhanttan",
    "FreqCatOnlineGFMM",
    "predict_with_manhattan_mixed_data",
    "predict_with_probability_mixed_data",
    "ExtendedImprovedOnlineGFMM"
]
