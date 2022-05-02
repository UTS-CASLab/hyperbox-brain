"""
The :mod:`hbbrain.base` module implements a variety of base classes and
functions for hyperbox-based classifiers.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

from hbbrain.base.base_estimator import BaseHyperboxClassifier
from hbbrain.base.base_gfmm_estimator import (
    BaseGFMMClassifier,
    predict_with_manhattan,
    predict_with_probability,
    convert_format_missing_input_zero_one,
    is_contain_missing_value,
)
from hbbrain.base.base_ensemble import (
    BaseEnsemble,
    _covert_empty_class,
    _generate_indices,
    _balanced_subsample,
    _stratified_subsample
)
from hbbrain.base.base_fmnn_estimator import (
    predict_with_manhattan_fmnn,
    BaseFMNNClassifier
)

__all__ = [
    "predict_with_manhattan",
    "predict_with_manhattan_fmnn",
    "predict_with_probability",
    "convert_format_missing_input_zero_one",
    "is_contain_missing_value",
    "BaseHyperboxClassifier",
    "BaseGFMMClassifier",
    "BaseEnsemble",
    "BaseFMNNClassifier",
    "_covert_empty_class",
    "_generate_indices",
    "_balanced_subsample",
    "_stratified_subsample",
]
