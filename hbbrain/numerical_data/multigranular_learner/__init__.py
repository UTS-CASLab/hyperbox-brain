"""
The :mod:`hbbrain.numerical_data.multigranular_learner` module implements a
variety of multi-granular learning algorithms using hyperbox representations.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

from hbbrain.numerical_data.multigranular_learner.multi_resolution_gfmm import (
    predict_with_centroids,
    MultiGranularGFMM
)

__all__ = [
    "predict_with_centroids",
    "MultiGranularGFMM"
]
