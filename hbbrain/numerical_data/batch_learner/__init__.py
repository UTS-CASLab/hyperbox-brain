"""
The :mod:`hbbrain.numerical_data.batch_learner` module implements a variety of
agglomerative learning algorithms using hyperbox representations.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

from hbbrain.numerical_data.batch_learner.agglo_gfmm import (
    AgglomerativeLearningGFMM
)
from hbbrain.numerical_data.batch_learner.accel_agglo_gfmm import (
    AccelAgglomerativeLearningGFMM
)

__all__ = [
    "AgglomerativeLearningGFMM",
    "AccelAgglomerativeLearningGFMM"
]
