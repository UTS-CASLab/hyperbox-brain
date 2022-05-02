"""
The :mod:`hbbrain.numerical_data.ensemble_learner` module implements a variety of 
ensemble learning algorithms using hyperbox representations.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

from hbbrain.numerical_data.ensemble_learner.cross_val_random_hyperboxes import (
    CrossValRandomHyperboxesClassifier
)
from hbbrain.numerical_data.ensemble_learner.random_hyperboxes import (
    RandomHyperboxesClassifier
)
from hbbrain.numerical_data.ensemble_learner.decision_comb_bagging import (
    DecisionCombinationBagging
)
from hbbrain.numerical_data.ensemble_learner.decision_comb_cross_val_bagging import (
    DecisionCombinationCrossValBagging
)
from hbbrain.numerical_data.ensemble_learner.model_comb_bagging import (
    ModelCombinationBagging
)
from hbbrain.numerical_data.ensemble_learner.model_comb_cross_val_bagging import (
    ModelCombinationCrossValBagging
)

__all__ = [
    "CrossValRandomHyperboxesClassifier",
    "RandomHyperboxesClassifier",
    "DecisionCombinationBagging",
    "DecisionCombinationCrossValBagging",
    "ModelCombinationBagging",
    "ModelCombinationCrossValBagging"
]
