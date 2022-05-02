"""
The :mod:`hbbrain.numerical_data.incremental_learner` module implements a
variety of incremental learning algorithms using hyperbox representations.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0
from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
from hbbrain.numerical_data.incremental_learner.rfmnn import RFMNNClassifier
from hbbrain.numerical_data.incremental_learner.knefmnn import KNEFMNNClassifier
from hbbrain.numerical_data.incremental_learner.efmnn import EFMNNClassifier
from hbbrain.numerical_data.incremental_learner.fmnn import FMNNClassifier
from hbbrain.numerical_data.incremental_learner.inf_onln_gfmm import InfOnlineGFMM

__all__ = [
      "OnlineGFMM",
      "ImprovedOnlineGFMM",
      "RFMNNClassifier",
      "KNEFMNNClassifier",
      "EFMNNClassifier",
      "FMNNClassifier",
      "InfOnlineGFMM"
]
