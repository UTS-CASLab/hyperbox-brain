"""
Base functions and classes for ensemble models using hyperbox-based models.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

from hbbrain.constants import UNLABELED_CLASS
from sklearn.utils.random import sample_without_replacement
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from typing import List
from abc import ABCMeta, abstractmethod
import numbers
import pandas as pd
import numpy as np


def _covert_empty_class(y):
    """ Covert missing values in classes into the standard form"""
    if np.isnan(y).sum() > 0:
        y_out = np.where(np.isnan(y), UNLABELED_CLASS, y)
        return y_out
    else:
        return y


def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )

    return indices


def _balanced_subsample(y, random_state, n_sampling_samples=None):
    """
    Draw randomly sampled indices to build class-balanced subsets

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        A vector stores all class labels of training data from which balanced subsets are drawn.
    random_state : int, RandomState instance or None
        Controls the random resampling of the original dataset
    n_sampling_samples : int, optional, default=None
        Total number of samples needs to draw to build a class-balanced set.

    Returns
    -------
    subsample : a list of shape (n_sampling_samples,)
        Indices of class-balanced samples which are drawn from y.

    """
    y = pd.Series(y)
    subsample = []

    if n_sampling_samples is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(n_sampling_samples / len(y.value_counts().index))
        n_samples_minor_class = y.value_counts().min()
        if n_smp > n_samples_minor_class:
            n_smp = n_samples_minor_class
            
    for label in y.value_counts().index:
        samples = y[y == label].index.values
        indexes = sample_without_replacement(samples.shape[0], n_smp, random_state=random_state)
        subsample += samples[indexes].tolist()

    return subsample


def _stratified_subsample(y, random_state, n_sampling_samples):
    """
    Draw randomly sampled indices to build class-balanced subsets

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        A vector stores all class labels of training data from which balanced subsets are drawn.
    random_state : RandomState instance
        Controls the random resampling of the original dataset
    n_sampling_samples : int
        Total number of samples needs to draw to build a subset of samples.

    Returns
    -------
    subsample : a list of shape (n_sampling_samples,)
        Indices of samples which are drawn from y.

    """
    n_samples = len(y)
    sampling_rate = n_sampling_samples/n_samples
    subsample = []
    classes = np.unique(y)

    for label in classes:
        id_samples = np.nonzero(y == label)[0]
        n_smp = int(sampling_rate * id_samples.shape[0] + 0.5)
        if n_smp > 0:
            indexes = sample_without_replacement(id_samples.shape[0], n_smp, random_state=random_state)
            subsample += id_samples[indexes].tolist()
    
    random_state.shuffle(subsample)
    
    return subsample


def _parallel_predict(estimators, X, classes):
    """Private function used to compute predictions within a job."""
    n_samples = X.shape[0]
    n_classes = len(classes)
    classes = np.sort(classes)
    proba = np.zeros((n_samples, n_classes))
    
    mapping_class_index = {}
    for i, val in enumerate(classes):
        mapping_class_index[val] = i
    
    for estimator in estimators:
        # Voting
        predictions = estimator.predict(X)
        
        for i in range(n_samples):
            proba[i, mapping_class_index[predictions[i]]] += 1

    return proba


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.
    
    It can't go locally in hyperbox-based ensemble models, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.
    Warning: This class should not be used directly. Use derived classes
    instead.
    
    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.
    n_estimators : int, default=10
        The number of estimators in the ensemble.
    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.
    
    Attributes
    -----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of estimators
        The collection of fitted base estimators.

    """

    # overwrite _required_parameters from MetaEstimatorMixin
    _required_parameters: List[str] = []

    @abstractmethod
    def __init__(self, base_estimator, *, n_estimators=10, estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute.
        Sets the base_estimator_` attributes.
        """
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(
                "n_estimators must be an integer, got {0}.".format(
                    type(self.n_estimators)
                )
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, got {0}.".format(
                    self.n_estimators
                )
            )

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True):
        """Make and configure a copy of the `base_estimator_` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)