"""
Base functions and classes for bagging models using hyperbox-based models.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import itertools
import numpy as np
import threading
from sklearn.utils import check_random_state
from sklearn.ensemble._base import _partition_estimators
from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from hbbrain.base.base_ensemble import (
    _generate_indices,
    _balanced_subsample,
    BaseEnsemble,
    _covert_empty_class
)
from hbbrain.base.base_gfmm_estimator import BaseGFMMClassifier

MAX_INT = np.iinfo(np.int32).max


def _parallel_build_bagging_estimators(n_estimators, ensemble, X, y, seeds):
    """Private function used to build a batch of bagging estimators within a job."""
    # Retrieve settings
    n_samples, n_features = np.shape(X)
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    
    # Build estimators
    estimators = list()
    
    for i in range(n_estimators):
        estimator = ensemble._make_estimator(append=False)

        random_state = check_random_state(seeds[i])
        
        # Draw random sample indices
        if ensemble.class_balanced == True:
            sample_indices = _balanced_subsample(y, random_state, max_samples)
        else:
            sample_indices = _generate_indices(random_state, bootstrap, n_samples, max_samples)

        estimator.fit(X[sample_indices], y[sample_indices])

        estimators.append(estimator)
        
    return estimators


class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=0.5,
        bootstrap=False,
        class_balanced=False,
        n_jobs=None,
        random_state=None
    ):
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators)

        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.class_balanced = class_balanced
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The real class labels
        
        Returns
        -------
        self : object
            Fitted estimator.

        """        
        return self._fit(X, y, self.max_samples)

    def _parallel_args(self):
        return {}

    def _fit(self, X, y, max_samples=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The class labels.
        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.
        
        Returns
        -------
        self : object
            Fitted estimator.

        """
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))
            
        y = _covert_empty_class(y)
        self._y = y
        
        random_state = check_random_state(self.random_state)
        self.estimators_ = list()
        n_samples = X.shape[0]
        self._n_samples = n_samples

        # Validate max_samples
        # Validate max_samples
        if self.max_samples is None:
            max_samples = self.max_samples
        elif 0 < self.max_samples <= 1:
            max_samples = int(self.max_samples * X.shape[0])
        else:
            max_samples = int(self.max_samples)

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples
        
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, **self._parallel_args()
        )(
            delayed(_parallel_build_bagging_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                seeds[starts[i] : starts[i + 1]]
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t for t in all_results)
        )

        return self

    def simple_pruning_base_estimators(self, X_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Simply prune low qualitied hyperboxes based on a pre-defined accuracy threshold for each hyperbox. This operation 
        is applied for all base estimators.

        Parameters
        ----------
        X_val : array-like of shape (n_samples, n_features)
            The data matrix contains validation patterns.
        y_val : ndarray of shape (n_samples,)
            A vector contains the true class label corresponding to each validation pattern.
        acc_threshold : float, optional, default=0.5
            The minimum accuracy for each hyperbox to be kept unchanged.
        keep_empty_boxes : boolean, optional, default=False
            Whether to keep the hyperboxes which do not join the prediction process on the validation set.
            If True, keep them, else the decision for keeping or removing based on the classification accuracy on the validation dataset

        Returns
        -------
        self
            A bagging model with base estimators prunned.

        """
        for estimator in self.estimators_:
            if isinstance(estimator, BaseGFMMClassifier):
                estimator.simple_pruning(X_val, X_val, y_val, acc_threshold, keep_empty_boxes)
            else:
                estimator.simple_pruning(X_val, y_val, acc_threshold, keep_empty_boxes)
                
        return self
    
    def get_n_hyperboxes(self):
        """
        Get total number of hyperboxes in all base learners.

        Returns
        -------
        n_hyperboxes : int
            Total number of hyperboxes in all base learners.

        """
        n_hyperboxes = 0
        for estimator in self.estimators_:
            n_hyperboxes += estimator.get_n_hyperboxes()
            
        return n_hyperboxes

    def _get_estimators_indices(self):
        # Get drawn indices along the sample axis
        for seed in self._seeds:
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_bagging_estimators()`
            random_state = check_random_state(seed)
            if self.class_balanced == False:
                sample_indices = _generate_indices(random_state, self.bootstrap, self._n_samples, self._max_samples)
            else:
                sample_indices = _balanced_subsample(self._y, random_state, self._max_samples)
            
            yield sample_indices
    
    def _get_estimator_sample_indices(self, estimator_id):
        # Get drawn indices along the sample axis for a specific estimator
        random_state = check_random_state(self._seeds[estimator_id])
        if self.class_balanced == False:
            sample_indices = _generate_indices(random_state, self.bootstrap, self._n_samples, self._max_samples)
        else:
            sample_indices = _balanced_subsample(self._y, random_state, self._max_samples)
        
        return sample_indices

    @property
    def estimators_samples_(self):
        """
        The subset of drawn samples for each base estimator.
        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        .. note::

            The list is re-created at each call to the property in order to
            reduce the object memory footprint by not storing the sampling data.
            Thus fetching the property may be slower than expected.

        """
        return [sample_indices for sample_indices in self._get_estimators_indices()]