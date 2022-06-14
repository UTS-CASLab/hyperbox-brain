"""
Functions and classes for the cross-validation random hyperboxes model.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import itertools
import numpy as np
import math
import time
import threading
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_random_state
from sklearn.ensemble._base import _partition_estimators
from joblib import Parallel, delayed
from hbbrain.base.base_ensemble import (
    _generate_indices,
    _balanced_subsample,
    _covert_empty_class,
    _stratified_subsample,
    _accumulate_prediction,
    BaseEnsemble
)
from hbbrain.base.base_gfmm_estimator import BaseGFMMClassifier
from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM

MAX_INT = np.iinfo(np.int32).max


def _parallel_build_cross_val_base_hyperboxes_estimators(n_estimators, ensemble, X, y, seeds):
    """Private function used to build a batch of base cross-validation hyperbox estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    base_estimator_params = ensemble.base_estimator_params
    scoring = ensemble.scoring
    k_fold = ensemble.k_fold
    n_iter = ensemble.n_iter
    
    # Build estimators
    estimators = list()
    estimators_features = list()
    
    for i in range(n_estimators):
        estimator = ensemble._make_estimator(append=False)

        random_state = check_random_state(seeds[i])
        # Draw random feature indices
        if ensemble.feature_balanced == True:
            features = _generate_indices(random_state, False, n_features, max_features)
        else:
            n_sampling_features = random_state.randint(1, max_features + 1, 1)[0]
            features = _generate_indices(random_state, False, n_features, n_sampling_features)
            
        features = np.sort(features)
        # Draw random sample indices
        if ensemble.class_balanced == True:
            sample_indices = _balanced_subsample(y, random_state, max_samples)
        else:
            sample_indices = _stratified_subsample(y, random_state, max_samples)

        clf_rd_search = RandomizedSearchCV(estimator, base_estimator_params, n_iter=n_iter, cv=k_fold, scoring=scoring, random_state=random_state, refit=True)
        clf_rd_search.fit((X[sample_indices])[:, features], y[sample_indices])
        
        estimators.append(clf_rd_search.best_estimator_)
        estimators_features.append(features)
        
    return estimators, estimators_features


def _parallel_predict(estimators, estimators_features, X, classes):
    """Private function used to compute predictions within a job."""
    n_samples = X.shape[0]
    n_classes = len(classes)
    classes = np.sort(classes)
    proba = np.zeros((n_samples, n_classes))
    
    mapping_class_index = {}
    for i, val in enumerate(classes):
        mapping_class_index[val] = i
    
    for estimator, features in zip(estimators, estimators_features):
        # Voting
        predictions = estimator.predict(X[:, features])
        
        for i in range(n_samples):
            proba[i, mapping_class_index[predictions[i]]] += 1

    return proba


class CrossValRandomHyperboxesClassifier(ClassifierMixin, BaseEnsemble):
    """A Corss-validation Random Hyperboxes classifier of base hyperbox-based
    models trained on a subset of features and a subset of samples together
    with random search-based hyper-parameter tuning and k-fold cross-validation.

    A Random Hyperboxes classifier of hyperbox-based models is an 
    ensemble meta-estimator that fits base hyperbox-based classifiers each 
    on random subsets of both original samples and features using k-fold cross-validation and hyper-parameter tuning 
    based on random search. Then, base learners are aggregated with their individual 
    predictions by voting to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a single estimator, by introducing randomization into 
    its construction procedures and then making an ensemble out of it. Subsets of features and samples 
    of the random hyperboxes are builts by random subsampling without replacement.
    See [1]_ for more detailed information regarding the random hyperboxes classifier.

    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~hbbrain.numerical_data.incremental_learner.onln_gfmm.OnlineGFMM`.
    base_estimator_params : dict or list of dicts, default={}
        Dictionary with parameters names (str) as keys and distributions or lists of parameters 
        to try. If a list is given, it is sampled uniformly. If a list of dicts is given, first 
        a dict is sampled uniformly, and then a parameter is sampled using that dict as above.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : int or float, default=0.5
        The number of samples to draw from X to train each base estimator (with
        no replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
    max_features : {"sqrt", "log2"}, int or float, default="sqrt"
        The maximum number of features to consider when building training data for base learners:

        - If int, then consider `max_features` features.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
    class_balanced : bool, default=False
        Whether samples are drawn without replacement to build a final subset 
        with the equal number of samples among classes.
    feature_balanced: bool, default = False
        Whether number of features of training sets for all base learners are equal to 
        each other or not.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    scoring : str or callable default='accuracy'
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:
        - a single string (see `The scoring parameter: defining model evaluation rules in sklearn <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
        - a callable (see `Defining your scoring strategy from metric functions <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_) that returns a single value.
    k_fold : int, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, Stratified K-Fold is used.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    n_features_ : int
        Number of features seen during :term:`fit`.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int or list
        The number of classes.

    References
    ----------
    .. [1] T. T. Khuat and B. Gabrys "Random Hyperboxes", IEEE Transactions on Neural Networks 
           and Learning Systems, 2021.
           
    Examples
    --------
    >>> from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
    >>> from hbbrain.numerical_data.ensemble_learner.cross_val_random_hyperboxes import CrossValRandomHyperboxesClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = CrossValRandomHyperboxesClassifier(base_estimator=ImprovedOnlineGFMM(0.1),
    ...                         base_estimator_params={'theta': np.arange(0.05, 1.01, 0.05), 'gamma':[0.5, 1, 2, 4, 8, 16]},
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> clf.predict([[1, 0.6, 0.5, 0.2]])
    array([1])

    """

    def __init__(
        self,
        base_estimator=None,
        base_estimator_params=dict(),
        n_estimators=10,
        max_samples=0.5,
        max_features='sqrt',
        class_balanced=False,
        feature_balanced=False,
        n_iter=10,
        scoring='accuracy',
        k_fold=5,
        n_jobs=1,
        random_state=None
    ):
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators)
        self.base_estimator_params = base_estimator_params
        self.max_samples = max_samples
        self.max_features = max_features
        self.class_balanced = class_balanced
        self.feature_balanced = feature_balanced
        self.n_iter = n_iter
        self.scoring = scoring
        self.k_fold = k_fold
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def _set_max_feature(self, n_features):
        """
        Set the value for self._max_features
            
        Parameters
        ----------
        n_features: int
            The number of features of input data

        """
        if (self.max_features == 'auto') or (self.max_features == 'sqrt'):
            self._max_features = int(math.sqrt(n_features))
        elif self.max_features == 'log2':
            self._max_features = int(math.log2(n_features))
        elif self.max_features is None:
            self._max_features = n_features
        elif 0 < self.max_features <= 1:
            self._max_features = max(int(self.max_features * n_features), 1)
        else:
            self._max_features = int(self.max_features)
            
    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(default=OnlineGFMM(0.1))
        
    def _get_estimators_indices(self):
        """
        Get drawn indices along the sample and feature axes for all base learners.

        Yields
        ------
        sample_indices : list of arrays
            A list of indices of samples drawn for base learners.
        estimators_features_ : list of arrays
            The subset of indices of the drawn features for each base
            estimator. Each subset is defined by an array of the indices selected.

        """
        for i, seed in enumerate(self._seeds):
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_bagging_estimators()`
            random_state = check_random_state(seed)
            if self.class_balanced == False:
                sample_indices = _stratified_subsample(self._y, random_state, self._max_samples)
            else:
                sample_indices = _balanced_subsample(self._y, random_state, self._max_samples)
            
            yield sample_indices, self.estimators_features_[i]
    
    def _get_given_estimator_indices(self, estimator_id):
        """
        Get drawn indices along the sample and feature axes for a given base
        learners.

        Returns
        -------
        sample_indices : array of int
            Indices of samples drawn for a given base learner.
        estimators_features_ : array of int
            Indices of the drawn features for each base learner.

        """
        random_state = check_random_state(self._seeds[estimator_id])
        if self.class_balanced == False:
            sample_indices = _stratified_subsample(self._y, random_state, self._max_samples)
        else:
            sample_indices = _balanced_subsample(self._y, random_state, self._max_samples)
        
        return sample_indices, self.estimators_features_[estimator_id]

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
        return [sample_indices for sample_indices, _ in self._get_estimators_indices()]

    def fit(self, X, y):
        """
        Build a random hyperbox model from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The class labels.
        
        Returns
        -------
        self : object
            Fitted estimator.

        """
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))
        
        time_start = time.perf_counter()
        
        # Check parameters
        self._validate_estimator()
        
        y = _covert_empty_class(y)
        self._y = y
        self.classes_= np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        random_state = check_random_state(self.random_state)
        self.estimators_ = list()
        self.estimators_features_ = list()
        
        n_samples, n_features = X.shape
        self._n_samples = n_samples
        self._set_max_feature(n_features)
        self.n_features_ = n_features
        
        if not (0 < self._max_features <= n_features):
            raise ValueError(f"max_features must be in (0, {n_features}]")

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
            n_jobs=n_jobs
        )(
            delayed(_parallel_build_cross_val_base_hyperboxes_estimators)(
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
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
        
        return self
    
    def predict(self, X):
        """Predict class for X.
        
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability using voting.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The testing input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.

        """
        X = np.array(X)
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        all_proba = Parallel(
            n_jobs=n_jobs
        )(
            delayed(_parallel_predict)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
                self.classes_,
            )
            for i in range(n_jobs)
        )

        # Reduce
        predicted_probabilitiy = sum(all_proba) / self.n_estimators

        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)

    def predict_with_membership(self, X):
        """
        Predict class memberships for X.

        The predicted class memberships of an input sample are computed as
        the mean predicted class memberships of the hyperbox-based learners in
        the ensemble model. The class membership of a single hyperbox-based
        learner is the membership from the input X to the representative
        hyperbox of that class to join the prediction procedure.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for prediction.

        Returns
        -------
        mem_vals : ndarray of shape (n_samples, n_classes)
            The class memberships of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.
        """
        # Assign chunk of hyperbox-based learners to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        mem_vals = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_with_membership, X[:, f], mem_vals, lock)
            for e, f in zip(self.estimators_, self.estimators_features_)
        )

        for mem in mem_vals:
            mem /= len(self.estimators_)

        if len(mem_vals) == 1:
            return mem_vals[0]
        else:
            return mem_vals

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the hyperbox-based learners
        in the ensemble model. The class probability of a single hyperbox-based
        learner is the fraction of the membership value of the representative
        hyperbox of that class and the sum of all membership values of all
        representative hyperboxes of all classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for prediction.

        Returns
        -------
        all_probas : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.
        """
        # Assign chunk of hyperbox-based learners to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_probas = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_proba, X[:, f], all_probas, lock)
            for e, f in zip(self.estimators_, self.estimators_features_)
        )

        for proba in all_probas:
            proba /= len(self.estimators_)

        if len(all_probas) == 1:
            return all_probas[0]
        else:
            return all_probas    

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
            A random hyperboxes model with base estimators prunned.

        """
        for estimator, features in zip(self.estimators_, self.estimators_features_):
            if isinstance(estimator, BaseGFMMClassifier):
                estimator.simple_pruning(X_val[:, features], X_val[:, features], y_val, acc_threshold, keep_empty_boxes)
            else:
                estimator.simple_pruning(X_val[:, features], y_val, acc_threshold, keep_empty_boxes)
                
        return self
    
    def get_n_hyperboxes(self):
        """
        Get total number of hyperboxes in all base learners.

        Returns
        -------
        int
            Total number of hyperboxes in all base learners.

        """
        n_hyperboxes = 0
        for estimator in self.estimators_:
            n_hyperboxes += estimator.get_n_hyperboxes()
            
        return n_hyperboxes


if __name__ == '__main__':
    import argparse
    import os
    from sklearn.metrics import accuracy_score

    def dir_path(path):
        if os.path.isfile(path) and os.path.exists(path):
            return path
        else:
            raise argparse.ArgumentTypeError(
                f"{path} is not a valid path or file does not exist")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Expect {v} is an boolean value")

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='The description of parameters')

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required positional arguments
    required.add_argument('-training_file', type=dir_path,
                          help='A required argument for the path to training data file (including file name)', required=True)
    required.add_argument('-testing_file', type=dir_path,
                          help='A required argument for the path to testing data file (including file name)', required=True)

    # Optional arguments
    optional.add_argument('--n_estimators', type=int, default=10,
                          help='Number of base estimators in the ensemble model (default: 10)')
    optional.add_argument('--max_samples', type=float, default=0.5,
                          help='Number of samples or sample rate of original datasets to build training set for base estimators (default: 0.5)')
    optional.add_argument('--max_features', type=float, default=0.5,
                          help='Number of features or feature rate of original datasets to build training set for base estimators (default: 0.5)')
    optional.add_argument('--feature_balanced', type=str2bool, default=False,
                          help='Whether the number of features of all base learners is equal to each other (default: False)')
    optional.add_argument('--class_balanced', type=str2bool, default=False,
                          help='Whether the number of samples of different classes is equal to each other (default: False)')
    optional.add_argument('--n_iter', type=int, default=10,
                          help='Number of parameter settings are sampled for each base learn during the parameter tuning process (default: 10)')
    optional.add_argument('--k_fold', type=int, default=5,
                          help='Number of folds are used for parameter tuning each base learner (default: 5)')
    optional.add_argument('--n_jobs', type=int, default=1,
                          help='The number of jobs to run in parallel for base model building (default: 1)')

    args = parser.parse_args()

    if args.n_estimators <= 0:
        parser.error("--n_estimators has to be larger than 0")

    if args.max_samples <= 0:
        parser.error("--max_samples has to be larger than 0")
        
    if args.max_features <= 0:
        parser.error("--max_features has to be larger than 0")
        
    if args.n_jobs <= 0:
        parser.error("--n_jobs has to be larger than 0")
        
    if args.n_iter <= 0:
        parser.error("--n_iter has to be larger than 0")
        
    if args.k_fold <= 0:
        parser.error("--k_fold has to be larger than 0")

    n_estimators = args.n_estimators
    max_samples = args.max_samples
    max_features = args.max_features
    class_balanced = args.class_balanced
    feature_balanced = args.feature_balanced
    n_iter = args.n_iter
    k_fold = args.k_fold
    n_jobs = args.n_jobs
    training_file = args.training_file
    testing_file = args.testing_file
    
    import pandas as pd
    df_train = pd.read_csv(training_file, header=None)
    df_test = pd.read_csv(testing_file, header=None)

    Xy_train = df_train.to_numpy()
    Xy_test = df_test.to_numpy()

    Xtr = Xy_train[:, :-1]
    ytr = Xy_train[:, -1]

    Xtest = Xy_test[:, :-1]
    ytest = Xy_test[:, -1]

    base_estimator = OnlineGFMM()
    base_estimator_params = {'theta': np.arange(0.05, 1.01, 0.05), 'theta_min':[1], 'gamma':[0.5, 1, 2, 4, 8, 16]}
    cross_val_rh_clf = CrossValRandomHyperboxesClassifier(base_estimator=base_estimator, base_estimator_params=base_estimator_params, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, class_balanced=class_balanced, feature_balanced=feature_balanced, n_iter=n_iter, k_fold=k_fold, n_jobs=n_jobs, random_state=0)
    cross_val_rh_clf.fit(Xtr, ytr)
    print("Training time: %.3f (s)"%(cross_val_rh_clf.elapsed_training_time))
    print('Number of hyperboxes = %d'%cross_val_rh_clf.get_n_hyperboxes())
    
    y_pred = cross_val_rh_clf.predict(Xtest)

    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/syn_num_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # cross_val_rh_clf.simple_pruning_base_estimators(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%cross_val_rh_clf.get_n_hyperboxes())
    
    # y_pred_2 = cross_val_rh_clf.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy = {acc_pruned * 100: .2f}%')