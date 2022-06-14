"""
A bagging of many base hyperbox-based models trained on the full set of features and a subset of samples.
Each base learner is trained by random search-based hyper-parameter tuning and k-fold cross-validation
The predicted class is computed based on the voting mechanism of decisions of base models.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import time
import numpy as np
import threading
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin
from sklearn.ensemble._base import _partition_estimators
from hbbrain.numerical_data.ensemble_learner.base_cross_val_bagging import BaseCrossValBagging
from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
from hbbrain.base.base_ensemble import (
    _covert_empty_class,
    _accumulate_prediction,
    _parallel_predict
)


class DecisionCombinationCrossValBagging(ClassifierMixin, BaseCrossValBagging):
    """A Bagging classifier of base hyperbox-based models trained on a full set of features and 
    a subset of samples, in which each base learner is trained by random search-based hyper-parameter 
    tuning and k-fold cross-validation.
    
    A decision combination cross-validation Bagging classifier of hyperbox-based models is an 
    ensemble meta-estimator that fits base hyperbox-based classifiers each 
    on random subsets of the original samples using random search-based hyper-parameter tuning and 
    k-fold cross-validation, and then aggregate their individual predictions by voting to form a 
    final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a single estimator, by introducing randomization into 
    its construction procedure and then making an ensemble out of it.
    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. See [3]_ for more detailed 
    information regarding the combination of base hyperbox-based models.
    
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
    bootstrap : bool, default=False
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.
    class_balanced : bool, default=False
        Whether samples are drawn without replacement to build a final subset 
        with the equal number of samples among classes.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    scoring : str or callable default='accuracy'
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:
        - a single string (see `The scoring parameter: defining model evaluation rules in sklearn <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>`_).
        - a callable (see `Defining your scoring strategy from metric functions <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_) that returns a single value.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    k_fold : int, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, Stratified K-Fold is used.
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
    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, vol. 36, no. 1, pp. 85-103, 1999.
    .. [2] L. Breiman, "Bagging predictors", Machine Learning, vol. 24, no. 2, pp. 123-140,
           1996.
    .. [3] B. Gabrys,"Combining neuro-fuzzy classifiers for improved generalisation and reliability",
           in Proceedings of the 2002 International Joint Conference on Neural Networks, vol. 3, pp. 2410-2415, 2002.
           
    Examples
    --------
    >>> from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
    >>> from hbbrain.numerical_data.ensemble_learner.decision_comb_cross_val_bagging import DecisionCombinationCrossValBagging
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = DecisionCombinationCrossValBagging(base_estimator=OnlineGFMM(0.1),
    ...                         base_estimator_params = {'theta': np.arange(0.05, 1.01, 0.05), 'theta_min':[1], 'gamma':[0.5, 1, 2, 4, 8, 16]},
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
        bootstrap=False,
        class_balanced=False,
        n_iter=10, 
        scoring='accuracy',
        k_fold=5,
        n_jobs=1,
        random_state=None
    ):
        super().__init__(
            base_estimator=base_estimator,
            base_estimator_params=base_estimator_params,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            class_balanced=class_balanced,
            n_iter=n_iter,
            scoring=scoring,
            k_fold=k_fold,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(default=OnlineGFMM(0.1))
        
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
        if X.ndim == 1:
            X = np.reshape(X, (1, -1))
            
        y = _covert_empty_class(y)
            
        self.classes_= np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        time_start = time.perf_counter()
        
        # Check parameters
        self._validate_estimator()
        
        fitted_est = self._fit(X, y, self.max_samples)
        
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
        
        return fitted_est
    
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
            n_jobs=n_jobs, **self._parallel_args()
        )(
            delayed(_parallel_predict)(
                self.estimators_[starts[i] : starts[i + 1]],
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
            delayed(_accumulate_prediction)(e.predict_with_membership, X, mem_vals, lock)
            for e in self.estimators_
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
            delayed(_accumulate_prediction)(e.predict_proba, X, all_probas, lock)
            for e in self.estimators_
        )

        for proba in all_probas:
            proba /= len(self.estimators_)

        if len(all_probas) == 1:
            return all_probas[0]
        else:
            return all_probas


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
    optional.add_argument('--bootstrap', type=str2bool, default=False,
                          help='Whether samples are drawn with replacement (default: False)')
    optional.add_argument('--class_balanced', type=str2bool, default=False,
                          help='Whether number of samples of different classes is equal to each other (default: False)')
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
        
    if args.n_jobs <= 0:
        parser.error("--n_jobs has to be larger than 0")
        
    if args.n_iter <= 0:
        parser.error("--n_iter has to be larger than 0")
        
    if args.k_fold <= 0:
        parser.error("--k_fold has to be larger than 0")

    n_estimators = args.n_estimators
    max_samples = args.max_samples
    bootstrap = args.bootstrap
    class_balanced = args.class_balanced
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
    dc_cv_bagging = DecisionCombinationCrossValBagging(base_estimator=base_estimator, base_estimator_params=base_estimator_params, n_estimators=n_estimators, max_samples=max_samples, bootstrap=bootstrap, class_balanced=class_balanced, n_iter=n_iter, k_fold=k_fold, n_jobs=n_jobs, random_state=0)
    dc_cv_bagging.fit(Xtr, ytr)
    print("Training time: %.3f (s)"%(dc_cv_bagging.elapsed_training_time))
    print('Number of hyperboxes = %d'%dc_cv_bagging.get_n_hyperboxes())
    
    y_pred = dc_cv_bagging.predict(Xtest)

    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy = {acc * 100: .2f}%')
    
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/syn_num_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy(copy=False)

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # dc_cv_bagging.simple_pruning_base_estimators(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning = %d'%dc_cv_bagging.get_n_hyperboxes())
    
    # y_pred_2 = dc_cv_bagging.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy = {acc_pruned * 100: .2f}%')
    