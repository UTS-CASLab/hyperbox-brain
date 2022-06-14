"""
A bagging of many base hyperbox-based models trained on the full set of features and a subset of samples.
After formulation of base learners models, their hyperboxes are combined into a single model.
The predicted class is computed based on the final single model.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import time
import numpy as np
import threading
from sklearn.base import ClassifierMixin
from joblib import Parallel, delayed
from sklearn.ensemble._base import _partition_estimators
from hbbrain.numerical_data.ensemble_learner.base_bagging import BaseBagging
from hbbrain.numerical_data.incremental_learner.onln_gfmm import OnlineGFMM
from hbbrain.numerical_data.batch_learner.accel_agglo_gfmm import AccelAgglomerativeLearningGFMM
from hbbrain.numerical_data.batch_learner.agglo_gfmm import AgglomerativeLearningGFMM
from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
from hbbrain.base.base_gfmm_estimator import BaseGFMMClassifier
from hbbrain.base.base_ensemble import (
    _covert_empty_class,
    _parallel_predict,
    _accumulate_prediction
)
from hbbrain.utils.membership_calc import (
    get_membership_gfmm_all_classes,
    get_membership_fmnn_all_classes
)


class ModelCombinationBagging(ClassifierMixin, BaseBagging):
    """A Bagging classifier of base hyperbox-based models trained on a full set of features and 
    a subset of samples. Then, the hyperboxes from all base learners are combined to a single model.
    
    A model-level combination Bagging classifier of hyperbox-based models is an 
    ensemble meta-estimator that fits base hyperbox-based classifiers each 
    on random subsets of the original samples and then aggregate their hyperboxes 
    into a single model and use this model for prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a single estimator, by introducing randomization into 
    its construction procedure and then making an ensemble out of it.
    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. See [3]_ for more detailed 
    information regarding the combination of hyperboxes from all base hyperbox-based models.
    
    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~hbbrain.numerical_data.incremental_learner.onln_gfmm.OnlineGFMM`.
    model_level_estimator : object, default=None
        The estimator is used to aggregate all resulting hyperboxes from all base learners
        into a single model. If None, then the base estimator is a
        :class:`~hbbrain.numerical_data.batch_learner.accel_agglo_gfmm.AccelAgglomerativeLearningGFMM`.
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
    model_level_estimator_ : estimator
        The final estimator is the combination of hyperboxes from all base learners.
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
    >>> from hbbrain.numerical_data.incremental_learner.iol_gfmm import ImprovedOnlineGFMM
    >>> from hbbrain.numerical_data.ensemble_learner.model_comb_bagging import ModelCombinationBagging
    >>> from hbbrain.numerical_data.batch_learner.accel_agglo_gfmm import AccelAgglomerativeLearningGFMM
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    MinMaxScaler()
    >>> X = scaler.transform(X)
    >>> clf = ModelCombinationBagging(base_estimator=ImprovedOnlineGFMM(0.1), 
    ...                            model_level_estimator=AccelAgglomerativeLearningGFMM(0.1),
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> clf.predict([[1, 0.6, 0.5, 0.2]])
    array([1])

    """

    def __init__(
        self,
        base_estimator=None,
        model_level_estimator=None,
        n_estimators=10,
        max_samples=0.5,
        bootstrap=False,
        class_balanced=False,
        n_jobs=1,
        random_state=None
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            class_balanced=class_balanced,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.model_level_estimator = model_level_estimator

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(default=OnlineGFMM(0.1))
        
        if self.model_level_estimator is not None:
            self.model_level_estimator_ = self.model_level_estimator
        else:
            self.model_level_estimator_ = AccelAgglomerativeLearningGFMM(theta=self.base_estimator_.theta)
    
        if self.model_level_estimator_ is None:
            raise ValueError("model_level_estimator cannot be None")
        else:
            if isinstance(self.model_level_estimator_, BaseGFMMClassifier) == False:
                raise ValueError("model_level_estimator must be a general fuzzy min-max neural network")

    def fit(self, X, y, is_pruning_base_learners = False, X_val=None, y_val=None, acc_threshold=0.5, keep_empty_boxes=False):
        """Build a Bagging ensemble of estimators from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The real class labels
        is_pruning_base_learners : boolean, optional, default=False
            Whether the pruning procedure can be applied for base learners or not
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
        
        if (is_pruning_base_learners == True) and (X_val is None or y_val is None):
            raise ValueError("Validation set needs to be provided for pruning procedure.")
        
        self._fit(X, y, self.max_samples)
        
        if is_pruning_base_learners == True:
            self.simple_pruning_base_estimators(X_val, y_val, acc_threshold, keep_empty_boxes)
        
        # merge all resulting hyperboxes from base learners
        count = 0
        N_samples = None
        for estimator in self.estimators_:
            count += 1
            if count == 1:
                V = estimator.V.copy()
                W = estimator.W.copy()
                C = estimator.C.copy()
                if isinstance(estimator, (AccelAgglomerativeLearningGFMM, AgglomerativeLearningGFMM, ImprovedOnlineGFMM)) == True:
                    N_samples = estimator.N_samples.copy()
            else:
                V = np.concatenate((V, estimator.V), axis=0)
                W = np.concatenate((W, estimator.W), axis=0)
                C = np.concatenate((C, estimator.C))
                if isinstance(estimator, (AccelAgglomerativeLearningGFMM, AgglomerativeLearningGFMM, ImprovedOnlineGFMM)) == True:
                    N_samples = np.concatenate((N_samples, estimator.N_samples))
        
        if isinstance(self.model_level_estimator_, (AccelAgglomerativeLearningGFMM, AgglomerativeLearningGFMM, ImprovedOnlineGFMM)) == True:
            self.model_level_estimator_._fit(V, W, C, N_samples)
        else:
            self.model_level_estimator_._fit(V, W, C)
        
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
        
        return self

    def predict(self, X, type_boundary_handling=-1):
        """Predict class for X.
        
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability using voting.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The testing input samples.
        type_boundary_handling : int, optional, default=-1
            The way of handling many winner hyperboxes.
            This parameter is only used in the case of `model_level_estimator` being
            improved online learing algorithm or aggolomerative learning algorithms.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.

        """
        X = np.array(X)
        if isinstance(self.model_level_estimator_, (AccelAgglomerativeLearningGFMM, AgglomerativeLearningGFMM, ImprovedOnlineGFMM)) == True:
            y_pred = self.model_level_estimator_.predict(X, type_boundary_handling)
        else:
            y_pred = self.model_level_estimator_.predict(X)
            
        return y_pred
    
    def predict_with_membership(self, X):
        """
        Predict class membership values of the input samples X.
        
        The predicted class membership value is the membership value
        of the representative hyperbox of that class in the prediction
        procedure using the final combined model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        mem_vals : ndarray of shape (n_samples, n_classes)
            The class membership values of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        if isinstance(self.model_level_estimator_, (AccelAgglomerativeLearningGFMM, AgglomerativeLearningGFMM, ImprovedOnlineGFMM, OnlineGFMM)) == True:
            mem_vals, _ = get_membership_gfmm_all_classes(X, X, self.model_level_estimator_.V, self.model_level_estimator_.W, self.model_level_estimator_.C, self.gamma)
        else:
            mem_vals, _ = get_membership_fmnn_all_classes(X, self.model_level_estimator_.V, self.model_level_estimator_.W, self.model_level_estimator_.C, self.gamma)

        return mem_vals

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.
        
        The predicted class probability is the fraction of the membership value
        of the representative hyperbox of that class and the sum of all
        membership values of all representative hyperboxes of all classes
        in the prediction procedure using the final combined model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in ascending integers of class labels.

        """
        mem_vals = self.predict_with_membership(X)
        normalizer = mem_vals.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = mem_vals / normalizer

        return proba

    def predict_voting(self, X):
        """Predict class for X.
        
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability using voting from all base learners.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The testing input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.

        """
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

    def predict_with_membership_all_base_learners(self, X):
        """
        Predict mean class memberships for X from all base learners.

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

    def predict_proba_all_base_learners(self, X):
        """
        Predict mean class probabilities for X from all base learners.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of all hyperbox-based learners
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

    def simple_pruning(self, X_val, y_val, acc_threshold=0.5, keep_empty_boxes=False):
        """
        Simply prune low qualitied hyperboxes based on a pre-defined accuracy threshold for each hyperbox. This operation 
        is applied for the final combined model.

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
            A final hyperbox-based model is prunned of low-quality hyperboxes.

        """
        self.model_level_estimator_.simple_pruning(X_val, X_val, y_val, acc_threshold, keep_empty_boxes)
           
        return self

    def get_n_hyperboxes_comb_model(self):
        """
        Get number of hyperboxes in the final combined model from all
        hyperboxes of base learners

        Returns
        -------
        int
            Total number of hyperboxes in the final combined models from
            all resulting hyperboxes of all base learners.

        """
        return self.model_level_estimator_.get_n_hyperboxes()


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
    optional.add_argument('--n_jobs', type=int, default=1,
                          help='The number of jobs to run in parallel for base model building (default: 1)')

    args = parser.parse_args()

    if args.n_estimators <= 0:
        parser.error("--n_estimators has to be larger than 0")

    if args.max_samples <= 0:
        parser.error("--max_samples has to be larger than 0")
        
    if args.n_jobs <= 0:
        parser.error("--n_jobs has to be larger than 0")

    n_estimators = args.n_estimators
    max_samples = args.max_samples
    bootstrap = args.bootstrap
    class_balanced = args.class_balanced
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
    
    base_estimator = ImprovedOnlineGFMM(0.1)
    model_level_estimator = AccelAgglomerativeLearningGFMM(theta=0.1, min_simil=0, simil_measure='long')
    model_comb_bagging = ModelCombinationBagging(base_estimator=base_estimator, model_level_estimator=model_level_estimator, n_estimators=n_estimators, max_samples=max_samples, bootstrap=bootstrap, class_balanced=class_balanced, n_jobs=n_jobs, random_state=0)
    model_comb_bagging.fit(Xtr, ytr)
    print("Training time: %.3f (s)"%(model_comb_bagging.elapsed_training_time))
    print('Number of hyperboxes in all base learners = %d'%model_comb_bagging.get_n_hyperboxes())
    print('Number of hyperboxes in the combined model = %d'%model_comb_bagging.get_n_hyperboxes_comb_model())
    
    y_pred_voting = model_comb_bagging.predict_voting(Xtest)
    y_pred = model_comb_bagging.predict(Xtest)
    
    acc_voting = accuracy_score(ytest, y_pred_voting)
    acc = accuracy_score(ytest, y_pred)
    print(f'Testing accuracy using voting of decisions from base learners = {acc_voting * 100 : .2f}%')
    print(f'Testing accuracy of the combined model = {acc * 100: .2f}%')
    
    # print("Do pruning")
    # val_file = "/hyperbox-brain/dataset/syn_num_val.csv"
    # df_val = pd.read_csv(val_file, header=None)
    # Xy_val = df_val.to_numpy()

    # X_val = Xy_val[:, :-1]
    # y_val = Xy_val[:, -1]
    
    # model_comb_bagging.simple_pruning(X_val, y_val, 0.5, False)
    # print('Number of hyperboxes after pruning final model = %d'%model_comb_bagging.get_n_hyperboxes_comb_model())
    
    # y_pred_2 = model_comb_bagging.predict(Xtest)
    # acc_pruned = accuracy_score(ytest, y_pred_2)
    # print(f'Testing accuracy after pruning the final model = {acc_pruned * 100: .2f}%')
    