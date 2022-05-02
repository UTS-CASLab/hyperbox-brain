# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0


import numpy as np
from sklearn.datasets import make_classification
from hbbrain.numerical_data.incremental_learner.iol_gfmm import (
    ImprovedOnlineGFMM
)
from hbbrain.numerical_data.ensemble_learner.random_hyperboxes import (
    RandomHyperboxesClassifier
)
from hbbrain.utils.model_storage import (
    store_model,
    load_model
)


def test_store_and_load_model():
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    clf = RandomHyperboxesClassifier(base_estimator=ImprovedOnlineGFMM(0.1),
                                     n_estimators=10, random_state=0).fit(X, y)
    # store model
    store_model(clf, "test_store_model.dump")
    # load model
    clf_load = load_model("test_store_model.dump")
    # check values
    assert clf.n_estimators == clf_load.n_estimators
    assert len(clf.estimators_) == len(clf_load.estimators_)
    assert clf.n_estimators == len(clf.estimators_)
    assert clf_load.n_estimators == len(clf_load.estimators_)
    assert clf.max_features == clf_load.max_features
    assert clf.max_samples == clf_load.max_samples
    assert clf.estimators_[0].theta == clf_load.estimators_[0].theta
    assert (clf.estimators_[0].V == clf_load.estimators_[0].V).all()
    assert (clf.estimators_[0].W == clf_load.estimators_[0].W).all()
    assert (clf.estimators_[0].C == clf_load.estimators_[0].C).all()
    sample_0_clf = np.array(clf.estimators_samples_[0])
    sample_0_clf_load = np.array(clf_load.estimators_samples_[0])
    assert (sample_0_clf == sample_0_clf_load).all()
