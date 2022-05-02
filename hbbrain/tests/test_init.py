# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

# Basic unittests to test functioning of module's top-level


__author__ = "Thanh Tung Khuat"
__license__ = "GPL-3.0"


try:
    from hbbrain import *
    _top_import_error = None
except Exception as e:
    _top_import_error = e

try:
    from hbbrain.mixed_data import *
    _mixed_data_import_error = None
except Exception as e:
    _mixed_data_import_error = e

try:
    from hbbrain.numerical_data.batch_learner import *
    _batch_learner_import_error = None
except Exception as e:
    _batch_learner_import_error = e

try:
    from hbbrain.numerical_data.ensemble_learner import *
    _ensemble_learner_import_error = None
except Exception as e:
    _ensemble_learner_import_error = e

try:
    from hbbrain.numerical_data.incremental_learner import *
    _incremental_learner_import_error = None
except Exception as e:
    _incremental_learner_import_error = e

try:
    from hbbrain.numerical_data.multigranular_learner import *
    _multigranular_learner_import_error = None
except Exception as e:
    _multigranular_learner_import_error = e

try:
    from hbbrain.utils import *
    _utils_import_error = None
except Exception as e:
    _utils_import_error = e


def test_import_hbbrain():
    # Test either above import has failed for some reason
    # "import *" is discouraged outside of the module level, hence we
    # rely on setting up the variable above
    assert _top_import_error is None


def test_import_mixed_data_learner():
    assert _mixed_data_import_error is None


def test_import_batch_learner():
    assert _batch_learner_import_error is None


def test_import_ensemble_learner():
    assert _ensemble_learner_import_error is None


def test_import_incremental_learner():
    assert _incremental_learner_import_error is None


def test_import_multigranular_learner():
    assert _multigranular_learner_import_error is None


def test_import_utils():
    assert _utils_import_error is None
