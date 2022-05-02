# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("numerical_data", parent_package, top_path)
    config.add_subpackage("batch_learner")
    config.add_subpackage("ensemble_learner")
    config.add_subpackage("incremental_learner")
    config.add_subpackage("multigranular_learner")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
