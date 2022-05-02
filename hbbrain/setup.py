# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import os


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config = Configuration("hbbrain", parent_package, top_path)

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage("base")
    config.add_subpackage("numerical_data")
    config.add_subpackage("mixed_data")
    config.add_subpackage("utils")
    # config.add_subpackage("numerical_data/batch_learner")
    # config.add_subpackage("numerical_data/ensemble_learner")
    # config.add_subpackage("numerical_data/incremental_learner")
    # config.add_subpackage("numerical_data/multigranular_learner")

    # submodules which have their own setup.py

    # add the test directory
    config.add_subpackage("tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
