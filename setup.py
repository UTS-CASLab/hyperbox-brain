#! /usr/bin/env python
#
# Copyright (C) 2021 Thanh Tung KHUAT <thanhtung09t2@gmai.com>
# License: GPL-3.0

import sys
import os
import platform
import shutil

# We need to import setuptools before because it monkey-patches distutils
import setuptools  # noqa
from distutils.command.clean import clean as Clean
from distutils.command.sdist import sdist

import traceback
import importlib

try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.7 is needed.
    import __builtin__ as builtins

# We are setting a global variable so that the
# main hbbrain __init__ can detect if it is being loaded by the setup
# routine, to avoid attempting to load components that aren't built yet:
# the numpy distutils extensions that are used by hyperbox-brain to
# recursively build the compiled extensions in sub-packages is based on the
# Python import machinery.
builtins.__HBBRAIN_SETUP__ = True


DISTNAME = "hyperbox-brain"
DESCRIPTION = "A scikit-learn compatible hyperbox-based machine learning library in Python"

here = os.path.abspath(os.path.dirname(__file__))
try:
    LONG_DESCRIPTION = open(os.path.join(here, 'README.rst'), encoding="utf-8").read()
except IOError:
    LONG_DESCRIPTION = ''

MAINTAINER = "Thanh Tung KHUAT"
MAINTAINER_EMAIL = "thanhtung09t2@gmail.com"
URL = "https://uts-caslab.github.io/hyperbox-brain/"
DOWNLOAD_URL = "https://pypi.org/project/hyperbox-brain/#files"
LICENSE = "GPLv3"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/UTS-CASLab/hyperbox-brain/issues",
    "Documentation": "https://hyperbox-brain.readthedocs.io/en/stable/",
    "Source Code": "https://github.com/UTS-CASLab/hyperbox-brain",
}

# We can actually import a restricted version of sklearn that
# does not need the compiled code
import hbbrain  # noqa
import hbbrain._min_dependencies as min_deps  # noqa
from hbbrain._check_version import parse as parse_version  # noqa

VERSION = hbbrain.__version__

# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    "develop",
    "release",
    "bdist_egg",
    "bdist_rpm",
    "bdist_wininst",
    "install_egg_info",
    "build_sphinx",
    "egg_info",
    "easy_install",
    "upload",
    "bdist_wheel",
    "--single-version-externally-managed",
}

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        extras_require={
            key: min_deps.tag_to_packages[key]
            for key in ["examples", "docs", "tests"]
        },
    )
else:
    extra_setuptools_args = dict()

# Custom clean command to remove build artifacts


class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("hhbrain"):
            for filename in filenames:
                if any(
                    filename.endswith(suffix)
                    for suffix in (".so", ".pyd", ".dll", ".pyc")
                ):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {"clean": CleanCommand, "sdist": sdist}

# Optional wheelhouse-uploader features
# To automate release of binary packages for hyperbox-brain we need a tool
# to download the packages generated by travis and appveyor workers (with
# version number matching the current release) and upload them all at once
# to PyPI at release time.
# The URL of the artifact repositories are configured in the setup.cfg file.

WHEELHOUSE_UPLOADER_COMMANDS = {"fetch_artifacts", "upload_all"}
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader.cmd

    cmdclass.update(vars(wheelhouse_uploader.cmd))


def configuration(parent_package="", top_path=None):
    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")

    from numpy.distutils.misc_util import Configuration
    
    config = Configuration(None, parent_package, top_path)

    # Avoid useless msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage("hbbrain")

    return config

def check_package_status(package, min_version):
    """
    Returns a dictionary containing a boolean specifying whether given package
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = parse_version(package_version) >= parse_version(
            min_version
        )
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "hyperbox-brain requires {} >= {}.\n".format(package, min_version)

    instructions = (
        "Installation instructions are available on the "
        "hyperbox-brain website: "
        "https://hyperbox-brain.readthedocs.io/en/latest/user/installation.html\n"
    )

    if package_status["up_to_date"] is False:
        if package_status["version"]:
            raise ImportError(
                "Your installation of {} {} is out-of-date.\n{}{}".format(
                    package, package_status["version"], req_str, instructions
                )
            )
        else:
            raise ImportError(
                "{} is not installed.\n{}{}".format(package, req_str, instructions)
            )


def setup_package():
    # These commands use setup from setuptools
    from setuptools import find_packages

    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/x-rst',
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 5 - Production/Stable",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
			"Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: Implementation :: PyPy",
        ],
        cmdclass=cmdclass,
        python_requires=">=3.6",
        install_requires=min_deps.tag_to_packages["install"],
        package_data={"": ["*.pxd"]},
        **extra_setuptools_args,
    )

    commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if all(
        command in ("egg_info", "dist_info", "clean", "check") for command in commands
    ):
        # These actions are required to succeed without Numpy for example when
        # pip is used to install hyperbox-brain when Numpy is not yet present in
        # the system.
        from setuptools import setup

        metadata["version"] = VERSION
    else:
        if sys.version_info < (3, 6):
            raise RuntimeError(
                "Hyperbox-brain requires Python 3.6 or later. The current"
                " Python version is %s installed in %s."
                % (platform.python_version(), sys.executable)
            )

        check_package_status("numpy", min_deps.NUMPY_MIN_VERSION)

        check_package_status("scipy", min_deps.SCIPY_MIN_VERSION)

        check_package_status("sklearn", min_deps.SKLEARN_MIN_VERSION)

        check_package_status("pandas", min_deps.PANDAS_MIN_VERSION)

        check_package_status("joblib", min_deps.JOBLIB_MIN_VERSION)

        check_package_status("matplotlib", min_deps.MATPLOTLIB_MIN_VERSION)

        # These commands require the setup from numpy.distutils because they
        # may use numpy.distutils compiler classes.
        from numpy.distutils.core import setup

        metadata["configuration"] = configuration

    setup(**metadata, packages=find_packages())


if __name__ == "__main__":
    setup_package()
