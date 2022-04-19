.. -*- mode: rst -*-
.. |PythonMinVersion| replace:: 3.6
.. |NumPyMinVersion| replace:: 1.14.6
.. |SciPyMinVersion| replace:: 1.1.0
.. |JoblibMinVersion| replace:: 0.11
.. |ThreadpoolctlMinVersion| replace:: 2.0.0
.. |MatplotlibMinVersion| replace:: 2.2.3
.. |Scikit-ImageMinVersion| replace:: 0.14.5
.. |SklearnMinVersion| replace:: 0.24.0
.. |PandasMinVersion| replace:: 0.25.0
.. |PlotlyMinVersion| replace:: 4.10.0
.. |PytestMinVersion| replace:: 5.0.1

.. raw:: html
   
   <p align="center"><img width="150" height="150" src="/images/logo.png" alt="Hyperbox-Brain logo"/></p>
   <H1 align="center">Hyperbox-Brain</H1>

**hyperbox-brain** is a Python open source library implementing hyperbox-based machine learning algorithms built on top of
scikit-learn and is distributed under the 3-Clause BSD license.

The project was started in 2018 by Prof. Bogdan Gabrys and Dr. Thanh Tung Khuat at the Complex Adaptive Systems Lab - The
University of Technology Sydney. This project is a core module aiming to the formulation of explainable life-long learning 
systems in near future.

============
Installation
============

Dependencies
~~~~~~~~~~~~

Hyperbox-brain requires:

- Python (>= |PythonMinVersion|)
- Scikit-learn (>= |SklearnMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)
- Pandas (>= |PandasMinVersion|)

=======

Hyperbox-brain plotting capabilities (i.e., functions start with ``show_`` or ``draw_``) 
require Matplotlib (>= |MatplotlibMinVersion|) and Plotly (>= |PlotlyMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| and Plotly >= |PlotlyMinVersion| are required.
A few examples require pandas >= |PandasMinVersion|.

conda installation
==================

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

To install hyperbox-brain, you need to use the conda-forge channel:

.. code:: bash

    conda install -c conda-forge hyperbox-brain

We recommend to use a `conda virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

pip installation
================
If you already have a working installation of numpy, scipy, pandas, matplotlib,
and scikit-learn, the easiest way to install hyperbox-brain is using ``pip``:

.. code:: bash

    pip install -U hyperbox-brain

Again, we recommend to use a `virtual environment
<https://docs.python.org/3/tutorial/venv.html>`_ for this.

From source
===========

If you would like to use the most recent additions to hyperbox-brain or
help development, you should install hyperbox-brain from source.

Using conda
-----------

To install hyperbox-brain from source using conda, proceed as follows:

.. code:: bash

    git clone https://github.com/UTS-CASLab/hyperbox-brain.git
    cd hyperbox-brain
    conda env create
    source activate hyperbox-brain
    pip install .

Using pip
---------

For pip, follow these instructions instead:

.. code:: bash

    git clone https://github.com/UTS-CASLab/hyperbox-brain.git
    cd hyperbox-brain
    # create and activate a virtual environment
    pip install -r requirements.txt
    # install hyperbox-brain version for your system (see below)
    pip install .