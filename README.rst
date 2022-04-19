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

**hyperbox-brain** is a Python open source toolbox implementing hyperbox-based machine learning algorithms built on top of
scikit-learn and is distributed under the 3-Clause BSD license.

The project was started in 2018 by Prof. `Bogdan Gabrys <https://profiles.uts.edu.au/Bogdan.Gabrys>`_ and Dr. `Thanh Tung Khuat <https://thanhtung09t2.wixsite.com/home>`_ at the Complex Adaptive Systems Lab - The
University of Technology Sydney. This project is a core module aiming to the formulation of explainable life-long learning 
systems in near future.

=========
Resources
=========

- `Documentation <https://hyperbox-brain.readthedocs.io/en/latest>`_
- `Source Code <https://github.com/UTS-CASLab/hyperbox-brain/>`_
- `Installation <https://github.com/UTS-CASLab/hyperbox-brain#installation>`_
- `Issue tracker <https://github.com/UTS-CASLab/hyperbox-brain/issues>`_
- `Examples <https://github.com/UTS-CASLab/hyperbox-brain/tree/main/examples>`_

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
~~~~~~~~~~~~~~~~~~

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

To install hyperbox-brain, you need to use the conda-forge channel:

.. code:: bash

    conda install -c conda-forge hyperbox-brain

We recommend to use a `conda virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

pip installation
~~~~~~~~~~~~~~~~

If you already have a working installation of numpy, scipy, pandas, matplotlib,
and scikit-learn, the easiest way to install hyperbox-brain is using ``pip``:

.. code:: bash

    pip install -U hyperbox-brain

Again, we recommend to use a `virtual environment
<https://docs.python.org/3/tutorial/venv.html>`_ for this.

From source
~~~~~~~~~~~

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

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have ``pytest`` >= |PyTestMinVersion| installed):

.. code:: bash

    pytest hbbrain

========
Features
========

Types of input variables
~~~~~~~~~~~~~~~~~~~~~~~~
The hyperbox-brain library separates learning models for continuous variables only
and mixed-attribute data.

Incremental learning
~~~~~~~~~~~~~~~~~~~~
Incremental (online) learning models are created incrementally and are updated continuously.
They are appropriate for big data applications where real-time response is an important requirement.
These learning models generate a new hyperbox or expand an existing hyperbox to cover each incoming
input pattern.

Agglomerative learning
~~~~~~~~~~~~~~~~~~~~~~
Agglomerative (batch) learning models are trained using all training data available at the
training time. They use the aggregation of existing hyperboxes to form new larger sized hyperboxes 
based on the similarity measures among hyperboxes.

Ensemble learning
~~~~~~~~~~~~~~~~~
Ensemble models in the hyperbox-brain toolbox build a set of hyperbox-based learners from a subset of
training samples or a subset of both training samples and features. The final predicted results of an 
ensemble model are an aggregation of predictions from all base learners based on a majority voting mechanism.
An intersting characteristic of hyperbox-based models is resulting hyperboxes from all base learners can 
be merged to formulate a single model. This contributes to increasing the explainability of the estimator while 
still taking advantage of strong points of ensemble models.

Multigranularity learning
~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-granularity learning algorithms can construct classifiers from multiresolution hierarchical granular representations 
using hyperbox fuzzy sets. This algorithm forms a series of granular inferences hierarchically through many levels of 
abstraction. An attractive characteristic of these classifiers is that they can maintain a high accuracy in comparison 
to other fuzzy min-max models at a low degree of granularity based on reusing the knowledge learned from lower levels 
of abstraction.

Scikit-learn compatible estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The estimators in hyperbox-brain is compatible with the well-known scikit-learn toolbox. 
Therefore, it is possible to use hyperbox-based estimators in scikit-learn `pipelines <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_, 
scikit-learn hyperparameter optimizers (e.g., `grid search <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_ 
and `random search <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_), 
and scikit-learn model validation (e.g., `cross-validation scores <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>`_). 
In addition, the hyperbox-brain toolbox can be used within hyperparameter optimisation libraries built on top of 
scikit-learn such as `hyperopt <http://hyperopt.github.io/hyperopt/>`_.

Explanability of predicted results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The hyperbox-brain library can provide the explanation of predicted results via visualisation. 
This toolbox provides the visualisation of existing hyperboxes and the decision boundaries of 
a trained hyperbox-based model if input features are two-dimensional features:

.. raw:: html
   
   <p align="left"><img width="350" height="300" src="/images/hyperboxes_and_boundaries.png" alt="Hyperboxes and Decision Boundaries"/></p>

For two-dimensional data, the toolbox also provides the reason behind the class prediction for each input sample 
by showing representative hyperboxes for each class which join the prediction process of the trained model for 
an given input pattern:

.. raw:: html
   
   <p align="left"><img width="350" height="300" src="/images/hyperboxes_explanation.png" alt="2D explainations"/></p>

For input patterns with two or more dimensions, the hyperbox-brain toolbox uses a parallel coordinates graph to display 
representative hyperboxes for each class which join the prediction process of the trained model for 
an given input pattern:

.. raw:: html
   
   <p align="left"><img width="500" height="300" src="/images/parallel_coord_explanation.PNG" alt="2D explainations"/></p>

Easy to use
~~~~~~~~~~~
Hyperbox-brain is designed for users with any experience level. Learning models are easy to create, setup, and run. Existing methods are easy to modify and extend.

Jupyter notebooks
~~~~~~~~~~~~~~~~~
The learning models in the hyperbox-brain toolbox can be easily retrieved in 
notebooks in the Jupyter or JupyterLab environments.

In order to display plots from hyperbox-brain within a `Jupyter Notebook <https://jupyter-notebook.readthedocs.io/en/latest/>`_ we need to define the proper mathplotlib
backend to use. This can be performed by including the following magic command at the beginning of the Notebook:

.. code:: bash

    %matplotlib notebook

`JupyterLab <https://github.com/jupyterlab/jupyterlab>`_ is the next-generation user interface for Jupyter, and it may display interactive plots with some caveats.
If you use JupyterLab then the current solution is to use the `jupyter-matplotlib <https://github.com/matplotlib/ipympl>`_ extension:

.. code:: bash

    %matplotlib widget

`Examples <https://github.com/UTS-CASLab/hyperbox-brain/tree/main/examples>`_ regarding how to use the classes and functions in the hyperbox-brain toolbox have been written under the form of Jupyter notebooks.

================
Available models
================
The following table summarises the supported hyperbox-based learning algorithms in this toolbox.

.. list-table:: **Hyperbox-based learning models**
   :widths: 15 10 10 10 30 15 10
   :align: center
   :header-rows: 1

   * - Model
     - Feature type 
     - Model type
     - Learning type 
     - Implementation 
     - Example 
     - References 
   * - EIOL-GFMM
     - Mixed
     - Single 
     - Incremental 
     - `ExtendedImprovedOnlineGFMM <hbbrain/mixed_data/eiol_gfmm.py>`_
     - `Notebook </examples/mixed_data/eiol_gfmm_general_use.ipynb>`_
     - [1]_

========
Citation
========

If you use hyperbox-brain in a scientific publication, we would appreciate
citations to the following paper::

  @article{khuat2022,
  author  = {Thanh Tung Khuat and Bogdan Gabrys},
  title   = {Hyerbox-brain: A Python Toolbox for Hyperbox-based Machine Learning Algorithms},
  journal = {ArXiv},
  year    = {2022},
  volume  = {},
  number  = {0},
  pages   = {1-7},
  url     = {}
  }

============
Contributing
============
Feel free to contribute in any way you like, we're always open to new ideas and approaches.

There are some ways for users to get involved:

- `Issue tracker <https://github.com/UTS-CASLab/hyperbox-brain/issues>`_: this place is meant to report bugs, request for minor features, or small improvements. Issues should be short-lived and solved as fast as possible.
- `Discussions <https://github.com/UTS-CASLab/hyperbox-brain/discussions>`_: in this place, you can ask for new features, submit your questions and get help, propose new ideas, or even show the community what you are achieving with hyperbox-brain! If you have a new algorithm or want to port a new functionality to hyperbox-brain, this is the place to discuss.

=======
License
=======
Hyperbox-brain is free and open-source software licensed under the `GNU General Public License v3.0 <https://github.com/UTS-CASLab/hyperbox-brain/blob/main/LICENSE>`_.

==========
References
==========

.. [1] : T. T. Khuat and B. Gabrys "`An Online Learning Algorithm for a Neuro-Fuzzy Classifier with Mixed-Attribute Data <https://arxiv.org/abs/2009.14670>`_", ArXiv preprint, arXiv:2009.14670, 2020.