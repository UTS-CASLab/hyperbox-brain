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
training samples or a subset of both training samples and features. Training subsets of base learners 
can be formed by stratified random subsampling, resampling, or class-balanced random subsampling. 
The final predicted results of an ensemble model are an aggregation of predictions from all base learners 
based on a majority voting mechanism. An intersting characteristic of hyperbox-based models is resulting 
hyperboxes from all base learners can be merged to formulate a single model. This contributes to increasing 
the explainability of the estimator while still taking advantage of strong points of ensemble models.

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

.. list-table::
   :widths: 20 10 10 10 30 10 10
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
     - Instance-incremental 
     - `ExtendedImprovedOnlineGFMM </hbbrain/mixed_data/eiol_gfmm.py>`_
     - `Notebook </examples/mixed_data/eiol_gfmm_general_use.ipynb>`_
     - [1]_
   * - Freq-Cat-Onln-GFMM 
     - Mixed 
     - Single 
     - Batch-incremental 
     - `FreqCatOnlineGFMM </hbbrain/mixed_data/freq_cat_onln_gfmm.py>`_
     - `Notebook </examples/mixed_data/freq_cat_onln_gfmm_general_use.ipynb>`_
     - [2]_
   * - OneHot-Onln-GFMM 
     - Mixed 
     - Single 
     - Batch-incremental 
     - `OneHotOnlineGFMM </hbbrain/mixed_data/onehot_onln_gfmm.py>`_
     - `Notebook </examples/mixed_data/onehot_onln_gfmm_general_use.ipynb>`_
     - [2]_
   * - Onln-GFMM 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `OnlineGFMM </hbbrain/numerical_data/incremental_learner/onln_gfmm.py>`_
     - `Notebook </examples/numerical_data/incremental_learner/onln_gfmm_general_use.ipynb>`_
     - [3]_, [4]_
   * - IOL-GFMM 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `ImprovedOnlineGFMM </hbbrain/numerical_data/incremental_learner/iol_gfmm.py>`_
     - `Notebook </examples/numerical_data/incremental_learner/iol_gfmm_general_use.ipynb>`_
     - [5]_, [4]_
   * - FMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `FMNNClassifier </hbbrain/numerical_data/incremental_learner/fmnn.py>`_
     - `Notebook </examples/numerical_data/incremental_learner/fmnn_general_use.ipynb>`_
     - [6]_
   * - EFMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `EFMNNClassifier </hbbrain/numerical_data/incremental_learner/efmnn.py>`_
     - `Notebook </examples/numerical_data/incremental_learner/efmnn_general_use.ipynb>`_
     - [7]_ 
   * - KNEFMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `KNEFMNNClassifier </hbbrain/numerical_data/incremental_learner/knefmnn.py>`_
     - `Notebook </examples/numerical_data/incremental_learner/knefmnn_general_use.ipynb>`_
     - [8]_ 
   * - RFMNN 
     - Continuous 
     - Single 
     - Instance-incremental 
     - `RFMNNClassifier </hbbrain/numerical_data/incremental_learner/rfmnn.py>`_
     - `Notebook </examples/numerical_data/incremental_learner/rfmnn_general_use.ipynb>`_
     - [9]_ 
   * - AGGLO-SM 
     - Continuous 
     - Single 
     - Batch 
     - `AgglomerativeLearningGFMM </hbbrain/numerical_data/batch_learner/agglo_gfmm.py>`_
     - `Notebook </examples/numerical_data/batch_learner/agglo_gfmm_general_use.ipynb>`_
     - [10]_ 
   * - AGGLO-2
     - Continuous 
     - Single 
     - Batch
     - `AccelAgglomerativeLearningGFMM </hbbrain/numerical_data/batch_learner/accel_agglo_gfmm.py>`_
     - `Notebook </examples/numerical_data/batch_learner/accel_agglo_gfmm_general_use.ipynb>`_
     - [10]_
   * - MRHGRC
     - Continuous 
     - Granularity 
     - Multi-Granular learning 
     - `MultiGranularGFMM </hbbrain/numerical_data/multigranular_learner/multi_resolution_gfmm.py>`_
     - `Notebook </examples/numerical_data/multigranular_learner/multi_resolution_gfmm_general_use.ipynb>`_
     - [11]_ 
   * - Decision-level Bagging of hyperbox-based learners
     - Continuous 
     - Combination 
     - Ensemble 
     - `DecisionCombinationBagging </hbbrain/numerical_data/ensemble_learner/decision_comb_bagging.py>`_
     - `Notebook </examples/numerical_data/ensemble_learner/decision_comb_bagging_general_use.ipynb>`_
     - [12]_
   * - Decision-level Bagging of hyperbox-based learners with hyper-parameter optimisation
     - Continuous
     - Combination 
     - Ensemble 
     - `DecisionCombinationCrossValBagging </hbbrain/numerical_data/ensemble_learner/decision_comb_cross_val_bagging.py>`_
     - `Notebook </examples/numerical_data/ensemble_learner/decision_comb_cross_val_bagging_general_use.ipynb>`_
     -  
   * - Model-level Bagging of hyperbox-based learners
     - Continuous 
     - Combination 
     - Ensemble 
     - `ModelCombinationBagging </hbbrain/numerical_data/ensemble_learner/model_comb_bagging.py>`_
     - `Notebook </examples/numerical_data/ensemble_learner/model_comb_bagging_general_use.ipynb>`_
     - [12]_
   * - Model-level Bagging of hyperbox-based learners with hyper-parameter optimisation 
     - Continuous 
     - Combination 
     - Ensemble 
     - `ModelCombinationCrossValBagging </hbbrain/numerical_data/ensemble_learner/model_comb_cross_val_bagging.py>`_
     - `Notebook </examples/numerical_data/ensemble_learner/model_comb_cross_val_bagging_general_use.ipynb>`_
     -   
   * - Random hyperboxes 
     - Continuous 
     - Combination 
     - Ensemble 
     - `RandomHyperboxesClassifier </hbbrain/numerical_data/ensemble_learner/random_hyperboxes.py>`_
     - `Notebook </examples/numerical_data/ensemble_learner/random_hyperboxes_general_use.ipynb>`_
     - [13]_
   * - Random hyperboxes with hyper-parameter optimisation for base learners 
     - Continuous 
     - Combination 
     - Ensemble 
     - `CrossValRandomHyperboxesClassifier </hbbrain/numerical_data/ensemble_learner/cross_val_random_hyperboxes.py>`_
     - `Notebook </examples/numerical_data/ensemble_learner/cross_val_random_hyperboxes_general_use.ipynb>`_
     -  

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
.. [2] : T. T. Khuat and B. Gabrys "`An in-depth comparison of methods handling mixed-attribute data for general fuzzy min–max neural network <https://doi.org/10.1016/j.neucom.2021.08.083>`_", Neurocomputing, vol 464, pp. 175-202, 2021.
.. [3] : B. Gabrys and A. Bargiela, "`General fuzzy min-max neural network for clustering and classification <https://doi.org/10.1109/72.846747>`_", IEEE transactions on neural networks, vol. 11, no. 3, pp. 769-783, 2000.
.. [4] : T. T. Khuat and B. Gabrys, "`Accelerated learning algorithms of general fuzzy min-max neural network using a novel hyperbox selection rule <https://doi.org/10.1016/j.ins.2020.08.046>`_", Information Sciences, vol. 547, pp. 887-909, 2021.
.. [5] : T. T. Khuat, F. Chen, and B. Gabrys, "`An improved online learning algorithm for general fuzzy min-max neural network <https://doi.org/10.1109/IJCNN48605.2020.9207534>`_", in Proceedings of the International Joint Conference on Neural Networks (IJCNN), pp. 1-9, 2020.
.. [6] : P. Simpson, "`Fuzzy min—max neural networks—Part 1: Classiﬁcation <https://doi.org/10.1109/72.159066>`_", IEEE transactions on neural networks, vol. 3, no. 5, pp. 776-786, 1992.
.. [7] : M. Mohammed and C. P. Lim, "`An enhanced fuzzy min-max neural network for pattern classification <https://doi.org/10.1109/TNNLS.2014.2315214>`_", IEEE Transactions on Neural Networks and Learning Systems, vol. 26, no. 3, pp. 417-429, 2014.
.. [8] : M. Mohammed and C. P. Lim, "`Improving the Fuzzy Min-Max neural network with a k-nearest hyperbox expansion rule for pattern classification <https://doi.org/10.1016/j.asoc.2016.12.001>`_", Applied Soft Computing, vol. 52, pp. 135-145, 2017.
.. [9] : O. N. Al-Sayaydeh, M. F. Mohammed, E. Alhroob, H. Tao, and C. P. Lim, "`A refined fuzzy min-max neural network with new learning procedures for pattern classification <https://doi.org/10.1109/TFUZZ.2019.2939975>`_", IEEE Transactions on Fuzzy Systems, vol. 28, no. 10, pp. 2480-2494, 2019.
.. [10] : B. Gabrys, "`Agglomerative learning algorithms for general fuzzy min-max neural network <https://link.springer.com/article/10.1023/A:1016315401940>`_", Journal of VLSI signal processing systems for signal, image and video technology, vol. 32, no. 1, pp. 67-82, 2002.
.. [11] : T.T. Khuat, F. Chen, and B. Gabrys, "`An Effective Multiresolution Hierarchical Granular Representation Based Classifier Using General Fuzzy Min-Max Neural Network <https://doi.org/10.1109/TFUZZ.2019.2956917>`_", IEEE Transactions on Fuzzy Systems, vol. 29, no. 2, pp. 427-441, 2021.
.. [12] : B. Gabrys, "`Combining neuro-fuzzy classifiers for improved generalisation and reliability <https://doi.org/10.1109/IJCNN.2002.1007519>`_", in Proceedings of the 2002 International Joint Conference on Neural Networks, vol. 3, pp. 2410-2415, 2002.
.. [13] : T. T. Khuat and B. Gabrys, "`Random Hyperboxes <https://doi.org/10.1109/TNNLS.2021.3104896>`_", IEEE Transactions on Neural Networks and Learning Systems, 2021.