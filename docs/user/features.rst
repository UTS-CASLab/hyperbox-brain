========
Features
========

The **hyper-box brain** toolbox has the following main characteristics:

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

Explainability of predicted results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The hyperbox-brain library can provide the explanation of predicted results via visualisation. 
This toolbox provides the visualisation of existing hyperboxes and the decision boundaries of 
a trained hyperbox-based model if input features are two-dimensional features:

.. image:: ../../_static/hyperboxes_and_boundaries.png
   :height: 300 px
   :width: 350 px
   :alt: Hyperboxes and Decision Boundaries
   :align: center

For two-dimensional data, the toolbox also provides the reason behind the class prediction for each input sample 
by showing representative hyperboxes for each class which join the prediction process of the trained model for 
an given input pattern:

.. image:: ../../_static/hyperboxes_explanation.png
   :height: 300 px
   :width: 350 px
   :alt: 2D explainations
   :align: center

For input patterns with two or more dimensions, the hyperbox-brain toolbox uses a parallel coordinates graph to display 
representative hyperboxes for each class which join the prediction process of the trained model for 
an given input pattern:

.. image:: ../../_static/parallel_coord_explanation.PNG
   :height: 300 px
   :width: 500 px
   :alt: Parallel coordinates explainations
   :align: center

Easy to use
~~~~~~~~~~~
Hyperbox-brain is designed for users with any experience level. Learning models are easy to create, setup, and run. Existing methods are easy to modify and extend.

Jupyter notebooks
~~~~~~~~~~~~~~~~~
The learning models in the hyperbox-brain toolbox can be easily retrieved in notebooks in the Jupyter or JupyterLab environments.

In order to display plots from hyperbox-brain within a `Jupyter Notebook <https://jupyter-notebook.readthedocs.io/en/latest/>`_ we need to define the proper mathplotlib
backend to use. This can be performed by including the following magic command at the beginning of the Notebook:

.. code:: bash

    %matplotlib notebook

`JupyterLab <https://github.com/jupyterlab/jupyterlab>`_ is the next-generation user interface for Jupyter, and it may display interactive plots with some caveats.
If you use JupyterLab then the current solution is to use the `jupyter-matplotlib <https://github.com/matplotlib/ipympl>`_ extension:

.. code:: bash

    %matplotlib widget

`Examples <https://github.com/UTS-CASLab/hyperbox-brain/tree/main/examples>`_ regarding how to use the classes and functions in the hyperbox-brain toolbox have been
written under the form of Jupyter notebooks.
