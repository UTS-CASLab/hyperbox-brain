hyperbox-brain documentation
============================

.. image:: _static/logo.png
   :height: 150px
   :width: 150px
   :align: center
   :target: https://uts-caslab.github.io/hyperbox-brain/

.. raw:: html

   <H1 align="center">Hyperbox-Brain</H1>

A scikit-learn compatible hyperbox-base machine learning library in Python.

Introduction
------------
**hyperbox-brain** is a Python open source toolbox implementing hyperbox-based machine learning algorithms built on top of
scikit-learn and is distributed under the 3-Clause BSD license.

The project was started in 2018 by Prof. `Bogdan Gabrys <https://profiles.uts.edu.au/Bogdan.Gabrys>`_ and Dr. `Thanh Tung Khuat <https://thanhtung09t2.wixsite.com/home>`_ at the Complex Adaptive Systems Lab - The
University of Technology Sydney. This project is a core module aiming to the formulation of explainable life-long learning 
systems in near future.

If you use hyperbox-brain, please use this BibTeX entry:

.. code:: bibtex

   @article{khgb22,
     author       = {Thanh Tung Khuat and Bogdan Gabrys},
     title        = {Hyperbox-brain: A Python Toolbox for Hyperbox-based Machine Learning Algorithms},
     journal      = {ArXiv},
     pages        = {1-7},
     year         = 2022,
     url          = {https://hyperbox-brain.readthedocs.io/en/latest/}
   }

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   user/installation
   user/features
   user/available_models
   user/quickstart
   developers/contributing
   user/about


API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 4
   :caption: Hyperbox-brain API

   api/utils
   api/base
   api/mixed_data
   api/batch_learner
   api/ensemble_learner
   api/incremental_learner
   api/multigranular_learner


.. toctree::
   :titlesonly:
   :caption: Tutorials

   tutorials/tutorial_index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
