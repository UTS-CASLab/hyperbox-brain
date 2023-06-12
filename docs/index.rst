hyperbox-brain documentation
============================

.. image:: _static/logo.png
   :height: 150px
   :width: 150px
   :align: center
   :target: https://uts-caslab.github.io/hyperbox-brain/

.. raw:: html

   <H1 align="center">Hyperbox-Brain</H1>

A scikit-learn compatible hyperbox-based machine learning library in Python.

Introduction
------------
**hyperbox-brain** is a Python open source toolbox implementing hyperbox-based machine learning algorithms built on top of
scikit-learn and is distributed under the GPL-3.0 license.

The project was started in 2018 by Prof. `Bogdan Gabrys <https://profiles.uts.edu.au/Bogdan.Gabrys>`_ and Dr. `Thanh Tung Khuat <https://thanhtung09t2.wixsite.com/home>`_ at the Complex Adaptive Systems Lab - The
University of Technology Sydney. This project is a core module aiming to the formulation of explainable life-long learning 
systems in near future.

If you use hyperbox-brain, please use this BibTeX entry:

.. code:: bibtex

   @article{khga23,
      title={hyperbox-brain: A Python toolbox for hyperbox-based machine learning algorithms},
      author={Khuat, Thanh Tung and Gabrys, Bogdan},
      journal={SoftwareX},
      volume={23},
      pages={101425},
      year={2023},
      url={https://doi.org/10.1016/j.softx.2023.101425},
      publisher={Elsevier}
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
