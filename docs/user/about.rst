====================
About hyperbox-brain
====================

``hyperbox-brain`` is an open-source machine learning package in Python for
hyperbox-based machine learning algorithms. Learning algorithms using hyperboxes
as fundamental representational and building blocks are a branch of machine
learning methods. These algorithms have enormous potential for high scalability
and online adaptation of predictors built using hyperbox data representations to
the dynamically changing environments. This library focuses on developing and
extending the learning algorithms for a specific type of universal hyperbox-based
classifiers, i.e., fuzzy min-max neural networks and general fuzzy min-max neural
network.

Hyperboxes can be used to deal with the pattern classification and clustering problems
effectively by partitioning the pattern space and assigning a class label or cluster
associated with a degree of certainty for each region. Each fuzzy min-max hyperbox is
represented by minimum and maximum points together with a fuzzy membership function.
The membership function is employed to compute the degree-of-fit of each input sample
to a given hyperbox. Meanwhile, the hyperboxes are continuously adjusted during the
training process to cover the input patterns. The use of hyperboxes for learning
systems can form a core module aiming to build smart adaptive systems and life-long
learning systems in the near future.

Ecosystem
---------
``hyperbox-brain`` is part of the hyperbox-based machine learning ecosystem. In
Python, this library can be used together with pipeline and hyper-parameter optimisers
in the `scikit-learn <https://scikit-learn.org/>`_ library. This library can be also
compatible with other optimisers in Python such as `hyperopt <http://hyperopt.github.io/hyperopt/>`_
and `Optuna <https://optuna.org/>`_.

Development team 
----------------
This library is the result of hyperbox-based machine learning project conducted by
the Complex Adaptive Systems in the `University of Technology Sydney <https://uts.edu.au/>`_.
Current members of the development team (in alphabetical order):

* Prof. Bogdan Gabrys
* Dr. Thanh Tung Khuat

We also acknowledge the `individual members <https://github.com/UTS-CASLab/hyperbox-brain/graphs/contributors>`_
of the open-source community who have contributed to this project.

Citing 
------
If ``hyperbox-brain`` has been useful for your research and you would like to cite it in
an academic publication, please use the following paper::
    @article{khgb22,
        author  = {Thanh Tung Khuat and Bogdan Gabrys},
        title   = {Hyperbox-brain: A Python Toolbox for Hyperbox-based Machine Learning Algorithms},
        journal = {ArXiv},
        year    = {2022},
        volume  = {},
        number  = {0},
        pages   = {1-7},
        url     = {}
    }

Logo 
----
The ``hyperbox-brain`` logo is designed by Thanh Tung Khuat.
