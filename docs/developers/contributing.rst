============
Contributing
============

We welcome contributions from the community. Here you will find information to start contributing to **hyperbox-brain**.

The project is hosted on https://github.com/UTS-CASLab/hyperbox-brain

.. topic:: **Our community, our values**

    We are a community based on openness and friendly, politeness, constructive discussions.

    We aspire to treat everybody equally, and value their contributions.  We
    are particularly seeking people from underrepresented backgrounds in Open
    Source Software and Hyperbox-based Machine Learning in particular to join and
    contribute their expertise and experience.

    Decisions are made based on technical merit and consensus.

    Code is not the only way to help the project. Reviewing pull
    requests, answering questions to help others on mailing lists or
    issues, organising and running tutorials, working on the website,
    enhancing the quality of documentation, are all priceless contributions.

    We abide by the principles of openness, respect, and consideration of
    others of the Python Software Foundation:
    https://www.python.org/psf/codeofconduct/

In case you experience issues using this package, do not hesitate to submit a
ticket to the `GitHub issue tracker
<https://github.com/UTS-CASLab/hyperbox-brain/issues>`_. You are also
welcome to post feature requests or pull requests.

Ways to contribute
==================

There are various methods to contribute to hyperbox-brain, with the most common ones
being contribution of code or documentation to the project. Enhancing the
documentation is no less important than enhancing the library itself.  If you
find a typo in the documentation, or have made improvements, do not hesitate to
send an email to the mailing list or preferably submit a GitHub pull request.
Full documentation can be found under the doc/ directory.

But there are many other ways to help. In particular helping to
`improve, triage, and investigate issues` and `reviewing other developers' pull requests` are very
valuable contributions that decrease the burden on the project maintainers.

Another way to contribute is to report issues you're facing, and give a "thumbs
up" on issues that others reported and that are relevant to you.  It also helps
us if you spread the word: reference the project from your blog and articles,
link to it from your website, or simply star to say "I use it":

In case a contribution/issue involves changes to the API principles
or changes to dependencies or supported versions, it must be essential to 
submit as a pull-request and send an email to inform the project owner.

Submitting a bug report or a feature request
============================================

We use GitHub issues to track all bugs and feature requests; feel free to open
an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a
ticket to the
`Bug Tracker <https://github.com/UTS-CASLab/hyperbox-brain/issues>`_. You are
also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/UTS-CASLab/hyperbox-brain/issues?q=>`_
   or `pull requests <https://github.com/UTS-CASLab/hyperbox-brain/pulls?q=>`_.

-  If you are submitting an algorithm or feature request, please verify the
   algorithm carefully and discuss it with the governance board.

-  If you are submitting a bug report, we strongly encourage you to follow the guidelines in :ref:`filing_bugs`.

.. _filing_bugs:

How to make a good bug report
-----------------------------

When you submit an issue to `Github
<https://github.com/UTS-CASLab/hyperbox-brain/issues>`__, please do your best to
follow these guidelines! This will make it a lot easier to provide you with good
feedback:

- The ideal bug report contains a description of how to reproduce this bug via code snippet.
  By doing this way, anyone can try to reproduce the bug easily (see `this
  <https://stackoverflow.com/help/mcve>`_ for more details). If your snippet is
  longer than around 50 lines, please link to a `gist
  <https://gist.github.com>`_ or a github repo.

- If it is not feasible to include a reproducible snippet, please be specific about
  what **estimators and/or functions are involved and the shape of the data**.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
  your **Python, hyperbox-brain, scikit-learn, joblib, numpy, matplotlib, plotly, and pandas versions**. This information
  can be found by running the following code snippet:

.. code:: python

    >>> import hbbrain
    >>> hbbrain.show_versions()

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**.  See `Creating and highlighting code blocks
  <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_
  for more details.

Contributing code
=================

.. note::
    To avoid duplicating work, it is highly recommended that you search through the
    `issue tracker <https://github.com/UTS-CASLab/hyperbox-brain/issues>`_ and
    the `PR list <https://github.com/UTS-CASLab/hyperbox-brain/pulls>`_.
    If in doubt about duplicated work, or if you want to work on a non-trivial
    feature, it's recommended to first open an issue in the
    `issue tracker <https://github.com/UTS-CASLab/hyperbox-brain/issues>`_
    to get some feedbacks from core developers.
    
    One easy way to find an issue to work on is by applying the "help wanted"
    label in your search. This lists all the issues that have been unclaimed
    so far. In order to claim an issue for yourself, please comment exactly
    ``/take`` on it to assign the issue to you.


How to contribute
-----------------

The best method to contribute to hyperbox-brain is to fork the `main
repository <https://github.com/UTS-CASLab/hyperbox-brain/>`__ on GitHub,
then submit a "pull request" (PR).

In the first few steps, we explain how to locally install hyperbox-brain, and
how to set up your git repository:

#. `Create an account <https://github.com/join>`_ on
   GitHub if you do not already have one.

#. Fork the `project repository
   <https://github.com/UTS-CASLab/hyperbox-brain>`__: click on the 'Fork'
   button near the top of the page. This creates a copy of the code under your
   account on the GitHub user account. For more details on how to fork a
   repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

#. Clone your fork of the hyperbox-brain repo from your GitHub account to your
   local disk:

   .. code:: bash
       
       git clone git@github.com:YourLogin/hyperbox-brain.git  # add --depth 1 if your connection is slow
       cd hyperbox-brain

#. Follow the steps in the `installation from source <https://hyperbox-brain.readthedocs.io/en/latest/user/installation.html#from-source>`_
   to build hyperbox-brain in development mode and return to this document.

#. Install the development dependencies:

   .. code:: bash
       
       pip install pytest pytest-cov flake8 mypy numpydoc black==22.3.0

#. Add the ``upstream`` remote. This saves a reference to the main
   scikit-learn repository, which you can use to keep your repository
   synchronized with the latest changes:

   .. code:: bash
       
       git remote add upstream git@github.com:UTS-CASLab/hyperbox-brain.git

#. Check that the `upstream` and `origin` remote aliases are configured correctly
   by running `git remote -v` which should display:
   
   .. code:: bash
       
       origin  git@github.com:YourLogin/hyperbox-brain.git (fetch)
       origin  git@github.com:YourLogin/hyperbox-brain.git (push)
       upstream    git@github.com:UTS-CASLab/hyperbox-brain.git (fetch)
       upstream    git@github.com:UTS-CASLab/hyperbox-brain.git (push)


   You should now have a working installation of hyperbox-brain, and your git
   repository properly configured. The next steps now describe the process of
   modifying code and submitting a PR.

#. Synchronize your ``main`` branch with the ``upstream/main`` branch,
   more details on `GitHub Docs <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_:

   .. code:: bash
       
       git checkout main
       git fetch upstream
       git merge upstream/main

#. Create a feature branch to hold your development changes:
    
    .. code:: bash
        
        git checkout -b my_feature

   and start making changes. Always use a feature branch. It's good
   practice to never work on the ``main`` branch!

#. (**Optional**) Install `pre-commit <https://pre-commit.com/#install>`_ to
   run code style checks before each commit:
    
    .. code:: bash
        
        pip install pre-commit
        pre-commit install
        
   pre-commit checks can be disabled for a particular commit with `git commit -n`.

#. Develop the feature on your feature branch on your computer, using Git to
   do the version control. When you're done editing, add changed files using
   ``git add`` and then ``git commit``:

    .. code:: bash
        
        git add modified_files
        git commit

   to record your changes in Git, then push the changes to your GitHub
   account with:

    .. code:: bash
        
        git push -u origin my_feature

#. Follow `these instructions 
   <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   to create a pull request from your fork. This will send an
   email to the committers. You may want to consider sending an email to the
   mailing list for more visibility.
   
   It is often helpful to keep your local feature branch synchronized with the
   latest changes of the main hyperbox-brain repository:
   
   .. code:: bash
       
       git fetch upstream
       git merge upstream/main
       
   Subsequently, you might need to solve the conflicts. You can refer to the
   `Git documentation related to resolving merge conflict using the command
   line <https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_.

.. topic:: **Learning git**:

    The `Git documentation <https://git-scm.com/documentation>`_ and
    http://try.github.io are excellent resources to get started with git,
    and understanding all of the commands shown here.

