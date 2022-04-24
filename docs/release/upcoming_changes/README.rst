Changelog
=========

This folder consists of "news fragments" that are short files which include a
small **ReST**-formatted text that will be inserted to the next what's new page.

Make sure to use full sentences with correct case and punctuation, and please
try to use Sphinx intersphinx using backticks. The fragment should have a
header line and an underline using ``------``

Each file should be named like ``<PULL REQUEST>.<TYPE>.rst``, where
``<PULL REQUEST>`` is a pull request number, and ``<TYPE>`` is one of:

* ``new_function``: New user facing functions.
* ``deprecation``: Changes existing code to emit a DeprecationWarning.
* ``future``: Changes existing code to emit a FutureWarning.
* ``expired``: Removal of a deprecated part of the API.
* ``compatibility``: A change which requires users to change code and is not
  backwards compatible. (Not to be used for removal of deprecated features.)
* ``new_feature``: New user facing features like ``kwargs``.
* ``improvement``: General improvements and edge-case changes which are
  not new features or compatibility related.
* ``performance``: Performance changes that should not affect other behaviour.
* ``change``: Other changes
* ``highlight``: Adds a highlight bullet point to use as a possibly highlight
  of the release.

It is possible to add two files with different categories (and text) if both
are relevant. For example a change may improve performance but have some
compatibility concerns.

Most categories should be formatted as paragraphs with a heading.
So for example: ``12345.new_feature.rst`` would have the content::

    ``my_new_feature`` option for `my_favorite_function`
    ----------------------------------------------------
    The ``my_new_feature`` option is now available for `my_favorite_function`.
    To use it, write ``hbbrain.my_favorite_function(..., my_new_feature=True)``.

``highlight`` is usually formatted as bulled points making the fragment
``* This is a highlight``.

Note the use of single-backticks to get an internal link (assuming
``my_favorite_function`` is exported from the ``hbbrain`` namespace),
and double-backticks for code.

If you are unsure what pull request type to use, don't hesitate to ask in your
PR.

You can install ``towncrier`` and run ``towncrier build --draft --version 1.18``
if you want to get a preview of how your change will look in the final release
notes.

.. note::

    This README was adapted from the pytest changelog readme and numpy changelog
    readme under the terms of the MIT licence.
