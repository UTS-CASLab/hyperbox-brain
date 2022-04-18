"""
Base granular class stores all coordinators of hyperboxes and their class
labels for base granular classifiers at the highest granularity.
This class is constructed as a generic structure containing a set of items with
the names defined by users as a dictionary.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: BSD 3 clause


class BaseGranular(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
