"""
The :mod:`hbbrain.utils.model_storage` submodule implements various functions
to store a trained model to the local file and load it from the file.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import pickle


def load_multi_models(filename):
    """
    Deserialize a file containing many trained models

    Parameters
    ----------
    filename : str
        The path to file storing the trained model.

    Yields
    ------
    objects
        An iterator through many models stored in the file.

    """
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def load_model(filename):
    """
    Load a stored model from a file

    Parameters
    ----------
    filename : str
        The path to file storing the trained model.

    Returns
    -------
    model : object
        An model stored in the file.

    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
        return model


def store_model(model, filename):
    """
    Store an trained model or a list of trained models to the file

    Parameters
    ----------
    model : object
        A trained model or a list of the trained models needs to store.
    filename : str
        The path to file storing the trained models.

    Returns
    -------
    None.

    """
    with open(filename, 'wb') as f:  # Overwrites any existing file.
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
