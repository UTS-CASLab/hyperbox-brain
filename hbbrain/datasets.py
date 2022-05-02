"""
Implement functions supporting for loading datasets contained in the dataset
folder.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import pandas as pd
import os
from pathlib import Path


def load_japanese_credit():
    """
    Load japanese credit dataset stored in the dataset folder.

    Returns
    -------
    X : ndarray of shape (653, 15)
        The data matrix.
    y : ndarray of shape (653, )
        The classification target.

    """
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    project_dir = Path(current_dir).parent
    path_file = os.path.join(project_dir, Path("dataset/japanese_credit.csv"))
    df = pd.read_csv(path_file, header=None)

    Xy = df.to_numpy()

    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)

    return (X, y)
