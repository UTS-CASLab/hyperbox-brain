"""
Base class for all hyperbox-based estimators.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.base import BaseEstimator, ClassifierMixin
from hbbrain.utils.drawing_func import (
    get_cmap,
    draw_box,
    generate_grid_decision_boundary_2D,
    draw_decision_boundary_2D,
    draw_box_parallel_coordinate,
    )
from hbbrain.constants import UNLABELED_CLASS


class BaseHyperboxClassifier(BaseEstimator, ClassifierMixin):
    """
    Base class for all hyperbox-based estimators in hyperbox-brain.

    .. note::

        All estimators should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``). This class only initialises
        all common parameters for hyperbox-based estimators.

    Parameters
    ----------
    theta : float or ndarray of shape (n_features,), optional, defaut = 0.5
        A maximum hyperbox size parameter for each dimension.
    gamma : float or ndarray of shape (n_features,), optional, default=1
        A sensitivity parameter describing the speed of decreasing of the membership function in each dimension.
    is_draw : boolean, optional, default = False
        A parameter is used to indicate whether the process of hyperbox building can be dynamically displayed 
        on a canvas or not. This functionality displays hyperboxes in the form of 2D or 3D. In the case that 
        the number of dimensions is higher than 3, only the three features are shown.
    V : array-like of shape (n_hyperboxes, n_features), default = an empty ``ndarray``
        A matrix stores all minimal coordinates of all existing hyperboxes, in which each row is a minimal coordinate of a hyperbox.
    W : array-like of shape (n_hyperboxes, n_features), default = an empty ``ndarray``
        A matrix stores all maximal coordinates of all existing hyperboxes, in which each row is a maximal coordinate of a hyperbox.
    C : ndarray of shape (n_hyperboxes,), default = an empty ``ndarray``
        An array contains all class lables for all existing hyperboxes.
        
    Attributes
    ----------
    n_hyperboxes : int 
        Number of hyperboxes built during :term:`fit`.

    """

    def __init__(self, theta=0.5, is_draw=False, V=None, W=None, C=None):
        self.theta = theta
        self.is_draw = is_draw
        if V is not None:
            self.V = V
        else:
            self.V = np.array([])
        if W is not None:
            self.W = W
        else:
            self.W = np.array([])
        if C is not None:
            self.C = C
        else:
            self.C = np.array([])

    def _init_hyperboxes(self):
        if self.C is None:
            self.V = np.array([])
            self.W = np.array([])
            self.C = np.array([])

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.

        """
        return self

    def delay(self, delay_constant=0.01):
        """
        Delay a time period to display hyperboxes

        Parameters
        ----------
        delay_constant : float
            Delay time period to display hyperboxes on the canvas

        """
        plt.pause(delay_constant)

    def initialise_canvas_graph(self, n_dims=2, figure_name='A trained hyperbox-based learning model', min_range=0, max_range=1):
        """
        Initialise a canvas to draw hyperboxes

        Parameters
        ----------
        n_dims : int, optional, default=2
            The number of dimensions of hyperboxes shown in the canvas (2D or 3D).
        figure_name : str, optional, default='A trained hyperbox-based learning model'
            Title name of the window containing hyperboxes.
        fig_num : int, optional, default=1
            Index of canvas.
        min_range : float, optional, default=0
            Minimum value in each axis.
        max_range : float, optional, default=1
            Maximum value in each axis.

        Returns
        -------
        drawing_canvas : `axes.SubplotBase`, or another subclass of `Axes` in the matplotlib library
            Plotting object of matplotlib.

        """
        fig = plt.figure(figure_name)
        plt.ion()
        if n_dims == 2:
            drawing_canvas = fig.add_subplot(1, 1, 1)
            drawing_canvas.axis([min_range, max_range, min_range, max_range])
        else:
            drawing_canvas = Axes3D(fig)
            drawing_canvas.set_xlim3d(min_range, max_range)
            drawing_canvas.set_ylim3d(min_range, max_range)
            drawing_canvas.set_zlim3d(min_range, max_range)

        return drawing_canvas

    def draw_hyperbox_and_boundary(self, window_name="Hyperbox-based classifier and its decision boundaries", min_range=0, max_range=1):
        """
        Draw the existing hyperboxes and their decision boundaries among classes

        .. note::

            This function only works on 2-dimensional datasets

        Parameters
        ----------
        window_name : str, optional, default="Hyperbox-based classifier and its decision boundaries"
            Name of plotting window showing hyperboxes and their decision boundaries.
        min_range : float, optional, default=0
            Minimum value in each axis.
        max_range : float, optional, default=1
            Maximum value in each axis.

        Returns
        -------
        None.

        """
        class_ids = np.unique(self.C)
        n_hyperboxes, n_dims = self.V.shape
        n_classes = len(class_ids)
        color_map = get_cmap(n_classes)
        # build a dictionary with the class label being key and color being value
        colors = {}
        for i in range(n_classes):
            colors[class_ids[i]] = color_map(i)

        drawing_canvas = self.initialise_canvas_graph(n_dims, window_name, min_range, max_range)
        # create a list of colors for the created hyperboxes
        box_colors = np.full(n_hyperboxes, None)
        for i in range(n_hyperboxes):
            box_colors[i] = colors[self.C[i]]
        # draw hyperboxes
        draw_box(drawing_canvas, self.V, self.W, box_colors)
        # Generate a grid of points in a 2D plane to determine corresponding
        # classes for various areas
        grid, xx, yy = generate_grid_decision_boundary_2D(min_range, max_range, min_range, max_range, (max_range - min_range)/200)
        # make predictions for the points in the grid
        yhat = self.predict(grid)
        # Draw decision boundary
        draw_decision_boundary_2D(drawing_canvas, xx, yy, yhat)

    def show_sample_explanation(self, xl, xu, dict_min_point_classes, dict_max_point_classes, y_pred, type_plot="par_cord", plot_width=800, plot_height=480, min_range=0, max_range=1, file_path="par_cord.html"):
        """
        Show explanation for predicted results of an input pattern under the
        form of parallel coordinates or hyperboxes in 2D or 3D planes.

        .. note::

            This function only works on numerical features.

        Parameters
        ----------
        xl : array-like of shape (n_features,)
            Lower bound of numerical features of the input pattern which needs
            to show explanation.
        xu : array-like of shape (n_features,)
            Upper bound of numerical features of an input pattern which needs
            to show explanation.
        dict_min_point_classes : dictionary
            A dictionary stores all mimimal points of hyperboxes having the
            maximum membership value for each class. The key is the class label
            and the value is the minimal points of all hyperboxes coressponding
            to each class.
        dict_max_point_classes : dictionary
            A dictionary stores all maximal points of hyperboxes having the
            maximum membership value for each class. The key is the class label
            and the value is the maximal points of all hyperboxes coressponding
            to each class.
        y_pred : int
            The predicted class of the input pattern.
        type_plot : str, optional, default="par_cord"
            Type of graph to show explanation. If the value is `par_cord`, a
            parallel coordinate is used. Otherwise, hyperboxes in 2D or 3D
            planes are shown.
        plot_width : int, optional, default=800
            Width of the window to show parallel coordinates.
        plot_height : int, optional, default=480
            Height of the window to show parallel coordinates.
        min_range : float, optional, default=0
            Minimum value in the axes to show hyperboxes in 2D or 3D planes.
        max_range : float, optional, default=1
            Maximum value in the axes to show hyperboxes in 2D or 3D planes.
        file_path : str, optional, default="par_cord.html"
            Path including a file name to the location storing the parallel
            coordinates graph.

        Returns
        -------
        None.

        """
        class_ids = np.unique([*dict_min_point_classes])
        n_dims = len(xl)
        n_classes = len(class_ids)

        if type_plot == "par_cord":
            box_color = []
            hyperboxes = np.zeros((2*n_classes + 2, n_dims), dtype=float)
            index = 0
            for c in dict_min_point_classes:
                box_color.append(int(c))
                hyperboxes[index] = dict_min_point_classes[c]
                index += 1
                box_color.append(int(c))
                hyperboxes[index] = dict_max_point_classes[c]
                index += 1

            # Add input sample
            hyperboxes[index] = xl
            box_color.append(UNLABELED_CLASS)
            index += 1
            hyperboxes[index] = xu
            box_color.append(UNLABELED_CLASS)
            draw_box_parallel_coordinate(hyperboxes, box_color, y_pred, plot_width, plot_height, file_path)
        else:
            color_map = get_cmap(n_classes)
            # build a dictionary with the class label being key and color being value
            colors = {}
            for i in range(n_classes):
                colors[class_ids[i]] = color_map(i)

            drawing_canvas = self.initialise_canvas_graph(
                n_dims, "Reprentative hyperboxes for each class with respect to the input sample", min_range, max_range)
            # Draw reprentative hyperboxes for all classes
            legend = []
            for c in dict_min_point_classes:
                box_color = colors[c]
                legend.append(str(c))
                if c == y_pred:
                    draw_box(drawing_canvas, np.asmatrix(dict_min_point_classes[c]), np.asmatrix(dict_max_point_classes[c]), box_color, 2)
                else:
                    draw_box(drawing_canvas, np.asmatrix(dict_min_point_classes[c]), np.asmatrix(dict_max_point_classes[c]), box_color, 0.5)
            # draw input pattern
            if (xl == xu).all():
                if n_dims == 2:
                    drawing_canvas.plot(xl[0], xl[1], color='black', marker='o')
                else:
                    drawing_canvas.plot([xl[0]], [xl[1]], [xl[2]], color='black', marker='o')
            else:
                draw_box(drawing_canvas, np.asmatrix(xl), np.asmatrix(xu), 'black', 1)
            legend.append('Input')
            drawing_canvas.legend(legend, loc=0, ncol=5, title="Class label", framealpha=0.4, fancybox=True)

    def get_n_hyperboxes(self):
        """
        Get number of hyperboxes in the trained hyperbox-based model

        Returns
        -------
        int
            Number of hyperboxes in the trained hyperbox-based classifier.

        """
        return self.V.shape[0]
