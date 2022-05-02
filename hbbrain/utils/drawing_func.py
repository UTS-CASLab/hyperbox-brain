"""
The :mod:`hbbrain.utils.drawing_func` submodule implements various functions 
to support for drawing the hyperboxes.
"""
# @Author: Thanh Tung KHUAT <thanhtung09t2@gmail.com>
# License: GPL-3.0

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from hbbrain.constants import UNLABELED_CLASS


def get_cmap(n, name='brg'):
    """Get a colormap instance mapping each index in 0, 1,..., n-1 to a distinct 
    RGB color.

    Parameters
    ----------
    n : int or None, default: None
        If name is not already a Colormap instance and n is not None, the colormap 
        will be resampled to have n entries in the lookup table.
    name : matplotlib.colors.Colormap or str or None, default: 'brg'
        If a Colormap instance, it will be returned. Otherwise, the name of a colormap 
        known to Matplotlib, which will be resampled by n.

    Returns
    -------
    Return a function that maps each index in 0, 1,..., n-1 to a distinct 
    RGB color.
    
    Examples
    --------
    >>> from hbbrain.utils.drawing_func import get_cmap
    >>> cmap = get_cmap(2)
    >>> cmap(0)
    (0.0, 0.0, 1.0, 1.0)...
    """
    return plt.cm.get_cmap(name, n)
    

def draw_box(drawing_canvas, lw_bound, up_bound, color, linewidth=1):
    """Drawing rectangular (2 dimensional inputs) or cube (3 and more dimensional inputs) shapes

    Parameters
    ----------
    drawing_canvas : `axes.SubplotBase`, or another subclass of `Axes` in the matplotlib library
        Plotting object of matplotlib.
    lw_bound : array-like of shape (n_hyperboxes, n_features)
        A matrix storing lower bounds of all hyperboxes that we want to show in the canvas.
    up_bound : array-like of shape (n_hyperboxes, n_features)
        A matrix storing upper bounds of all hyperboxes that we want to show in the canvas.
    color : int, tupple, or array-like of shape (n_hyperboxes,)
        A constant value or a tuple showing the same color for all hyperboxes 
        or a vector storing the colors corresponding to the hyperboxes represented by `lw_bound` and `up_bound`
    linewidth : float, default=1
        The width of hyperbox lines

    Returns
    -------
    handler : list of Line2D or Line3D
        A list of Line2D or Line3D depending on the number of dimensions initialised in drawing_canvas to show the plotted objects.

    """
    lw_bound = np.asmatrix(lw_bound)
    up_bound = np.asmatrix(up_bound)
    
    n_hyperboxes = up_bound.shape[0]
    handler = np.empty(n_hyperboxes, dtype=object)
    
    if (isinstance(color, int) == True) or (isinstance(color, tuple) == True):
        is_constant_color = True
    else:
        is_constant_color = False
        
    for i in range(n_hyperboxes):
        if is_constant_color:
            selected_color = color
        else:
            selected_color = color[i]
            
        if lw_bound[i].size == 2:
            # plot actually returns a list of artists, hence the ,
            if lw_bound[i, 0] == up_bound[i, 0] and lw_bound[i, 1] == up_bound[i, 1]:
                handler[i], = drawing_canvas.plot(lw_bound[i, 0], lw_bound[i, 1], color=selected_color, marker='+')
            else:
                handler[i], = drawing_canvas.plot([lw_bound[i, 0], lw_bound[i, 0], up_bound[i, 0], up_bound[i, 0], lw_bound[i, 0]], [lw_bound[i, 1], up_bound[i, 1], up_bound[i, 1], lw_bound[i, 1], lw_bound[i, 1]], color=selected_color, linewidth=linewidth)
        else:
            if lw_bound[i, 0] == up_bound[i, 0] and lw_bound[i, 1] == up_bound[i, 1] and lw_bound[i, 2] == up_bound[i, 2]:
                handler[i], = drawing_canvas.plot([lw_bound[i, 0]], [lw_bound[i, 1]], [lw_bound[i, 2]], color=selected_color, marker='+')
            else:
                handler[i], = drawing_canvas.plot([lw_bound[i, 0], lw_bound[i, 0], up_bound[i, 0], up_bound[i, 0], lw_bound[i, 0], lw_bound[i, 0], lw_bound[i, 0], lw_bound[i, 0], up_bound[i, 0], up_bound[i, 0], lw_bound[i, 0], up_bound[i, 0], up_bound[i, 0], up_bound[i, 0], up_bound[i, 0], lw_bound[i, 0], lw_bound[i, 0]], \
                                   [lw_bound[i, 1], up_bound[i, 1], up_bound[i, 1], lw_bound[i, 1], lw_bound[i, 1], lw_bound[i, 1], lw_bound[i, 1], up_bound[i, 1], up_bound[i, 1], lw_bound[i, 1], lw_bound[i, 1], lw_bound[i, 1], lw_bound[i, 1], up_bound[i, 1], up_bound[i, 1], up_bound[i, 1], up_bound[i, 1]], \
                                   [lw_bound[i, 2], lw_bound[i, 2], lw_bound[i, 2], lw_bound[i, 2], lw_bound[i, 2], up_bound[i, 2], up_bound[i, 2], up_bound[i, 2], up_bound[i, 2], up_bound[i, 2], up_bound[i, 2], up_bound[i, 2], lw_bound[i, 2], lw_bound[i, 2], up_bound[i, 2], up_bound[i, 2], lw_bound[i, 2]], color=selected_color, linewidth=linewidth)
            
    return handler


def generate_grid_decision_boundary_2D(min_x=0, max_x=1, min_y=0, max_y=1, step=0.01):
    """Generate a grid of points on the 2-D plane to determine the class label of 
    these points from which decision boundary can be deduced.

    Parameters
    ----------
    min_x : float, optional, default = 0
        Starting coordinate of the 2-D grid on the X-axis.
    max_x : float, optional, default = 0
        Ending coordinate of the 2-D grid on the X-axis.
    min_y : float, optional, default = 0
        Starting coordinate of the 2-D grid on the Y-axis.
    max_y : float, optional, default = 0
        Ending coordinate of the 2-D grid on the Y-axis.
    step : float, optional, default = 0.01
        The distance between two next points.

    Returns
    -------
    a grid of points and coordinate matrices from coordinate vectors
    grid : array-like of shape (n_points, 2)
        A matrix contains all pairs of points of a 2-D grid.
    XX : array-like of shape (Ny, Nx)
        A coordinate matrix generated from a coordinate vector on the X-axis defined by `min_x`, `max_x`, and `step`.
        `Ny = (max_y - min_y)/step` and `Nx = (max_x - min_x)/step`.
    YY : array-like of shape (Ny, Nx)
        A coordinate matrix generated from a coordinate vector on the Y-axis defined by `min_y`, `max_y`, and `step`.
        `Ny = (max_y - min_y)/step` and `Nx = (max_x - min_x)/step`.

    .. note::
        The number of elements `n_points` in the matrix `grid` is computed by
        :math:`\cfrac{max_x - min_x}{step} \cdot \cfrac{max_y - min_y}{step}`.

    """
    # define the x and y scale
    x_grid = np.arange(min_x, max_x, step)
    y_grid = np.arange(min_y, max_y, step)
    # create all of the lines and rows of the grid
    XX, YY = np.meshgrid(x_grid, y_grid)
    # flatten each grid to a vector
    r1, r2 = XX.flatten(), YY.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create [x1, x2] input for the model
    grid = np.hstack((r1,r2))
    
    return grid, XX, YY


def draw_decision_boundary_2D(drawing_canvas, XX, YY, yhat):
    """Draw decision boundary in a 2-D plane

    Parameters
    ----------
    drawing_canvas : `axes.SubplotBase`, or another subclass of `Axes` in the matplotlib library
        A ploting object of matplotlib.
    XX : array-like of shape (Ny, Nx)
        A coordinate matrix of the values on the X-axis created via **numpy.meshgrid**.
        The values of X must be ordered monotonically.
    YY : array-like of shape (Ny, Nx)
        A coordinate matrix of the values on the Y-axis created via **numpy.meshgrid**.
        The values of X must be ordered monotonically.
    yhat : array-like of shape (n_points,)
        Predicted class labels for all points in the grid generated by `XX` and `YY`.

    Returns
    -------
    None.

    """
    # reshape the predictions back into a grid
    ZZ = yhat.reshape(XX.shape)
    # plot the grid of x, y and z values as a surface
    drawing_canvas.contour(XX, YY, ZZ, cmap=mpl.cm.Blues)


def draw_box_parallel_coordinate(X, y, y_pred, plot_width=800, plot_height=480, file_path="par_coor.html"):
    """
    Draw input samples in the form of parallel coordinates.

    Parameters
    ----------
    X : array like of shape (n_samples, n_features)
        A matrix of samples needs to display in the parallel coordinates.
    y : array like of shape (n_samples, )
        Class labels of samples stored in `X`.
    y_pred : int
        The samples with the same label as `y_pred` will be higlighted.
    plot_width : int, optional, default=800
        Width of the window to show graphs.
    plot_height : int, optional, default=480
        Height of the window to show graphs.
    file_path : str, optional, default="par_cord.html"
        Path including a file name to the location storing the parallel
        coordinates graph.

    Returns
    -------
    None.

    """
    import plotly.graph_objects as go
    y = np.array(y)
    # define color map
    unique_y = np.unique(y)
    color_map = get_cmap(len(unique_y) - 1)
    normalized_y = (unique_y - np.min(unique_y))/np.ptp(unique_y)
    color_scale = []
    count = 0
    for val in normalized_y:
        if val == 0:
            color_scale.append([0, 'black'])
        else:
            color_scale.append([val, mpl.colors.rgb2hex(color_map(count))])
            count += 1
    
    # Create vertical Axes
    dim_list = list()
    n_dims = X.shape[1]
    for i in range(n_dims):
        dt = dict(range = [X[:, i].min(), X[:, i].max()], label = "F" + str(i + 1), values = X[:, i])
        dim_list.append(dt)
    
    # Create an axis for the class label
    tickvals = []
    ticktext = []
    for i in unique_y:
        tickvals.append(i)
        if i == UNLABELED_CLASS:
            ticktext.append('Input')
        else:
            ticktext.append(str(i))
            
    dt = dict(range = [y.min(), y.max()], constraintrange=[y_pred, y_pred], label = "Class label", tickvals = tickvals, ticktext = ticktext, values = y)
    dim_list.append(dt)
    
    # define layout
    layout = go.Layout(
        autosize=False,
        width=plot_width,
        height=plot_height,
        margin=dict(
            l=50,
            r=50,
            b=20,
            t=80,
            pad=4
        ),
        plot_bgcolor = 'white',
        paper_bgcolor="white",
    )
    
    fig = go.Figure(data=go.Parcoords(line = dict(color=y, colorscale = color_scale), dimensions=dim_list), layout=layout, layout_title_text="Hyperboxes joining the class prediction of an input sample")
    #fig.show(renderer="iframe")
    fig.write_html(file_path, full_html=False, include_plotlyjs='cdn')
    