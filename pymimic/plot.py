#!/usr/bin/env python3

"""Plotting module for PyMimic

This module provides the functions :func:`plot` and :func:`diagnostic`
for use in plotting the results of linear prediction.

"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def plot(Z, extent, figsize=(3.3071, 3.3071), xlabel_text=None,
         ylabel_text=None, label_kwargs={}, xtick_loc=None,
         ytick_loc=None, ticks_kwargs=None, imshow=True,
         imshow_kwargs={}, contour=True, contour_kwargs={},
         cbar=True, cbar_kwargs={}):
    r"""Return plot of the one- and two-dimensional sums of an array.
    
    Parameters
    ----------

    Z : (n, dim) array_like

        Sample of function to be plotted.

    extent : (2, dim) array

        Limits of the data for each subplot.

    figsize : (2,) array_like, optional

        Figure size, in the form `(width, height)`.

    xlabel_text : (dim,) array_like, optional

        Labels for :math:`x` axes.

    ylabel_text : (dim,) array_like, optional

        Labels for :math:`y` axes.

    label_kwargs : dict, optional

        Keyword arguments to be passed to :func:`matplotlib.axis.Axis.set_xlabel`

    xtick_loc : (dim,) array_like, optional

        Position of :math:`x`-ticks for each :math:`x`-axis.

    ytick_loc : (dim,) array_like, optional

        Position of :math:`y`-ticks for each :math:`y`-axis.

    ticks_kwargs : dict, optional

        Keyword arguments to be passed to :func:`matplotlib.axis.Axis.set_xticks` and :func:`matplotlib.axis.Axis.set_yticks`

    imshow : bool, optional

        If ``True`` display heatmap on off-diagonal axes. Default is ``True``.

    imshow_kwargs : dict, optional

        Keyword arguments to be passed to matplotlib.pyplot.imshow.

    contour : bool, optional

        If ``True`` display contours on off-diagonal axes. Default is ``True``.

    contour_kwargs :dict, optional

        Keyword arguments to be passed to :func:`matplotlib.pyplot.contour`.

    cbar : bool, optional

        If ``True`` display colour bar. Default is ``True``.

    cbar_kwargs : dict

        Keyword arguments to be passed to :func:`matplotlib.pyplot.colorbar`.

    Returns
    -------

    fig : matplotlib.figure.Figure

        Matrix of plots.

    Raises
    ------

    Warns
    -----

    See also
    --------

    matplotlib.figure.Figure :

        Matplotlib's Figure class.

    matplotlib.pyplot.imshow :

        Matplotlib's imshow function.

    matplotlib.pyplot.contour :

        Matplotlib's contour function.

    Notes
    -----

    The function :func:`plot` can be used to plot the one- and
    two-dimensional marginalizations of a probability density function
    or likelihood function. It can also be used, in a rough-and-ready
    way, to visualize functions of many variables.

    Each panel of the plot is produced by summing over the unshown
    dimensions.

    The matrix of axes is generated using
    :func:`matplotlib.pyplot.subplots`. Heat maps are generated using
    :func:`matplotlib.pyplot.imshow`. Contours are generated using
    :func:`matplotlib.pyplot.contour`.

    Example
    -------

    Generate and plot a sample of a three-dimensional Gaussian.

    >>> import scipy as sp
    >>> bounds = [[-3., 3.], [-3., 3.], [-3., 3.]]
    >>> x = mim.design(bounds=bounds, method="regular", n=25)
    >>> z = sp.stats.multivariate_normal.pdf(x, mean=np.zeros(3), cov=np.eye(3))
    >>> z = z.reshape(25, 25, 25)
    >>> mim.plot.plot(z, bounds, xlabel_text=["$x$", "$y$", "$z$"], ylabel_text=["", "$y$", "$z$"], contour_kwargs={"colors":"k",})
    >>> plt.show()

    .. figure:: ./plot.jpg

    """
    # Preliminary definitions
    Z = np.asarray(Z)
    Z = np.swapaxes(Z, 0, 1)
    dim = Z.ndim
    shape = Z.shape
    # Define the summation axes over which to compute the marginalizations
    sum_axes = np.array(
        [[tuple(k for k in range(dim) if k != i and k != j) for j in range(dim)]
         for i in range(dim)]
    )
    # Compute the marginalizations
    # TO DO: make this a generator expression.
    Z_marg = [[np.sum(Z, axis=axis) for axis in row] for row in sum_axes]
    # Min, max of diagonal plots
    if not ytick_loc:
        vmax = np.max([Z_marg[i][i] for i in range(dim)])
        vmin = np.min([Z_marg[i][i] for i in range(dim)])
        ylim = [vmin, vmax]
    # Orientation of imshow 
    if "origin" not in imshow_kwargs:
        imshow_kwargs["origin"] = "lower"
    # Generate plots
    fig, rows = plt.subplots(dim, dim, figsize=figsize)
    for i, row in enumerate(rows):
        for j in range(dim):
            ax = row[j]
            if i > j:
                # Generate lower triangle plots
                if imshow:
                    im = ax.imshow(Z_marg[i][j].T,
                                   extent=np.hstack((extent[j], extent[i])),
                                   **imshow_kwargs)
                if contour:
                    x = np.linspace(extent[j][0], extent[j][1], shape[i])
                    y = np.linspace(extent[i][0], extent[i][1], shape[j])
                    ax.contour(x, y, Z_marg[i][j].T, **contour_kwargs)
                if xtick_loc:
                    ax.set_xticks(xtick_loc[j], **ticks_kwargs)
                if ytick_loc:
                    ax.set_yticks(ytick_loc[i], **ticks_kwargs)
                ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0])
                              /(ax.get_ylim()[1] - ax.get_ylim()[0]))
            elif i < j:
                # Generate upper triangle plots
                if imshow:
                    im = ax.imshow(Z_marg[i][j],
                                   extent=np.hstack((extent[j], extent[i])),
                                   **imshow_kwargs)
                if contour:
                    x = np.linspace(extent[j][0], extent[j][1], shape[i])
                    y = np.linspace(extent[i][0], extent[i][1], shape[j])
                    ax.contour(x, y, Z_marg[i][j], **contour_kwargs)
                if xtick_loc:
                    ax.set_xticks(xtick_loc[j], **ticks_kwargs)
                if ytick_loc:
                    ax.set_yticks(ytick_loc[i], **ticks_kwargs)
                ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0])
                              /(ax.get_ylim()[1] - ax.get_ylim()[0]))
            else:
                # Generate diagonal plots
                ax.plot(np.linspace(extent[j][0], extent[j][1], shape[j]),
                        Z_marg[j][i])#, **plot_kwargs)
                if xtick_loc:
                    ax.set_xticks(xtick_loc[i])
                ax.set_ylim(ylim)
                if ytick_loc:
                    ax.set_yticks(ytick_loc[-1])
                    ax.yaxis.get_label().set_verticalalignment("baseline")
                else:
                    ax.set_ylim(ylim)
                ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0])
                              /(ax.get_ylim()[1] - ax.get_ylim()[0]))
    # Specify plot furniture
    for i, row in enumerate(rows):
        for j in range(dim):
            ax = row[j]
            if i == dim - 1:
                # Labels for x-axis on lowest row only
                if xlabel_text:
                    ax.set_xlabel(xlabel_text[j], **label_kwargs)
                else:
                    ax.set_xlabel("")
            else:
                # Tick labels for x-axis on lowest row only
                ax.set_xticklabels("")
            if j == 0:
                # Labels for y-axis on left-most column only
                if ylabel_text:
                    ax.set_ylabel(ylabel_text[i], **label_kwargs)
                else:ax.set_ylabel("")
                # if ylabel_offset:
                #     ax.yaxis.set_label_coords(-ylabel_offset, 0.5)
            else:
                # Tick labels for y-axis on left-most column only
                ax.set_yticklabels("")
    # Plot placement
    fig.subplots_adjust(bottom=0.2, top=0.8, left=0.2, right=0.8)
    # Colour bar
    if cbar:
        axes_pad = rows[0][1].get_position().x0 - rows[0][0].get_position().x1
        cax = fig.add_axes([rows[dim - 1][dim - 1].get_position().x1 + axes_pad,
                            rows[dim - 1][dim - 1].get_position().y0,
                            0.5*axes_pad,
                            rows[0][0].get_position().y1
                            - rows[dim - 1][dim - 1].get_position().y0]
        )
        cbar = fig.colorbar(im, cax=cax, **cbar_kwargs)
    return fig

def diagnostic(xtrain, residuals, var_residuals,
               figsize=(6.6142, 3.3071), xlabels=None, ylabels=None,
               xlim = None, ylim=None, ylabel_offset=None,
               subplots_kwargs={}, plot_kwargs={}, scatter_kwargs={}):
    r"""Return diagnostic plots for a linear predictor

    Returns a two-panel plot consisting of (1) the standardized
    leave-one-out residuals against the leave-one-out predictions, and
    (2) the leave-one-out predictions against their true values.

    Parameters
    ----------

    xtrain : (n,) array_like

        Training output.

    residuals : (n,) array_like

        Leave-one-out residuals.

    var_residuals : (n,) array_like

        Variances of the leave-one-out residuals.

    figsize : (2,) array_like

        Figure size, in the form `(width, height)`.

    xlabels : (dim,) tuple, optional

        Tuple of labels for :math:`x` axes.

    ylabels : (dim,) tuple, optional

        Tuple of labels for :math:`y` axes.

    Returns
    -------

    fig : matplotlib.figure.Figure

        Matrix of plots.

    Raises
    ------

    Warns
    -----

    See also
    --------

    matplotlib.figure.Figure :

        Matplotlib's Figure class.

    matplotlib.pyplot.subplots :

        Matpotlib's subplots function

    matplotlib.pyplot.scatter :

        Matplotlib's scatter function.

    Example
    -------

    First fit a curve to a sample of the Forrester function.

    >>> ttrain = np.linspace(0., 1., 10)
    >>> xxtrain = mim.testfunc.forrester(ttrain)
    >>> blup = mim.Blup(ttrain, xtrain, args=(60., 40.))
    
    Now make diagnostic plots using the LOO residuals and their variances.

    >>> mim.plot.diagnostic(xtrain, *blup.loocv)
    >>> import matplotlib.pyplot as plt
    >>> plt.show()

    .. figure:: forrester_diagnostic.jpg
       :figwidth: 100%
       :align: center

    """
    # Compute tandardized residual
    residuals_std = residuals/np.sqrt(var_residuals)
    # Make plots
    fig, ax = plt.subplots(1, 2, figsize=figsize,
                           gridspec_kw={"hspace":0.5})

    # LOOCV standardized residuals
    ax[0].scatter(xtrain, residuals_std, s=4.)
    ax[0].plot([min(xtrain), max(xtrain)], [0., 0.], color="k",
               ls="dashed")
    y_lim = np.max(np.abs(ax[0].get_ylim()))
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(- y_lim, y_lim)
    ax[0].set_aspect(
        (ax[0].get_xlim()[1] - ax[0].get_xlim()[0])
        /(ax[0].get_ylim()[1] - ax[0].get_ylim()[0])
        )
    ax[0].set_xlabel(r"$X^{\dagger}_{-i}$")
    ax[0].set_ylabel(r"$\hat{d}(X^{\dagger}_{-i})$")

    # LOOCV predictions against true value
    ax[1].scatter(xtrain, xtrain - residuals, s=4.)
    lim_min = np.min((ax[1].get_xlim()[0], ax[1].get_ylim()[0]))
    lim_max = np.max((ax[1].get_xlim()[1], ax[1].get_ylim()[1]))
    ax[1].set_xlim(lim_min, lim_max)
    ax[1].set_ylim(lim_min, lim_max)
    ax[1].set_aspect(
        (ax[1].get_xlim()[1] - ax[1].get_xlim()[0])
        /(ax[1].get_ylim()[1] - ax[1].get_ylim()[0])
        )
    lim_lower = np.min((ax[1].get_xlim()[0], ax[1].get_ylim()[0]))
    lim_upper = np.max((ax[1].get_xlim()[1], ax[1].get_ylim()[1]))
    ax[1].plot([lim_lower, lim_upper], [lim_lower, lim_upper],
               color="k", ls="dashed")
    ax[1].set_xlim(lim_lower, lim_upper)
    ax[1].set_ylim(lim_lower, lim_upper)
    ax[1].set_xlabel(r"$X_{i}$")
    ax[1].set_ylabel(r"$X^{\dagger}_{-i}$")
    return fig
