#!/usr/bin/env python3

"""A suite of test functions

This module provides several test functions for use in generating
illustrative training data (parameters ``ttrain`` and ``xtrain`` in
the classes :class:`pymimic.emulator.Blp` and
:class:`pymimic.emulator.Blup`).

"""

import numpy as np
import pymimic as mim

def gp(t, covfunc=mim.kernel.squared_exponential, args=()):
    r"""Return a realization a centred Gaussian random process

    Parameters
    ----------

    t : (n, d) array_like

       Argument of the function.

    covfunc : callable

       Covariance kernel of the Gaussian random process

    args : tuple

       Extra arguments to be passed to the covariance kernel,
       ``covfunc``.

    Returns
    -------

    res : (n,) ndarray

        Sample of a realization of a centred Gaussian random process.

    Example
    -------

    One one-dimensional argument:

    .. sourcecode:: python

       >>> mim.testfunc.gp(1., mim.kernel.squared_exponential, (1., 1.))
       array([0.83792787])

    Multiple one-dimensional arguments:

    .. sourcecode:: python

       >>> t = np.linspace(0., 1.)
       >>> mim.testfunc.gp(t, mim.kernel.squared_exponential, (1., 1.))
       array([-0.9635262 , -0.9457061 , ... , -0.25218523])

    One two-dimensional argument:

    .. sourcecode:: python

       >>> mim.testfunc.gp([[0., 1.]], mim.kernel.squared_exponential, (1., 1., 1.))
       array([0.59491435])

    Multiple two-dimensional arguments:

    .. sourcecode:: python

       >>> t = mim.design.design([[0., 1.], [0., 1.]], "regular", 10)
       >>> mim.testfunc.gp(t, mim.kernel.squared_exponential, (1., 1., 1.)).reshape(10, 10)
       array([[-0.9635262 , -0.9457061 , ... , -0.89178485],
              ...
              [-0.29947984, -0.28774821, ... , -0.25218523]])

    """
    t = np.atleast_1d(t)
    mu = np.zeros(len(t))
    Sigma = covfunc(t, t, *args)
    return np.random.multivariate_normal(mu, Sigma)

def forrester(x):
    r"""Return the value of the Forrester function for given argument

    The *Forrester function* [F08_] is the function :math:`f:
    \mathbf{R} \longrightarrow \mathbf{R}` given by

    .. math::

       f(x) = (6x - 2)^2 \sin(12x - 4).

    When used as a test function for optimization its domain is
    typically restricted to the region :math:`[0, 1]`.

    Parameters
    ----------
    x : (n,) array_like

        Argument of the function. 

    Returns
    -------

    res : (n,) ndarray

        Value of the Forrester function.

    Example
    -------

    Single argument:

    .. sourcecode:: python

       >>> mim.testfunc.forrester(0.5)
       0.9092974268256817

    Multiple arguments:

    .. sourcecode:: python

       >>> t = np.linspace(0., 1., 10)
       >>> mim.testfunc.forrester(t)
       array([ 3.02720998, -0.81292911, ... , 15.82973195])

    References
    ----------

    .. [F08]

       Forrester, A., Sobester, A., and A. Keane. 2008. *Engineering
       design via surrogate modelling: a practical guide*. Wiley.

    """
    x = np.asarray(x)
    forrester = (6.0*x - 2.0)**2.0 * np.sin(12.0*x - 4.0)
    return forrester


def branin(x, y):
    r"""Return the value of the Branin function for given argument

    The *Branin function* [B72_] is the function :math:`f: \mathbf{R}
    \times \mathbf{R} \longrightarrow \mathbf{R}` given by

    .. math::

       f(x, y) = a (y - bx^2 + cx - d)^2 + e(1 - f) \cos x + e,

    where :math:`a = 1`, :math:`b = 5.1 / (4 \pi^2)`, :math:`c = 5 /
    \pi`, :math:`d = 6`, :math:`e = 10`, :math:`f = 1 / (8 \pi)`.
    When used as a test function for optimization its domain is
    typically restricted to the region :math:`[-5, 10] \times [0,
    15]`. In this region the function has three global minima, at
    :math:`(x, y) = (-\pi, 12.275)`, :math:`(\pi, 2.275)`, and
    :math:`(9.425, 2.475)`, where it takes the value :math:`0.398`.

    Parameters
    ----------

    x : (n,) array_like

        First argument of the function.

    y : (n,) array_like

        Second argument of the function.

    Returns
    -------

    res : (n,) ndarray

        Value of the Branin function.

    Example
    -------
    
    Single pair of arguments:

    .. sourcecode:: python
    
       >>> mim.testfunc.branin(-np.pi, 12.275)
       0.39788736

    Multiple pairs of arguments, as two lists:

    .. sourcecode:: python

       >>> x = [-np.pi, np.pi, 9.425]
       >>> y = [12.275, 2.275, 2.475]
       >>> mim.testfunc.branin(x, y)
       array([0.39788736, 0.39788736, 0.39788763])

    Multiple pairs of elements, as a `meshgrid` object:

    .. sourcecode:: python

       >>> x = np.linspace(-5., 10.)
       >>> y = np.linspace(0., 15.)
       >>> xx, yy = np.meshgrid(x, y)
       >>> mim.testfunc.branin(xx, yy)
       array([[308.12909601, 276.06003083, ... ,  10.96088904],
              ...
              [ 17.50829952,  11.55620801, ... , 145.87219088]])

    References
    ----------

    .. [B72]

       Branin, F.H. 1972. \'Widely convergent method for finding
       multiple solutions of simultaneous nonlinear equations\' in
       *IBM Journal of Research and Development*, 16 (5): 504--22.
       Available at http://www.doi.org/10.1147/rd.165.0504.

    """
    a = 1.0
    b = 5.1/(4.0*np.pi**2.0)
    c = 5.0/np.pi
    r = 6
    s = 10
    t = 1.0/(8.0*np.pi)
    branin = (a*(y - b*x**2.0 + c*x - r)**2.0
              + s*(1.0 - t)*np.cos(x)
              + s)
    return branin
