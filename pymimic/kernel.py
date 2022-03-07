#!/usr/bin/env python3

"""A suite of positive-semidefinite kernels

This module provides positive-semidefinite kernels for use as
second-moment or covariance kernels (parameters ``covfunc`` in the
classes :class:`pymimic.emulator.Blp` and
:class:`pymimic.emulator.Blup`).

The positive-semidefinite kernels included are:

   - exponential,
   - Gneiting,
   - linear,
   - Matern
   - neural network,
   - periodic,
   - polynomial,
   - rational quadratic,
   - squared exponential, and
   - white noise.

The linear, polynomial, and white-noise kernels are not supported by
PyMimic's optimization routines, :meth:`Blp.opt` and :meth:`Blup.opt`.

"""

import numpy as np
import pymimic as mim

from scipy import special
from scipy.spatial.distance import cdist

def _inner(x, y, M):
    """Evaluate the inner product of two vectors"""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of columns.")
    try:
        # If x and y are not scalars
        return np.atleast_2d(x@M@y.T)
    except ValueError:
        # If x and y are scalars
        x = np.atleast_2d(x)
        return x.T*M*y

def _metric(x, y, M):
    """Evaluate the distance between two points"""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of columns.")
    try:
        # If x and y are not scalars
        return cdist(x, y, "mahalanobis", VI=M)
    except ValueError:
        # If x and y are scalars
        x = np.atleast_2d(x)
        return np.sqrt(M)*np.abs(y - x.T)

def exponential(x, y, sigma2, gamma, *M):
    r"""Evaluate the :math:`\gamma`-exponential kernel

    The :math:`\gamma`-*exponential kernel* is the positive-semidefinite
    kernel :math:`k: \mathbf{R}^{d} \times \mathbf{R}^{d}
    \longrightarrow \mathbf{R}` given by

    .. math::

       k(s, t) = \sigma^{2}\exp(- \|(t - s)\|^{\gamma})

    where :math:`\sigma^{2}` is a positive number, :math:`\|t - s\| :=
    \sqrt{(t - s)^{\mathrm{t}}M(t - s)}`, :math:`M :=
    \operatorname{diag}(m_{1}, m_{2}, \dots, m_{d})` is a
    positive-semidefinite diagonal matrix and :math:`\gamma \in (0,
    2]`.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    sigma2 : scalar

        Variance.

    gamma : scalar

        Exponent.

    *M : scalars
    
        Diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The :math:`\gamma`-exponential kernel evaluated for ``x`` and
        ``y``.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.exponential(0., 1., 1., 2., 1.)
       array([[0.36787944]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.exponential([0., 1.], [2., 3.], 1., 2., 1.)
       array([[1.83156389e-02, 1.23409804e-04],
              [3.67879441e-01, 1.83156389e-02]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.exponential([[0., 0.]], [[0., 1.]], 1., 2., 1., 1.)
       array([[0.36787944]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.exponential([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 2., 1., 1.)
       array([[0.36787944, 0.13533528],
              [0.13533528, 0.36787944]])

    """
    M = np.diag(M)
    return sigma2*np.exp(- _metric(x, y, M)**gamma)

def gneiting(x, y, sigma2, alpha, *M):
    r"""Evaluate the Gneiting kernel

    The *Gneiting kernel* is the positive-semidefinite kernel
    :math:`k: \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow
    \mathbf{R}` given by

    .. math:: 

       k(s, t) = 
       \begin{cases} 
       \sigma^{2}(1 + t^{\alpha})^{-3}((1 - t)\cos(\pi{}t) + \dfrac{1}{\pi}\sin(\pi{}t)) &\text{if } t \in [0, 1],\\
       0 &\text{otherwise.}
       \end{cases}

    where :math:`\sigma^{2}` is a positive number, :math:`\alpha \in
    (0, 2]`, :math:`\|t - s\| := \sqrt{(t - s)^{\mathrm{t}}M(t - s)}`,
    and :math:`M := \operatorname{diag}(m_{1}, m_{2}, \dots, m_{d})`
    is a positive-semidefinite diagonal matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    sigma2 : scalar

        Variance.

    alpha : scalar

        Exponent.

    *M : scalars
    
        Diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The Gneiting kernel evaluated for ``x`` and ``y``.

    References
    ----------
    
    .. [G02]

       Gneiting, T. 2002. \'Compactly supported correlation
       functions\' in *Journal of multivariate analysis*, 83 (2):
       493--508. Available at https://doi.org/10.1006/jmva.2001.2056.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.gneiting(0., 1., 1., 1., 1.)
       array([[4.87271479e-18]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.gneiting([0., 1.], [2., 3.], 1., 1., 1., 1.)
       array([[0., 1.]
              [1., 0.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.gneiting([[0., 0.]], [[0., 1.]], 1., 1., 1., 1.)
       array([[4.87271479e-18]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.gneiting([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1., 1.)
       array([[4.87271479e-18, 0.00000000e+00]
              [0.00000000e+00, 4.87271479e-18]])

    """
    def fun(t):
        return sigma2*(1. + t**alpha)**(-3.)*(
            (1. - t)*np.cos(np.pi*t) + np.sin(np.pi*t)/np.pi
        )

    M = np.diag(M)
    t = _metric(x, y, M)
    # This is slow. Use altervative to np.select
    return np.select([t<=1., t>1.], [fun(t), 0.])

def linear(x, y, a, *M):
    r"""Evaluate the linear kernel

    The *linear kernel* is the positive-semidefinite kernel :math:`k:
    \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow \mathbf{R}`
    given by

    .. math::

       k(s, t) = a s^{\mathrm{t}}Mt

    where :math:`a` is a positive integer, and :math:`M :=
    \operatorname{diag}(m_{1}, m_{2}, \dots, m_{d})` is a
    positive-semidefinite diagonal matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    a : scalar

        Scaling.

    *M : scalars
    
        Diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The linear kernel evaluated for ``x`` and ``y``.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.linear(0., 1., 1., 1.)
       array([[0.0]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.linear([0., 1.], [2., 3.], 1., 1.)
       array([[0. 0.]
              [2. 3.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.linear([[0., 0.]], [[0., 1.]], 1., 1., 1.)
       array([[0.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.linear([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1.)
       array([[0., 0.],
              [0., 1.]])

    """
    M = np.diag(M)
    return a*_inner(x, y, M)

def matern(x, y, sigma2, nu, *M):
    r"""Evaluate the Matern kernel

    The *Matern kernel* is the positive-semidefinite kernel :math:`k:
    \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow \mathbf{R}`
    given by

    .. math::

       k(s, t) = \sigma^{2}\dfrac{2^{1 - \nu}}{\Gamma(\nu)}\|t - s\|^{\nu}K_{\nu}(\|t - s\|)
    
    where :math:`\sigma^{2}` is a positive number, :math:`\nu` is a
    positive number, :math:`K_{\nu}` is a modified Bessel function of
    the second kind, :math:`\|t - s\| := \sqrt{(t - s)^{\mathrm{t}}M(t
    - s)}`, and :math:`M := \operatorname{diag}(m_{1}, m_{2}, \dots, m_{d})`
    is a positive-semidefinite diagonal matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.n

    sigma2 : scalar

        Variance.

    nu : scalar

        Exponent.

    *M : scalars
    
        Diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The Matern kernel evaluated for ``x`` and ``y``.

    Notes
    -----
    
    Evaluation of :math:`K_{\nu}` is done using
    :func:`scipy.special.kv`. This is computationally expensive. Hence
    calls to ``matern`` are slow.

    References
    ----------

    .. [M86]

       Matern, Bertil. 1986. *Spatial Variation*. Berlin: Springer-Verlag.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.matern(0., 1., 1., 1., 1.)
       array([[0.44434252]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.matern([0., 1.], [2., 3.], 1., 1., 1., 1.)
       array([[0.13966747, 1.        ]
              [1.        , 0.13966747]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.matern([[0., 0.]], [[0., 1.]], 1., 1., 1., 1.)
       array([[0.44434252]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.matern([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1., 1.)
       array([[0.44434252, 0.27973176]
              [0.27973176, 0.44434252]])

    """
    # TO DO: evaluation of K_{\nu} is expensive. Evaluate special
    # cases (e.g. half-integer \nu) separately
    def fun(t):
        return (2.**(1 - nu)
                *(np.sqrt(2.*nu)*t)**(nu)
                *special.kv(nu, np.sqrt(2.*nu)*t)
                /special.gamma(nu))

    M = np.diag(M)
    t = _metric(x, y, M)
    # This is slow. Use altervative to np.select
    return sigma2*np.select([t==0., t>0.], [1., fun(t)])
    
def neural_network(x, y, a, *M):
    r"""Evaluate the neural-network kernel

    The *neural-network kernel* is the positive-semidefinite kernel
    :math:`k: \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow
    \mathbf{R}` given by

    .. math::

       k(s, t) = a\dfrac{2}{\pi}\operatorname{arcsin}\left(\dfrac{2\tilde{s}^\mathrm{t}M\tilde{t}}{\sqrt{(1 + 2\tilde{s}^\mathrm{t}M\tilde{s})(1 + 2\tilde{t}^\mathrm{t}M\tilde{t})}}\right)

    where :math:`a` is a positive number, :math:`\tilde{s} := (1,
    s_{1}, \dots , s_{n})`, :math:`\tilde{t} := (1, t_{1}, \dots , t_{n})`,
    and :math:`M := \operatorname{diag}(m_{0}, m_{1}, \dots , m_{d})` is
    a positive-semidefinite matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    a : scalar

        Scaling.

    *M : scalars
    
        Diagonal elements of the augmented metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The neural-network kernel evaluated for ``x`` and ``y``.

    References
    ----------

    .. [W98]

       Williams, C.K.I. 1998. \'Computation with Infinite Neural 
       Networks\' in *Neural Computation*, 10 (5): 1203--1216. Available at 
       https://doi.org/10.1162/089976698300017412.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.neural_network(0., 1., 1., 1., 1.)
       array([[0.34545478]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.neural_network([0., 1.], [2., 3.], 1., 1., 1.)
       array([[0.22638364, 0.16216101]
              [0.60002474, 0.57029501]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.neural_network([[0., 0.]], [[0., 1.]], 1., 1., 1., 1.)
       array([[0.34545478]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.neural_network([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1., 1.)
       array([[0.34545478, 0.28751878]
              [0.26197976, 0.47268278]])

    """
    m_0 = M[0]
    M = np.diag(M[1:])
    num = 2.*(m_0 + _inner(x, y, M))
    denom_1 = 1. + 2.*(m_0 + np.diag(_inner(x, x, M)))
    denom_2 = 1. + 2.*(m_0 + np.diag(_inner(y, y, M)))
    denom = np.sqrt(np.atleast_2d(denom_1).T@np.atleast_2d(denom_2))
    return a*np.arcsin(num/denom)*(2./np.pi)

def periodic(x, y, sigma2, *args):
    r"""Evaluate the periodic kernel

    The *periodic kernel* is the positive-semidefinite kernel :math:`k:
    \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow \mathbf{R}`
    given by

    .. math::

       k(s, t) = \sigma^{2}\exp\left(-\dfrac{1}{2}\sum_{i = 1}^{d}m_{i}\sin^{2}\left(\dfrac{\pi(s_{i} - t_{i})}{\lambda_{i}}\right)\right)

    where :math:`\sigma^{2}` is a positive number, :math:`\lambda =
    (\lambda_{i})_{i}^{d}` is a tuple of positive numbers, and
    :math:`M := \operatorname{diag}(m_{1}, m_{2}, \dots m_{d})` is
    positive-semidefinite matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    sigma2 : scalar

        Variance.

    *args : scalars
    
        Periods for each dimension and diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The periodic kernel evaluated for ``x`` and ``y``.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.periodic(0., 1., 1., 1., 1.)
       array([[1.]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.periodic([0., 1.], [2., 3.], 1., 1., 1., 1., 1.)
       array([[1., 1.]
              [1., 1.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.periodic([[0., 0.]], [[0., 1.]], 1., 1., 1., 1., 1.)
       array([[1.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.periodic([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1., 1., 1.)
       array([[1., 1.]
              [1., 1.]])

    """
    x = np.atleast_1d(x)
    # y = np.atleast_1d(y)
    args = np.array(args)
    k = np.pi/args[:x.ndim]
    M = args[x.ndim:]
    # TO DO: list-comp for delta is slow, replace it
    delta = np.array([x_i - y for x_i in x])
    delta = np.atleast_3d(delta)
    return sigma2*np.exp(-0.5*np.sum(M*np.sin(k*delta)**2., axis=-1))

def polynomial(x, y, a, b, p, *M):
    r"""Evaluate the polynomial kernel

    The *polynomial kernel* is the positive-semidefinite kernel
    :math:`k: \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow
    \mathbf{R}` given by

    .. math::

       k(s, t) = a(b + s^{\mathrm{t}}Mt)^{p}

    where :math:`a` is a positive number, :math:`p` is a positive
    integer, and :math:`M := \operatorname{diag}(m_{1}, m_{2}, \dots ,
    m_{d})` is positive-semidefinite matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    a : scalar

        Scaling.

    b : scalar

        Offset.

    p : int

        Exponent.

    *M : scalars
    
        Diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The polynomial kernel evaluated for ``x`` and ``y``.

    Notes
    -----

    The kernel :func:`polynomial` is not supported by PyMimic's
    optimization routines, :meth:`Blp.opt` and :meth:`Blup.opt`. This
    is because the degree of the polynomial, :math:`p`, is restricted
    to the natural numbers.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.polynomial(0., 1., 1., 1., 1, 1.)
       array([[1.]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.polynomial([0., 1.], [2., 3.], 1., 1., 1, 1.)
       array([[1., 1.]
              [3., 4.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.polynomial([[0., 0.]], [[0., 1.]], 1., 1., 1, 1., 1.)
       array([[1.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.polynomial([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1, 1., 1.)
       array([[1., 1.]
              [1., 2.]])

    """
    M = np.diag(M)
    return a*(b + _inner(x, y, M))**p

def rational_quadratic(x, y, sigma2, alpha, *M):
    r"""Evaluate the rational-quadratic kernel

    The *rational-quadratic kernel* is the positive-semidefinite kernel
    :math:`k: \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow
    \mathbf{R}` given by

    .. math::

       k(s, t) = \sigma^{2}\left(1 + \dfrac{\|t - s\|}{2\alpha}\right)^{- \alpha}

    where :math:`\sigma^{2}` and :math:`\alpha` are positive numbers,
    :math:`\|t - s\| := \sqrt{(t - s)^{\mathrm{t}}M(t - s)}`, and
    :math:`M := \operatorname{diag}(m_{1}, m_{2}, \dots, m_{d})` is a
    positive-semidefinite diagonal matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    sigma2 : scalar

        Variance.

    alpha : scalar

        Exponent.

    *M : scalars
    
        Diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The rational-quadratic kernel evaluated for ``x`` and ``y``.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.rational_quadratic(0., 1., 1., 1., 1.)
       array([[0.66666667]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.rational_quadratic([0., 1.], [2., 3.], 1., 1., 1.)
       array([[0.33333333, 0.18181818]
              [0.66666667, 0.33333333]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.rational_quadratic([[0., 0.]], [[0., 1.]], 1., 1., 1., 1.)
       array([[0.66666667]])

    Two two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.rational_quadratic([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1., 1.)
       array([[0.66666667, 0.5       ]
              [0.5       , 0.66666667]])

    """
    M = np.diag(M)
    return sigma2*(1. + 0.5*_metric(x, y, M)**2./alpha)**(- alpha)

def squared_exponential(x, y, sigma2, *M):
    r"""Evaluate the squared-exponential kernel

    The *squared-exponential kernel* is the positive-semidefinite kernel
    :math:`k: \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow
    \mathbf{R}` given by

    .. math:: 

       k(s, t) = \sigma^{2}\exp\left(- \dfrac{1}{2}\|(t - s)\|^{2}\right)

    where :math:`\sigma^{2}` is a positive number, :math:`\|t - s\| :=
    \sqrt{(t - s)^{\mathrm{t}}M(t - s)}`, and :math:`M :=
    \operatorname{diag}(m_{1}, m_{2}, \dots, m_{d})` is a
    positive-semidefinite diagonal matrix.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    sigma2 : scalar

        Variance.

    *M : scalars
    
        Diagonal elements of the metric matrix.

    Returns
    -------

    res : (n, m) ndarray

        The squared-exponential kernel evaluated for ``x`` and ``y``.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.squared_exponential(0., 1., 1., 1.)
       array([[0.60653066]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.squared_exponential([0., 1.], [2., 3.], 1., 1.)
       array([[0.13533528, 0.011109  ],
              [0.60653066, 0.13533528]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.squared_exponential([[0., 0.]], [[0., 1.]], 1., 1., 1.)
       array(0.60653066)

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.squared_exponential([[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]], 1., 1., 1.)
       array([[0.60653066, 0.36787944],
              [0.36787944, 0.60653066]])

    """
    M = np.diag(M)
    return sigma2*np.exp(- 0.5*_metric(x, y, M)**2.)

def white_noise(x, y, sigma2):
    r"""Evaluate the white-noise kernel

    The *white-noise kernel* is the positive-semidefinite kernel :math:`k:
    \mathbf{R}^{d} \times \mathbf{R}^{d} \longrightarrow \mathbf{R}`
    given by

    .. math:: 

       k(s, t) = \sigma^{2}\delta(s, t)

    where :math:`\sigma^{2}` is a positive number, :math:`\delta` is
    the Kronecker delta.

    Parameters
    ----------

    x : (n, d) array_like

        First argument of the kernel.

    y : (m, d) array_like

        Second argument of the kernel.

    sigma2 : scalar

        Variance.

    Returns
    -------

    res : (n, m) ndarray

        The white-noise kernel evaluated for ``x`` and ``y``.

    Example
    -------

    One one-dimensional argument in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.white_noise(0., 1., 1.)
       array([[0.]])

    Two one-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.white_noise([0., 1.], [1., 2.], 1.)
       array([[0., 0.]
              [1., 0.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.white_noise([[0., 0.]], [[0., 1.]], 1.)
       array([[0.]])

    One two-dimensional arguments in each slot:

    .. sourcecode:: python

       >>> pymimic.kernel.white_noise([[0., 0.], [0., 1.]], [[0., 1.], [1., 1.]], 1.)
       array([[0., 0.]
              [1., 0.]])

    """
    x = np.atleast_1d(x)
    M = np.eye(x.ndim)
    t = _metric(x, y, M)
    # This is slow. Use altervative to np.select
    return sigma2*np.select([t==0, t!=0], [1., 0.])
        
