#!/usr/bin/env python3

"""Experimental design module for PyMimic

This module provides the :func:`design` function for use in generating
experimental designs.

"""

import os
import pymimic as mim
import numpy as np

def design(bounds, method="lhs", n=None):
    r"""Return an experimental design.

    Let :math:`X` be a real-valued random process, indexed by the set
    :math:`T`, a compact rectangular region of
    :math:`\mathbf{R}^{d}`. An *experimental design* of size :math:`n`
    is a set of :math:`n` elements of this region.

    Parameters
    ----------

    bounds : (d, 2) array-like

        Bounds of design.

    method : str, optional

        Sample method. This may be one of:

            - ``"lhs"`` (:math:`2 \leq d \leq 10` and :math:`d < n \leq
              100`, :math:`n = 10d` by default) Latin-hypersquare
              design;
            - ``"gmlhs"`` (:math:`2 \leq d \leq 10` and :math:`d < n
              \leq 100`, :math:`n = 10d` by default) generalized
              Latin-hypersquare design;
            - ``"random"`` (:math:`n = 10d` by default) random design;
              or
            - ``"regular"`` (:math:`n = 10` by default) regular-lattice
              design.

        The default is ``"lhs"``.

    n : int, optional

        Size of design. The default value is dependent on the design method.

    Returns
    -------

    design : (n, d) ndarray

        Design.

    Raises
    ------

    Warns
    -----

    See also
    --------

    Notes
    -----

    Latin-hypersquare designs are read from look-up tables, generated
    using the method of maximum projection [JB15_, BJ18_]. They are
    then permuted using a randomly chosen hyperoctohedral
    transformation.

    References
    ----------

    .. [M79]

       McKay, M. D., Beckman, R. J., and W. J. Conover. 1979. \'A
       comparison of three methods for selecting values of input
       variables in the analysis of output from a computer code\' in
       *Technometrics*, 21 (2): 239--45.  Available at
       https://www.doi.org/10.2307/1268522.

    .. [L09]

       Loeppky, J.L., Sacks, J., and W.J. Welch. 2009. \'Choosing the
       sample size of a computer experiment: a practical guide\' in
       *Technometrics* 51 (4): 366--76. Available at
       https://doi.org/10.1198/TECH.2009.08040.

    .. [DP10]

       Dette, H., and A. Pepelyshev. 2010. \'Generalized Latin
       hypercube design for computer experiment\' in *Technometrics*,
       51 (4): 421--9. Available at
       https://doi.org/10.1198/TECH.2010.09157.

    .. [JB15]

       Joseph, V. R., Gul, E., and Ba, S. 2015. \'Maximum projection
       designs for computer experiments\' in *Biometrika*, 102:
       371--80.  Available at https://doi.org/10.1093/biomet/asv002.

    .. [BJ18]

       Ba, S., and V.R. Joseph. 2018. MaxPro: maximum projection
       designs [software]. Available at
       https://cran.r-project.org/web/packages/MaxPro/index.html.

    Examples
    --------

    A Latin-hypersquare sample of the unit square:

    .. sourcecode:: python

       >>> import pymimic as mim
       >>> bounds = [[0., 1.], [0., 1.]]
       >>> mim.design(bounds)
       array([[ 0.025,  0.375],
              [ 0.075,  0.775],
              ...
              [ 0.975,  0.625]])

    A random sample of the unit square of, size 3:

    .. sourcecode:: python

       >>> mim.design(bounds, method="random", n=3)
       array([[ 0.69109338,  0.58548181],
              [ 0.42631358,  0.74645846],
              [ 0.39385127,  0.11020576]])

    """
    def transform_design(x):
        """Return random hyperoctohedral transformation of a design"""
        x = x - 0.5*np.ones(dim)
        for i in range(dim):
            # Reverse randomly chosen columns
            if np.random.randint(2):
                x[:,i] = x[::-1,i]
        return x + 0.5*np.ones(dim)

    bounds = np.asarray(bounds)
    dim = bounds.shape[0]
    if method == "lhs":
        if n == None:
            n = 10*dim
        try:
            file = os.path.join(os.path.dirname(__file__),
                                "design_data/lhs_design_{}_{}.dat".format(dim, n))
            design = np.loadtxt(file)
            design = np.loadtxt(file)
            design = transform_design(design)
        except OSError:
            raise ValueError(
                "size of sample less than or equal to dimension of "
                "parameter space."
            )
    elif method == "gmlhs":
        if n == None:
            n = 10*dim
        try:
            file = os.path.join(os.path.dirname(__file__),
                                "design_data/lhs_design_{}_{}.dat".format(dim, n))
            design = np.loadtxt(file)
            design = transform_design(design)
            design = 0.5*(1. - np.cos(np.pi*design))
        except OSError:
            raise ValueError(
                "size of sample less than or equal to dimension of "
                "parameter space."
            )
    elif method == "random":
        if n == None:
            n = 10*dim
        design = np.random.rand(n, dim)
    elif method == "regular":
        if n == None:
            n = 10
        x_dim = [np.linspace(0, 1, n) for i in range(dim)]
        design = np.vstack(map(np.ravel, np.meshgrid(*x_dim))).T
    else:
        raise ValueError("method not recognized.")
    design = design*(bounds.T[1] - bounds.T[0]) + bounds.T[0]

    return design
