#!/usr/bin/env python3

"""A suite of basis functions

A suite of bases for spaces of mean-value functions (parameter
``basis`` in the class :class:`pymimic.emulator.Blup`)

"""

import itertools
import numpy as np

def const():
    r"""Return a tuple consisting of the constantly one function

    The *constantly one function* is the function
    :math:`f:\mathbf{R}^{n} \longrightarrow \mathbf{R}` given by

    .. math::

       f(t) = 1.

    Returns
    -------

    res : (1,) tuple

        A tuple consisting of the constantly one function

    Example
    -------

    Generate a basis:

    .. sourcecode:: python

       >>> pymimic.basis.const()
       (<function const.<locals>.fun at 0x7f800305cc80,)

    Evaluate elements of the tuple for one one-dimensional argument:

    .. sourcecode:: python

       >>> pymimic.basis.const()[0](0.)
       array(1.)

    Evaluate elements of the tuple for two one-dimensional arguments:

    .. sourcecode:: python

       >>> pymimic.basis.const()[0]([0., 1.])
       array([1., 1.])

    Evaluate elements of the tuple for one two-dimensional argument:

    .. sourcecode:: python

       >>> pymimic.basis.const()[0]([[0., 1.]])
       array(1.)

    Evaluate elements of the tuple for two two-dimensional arguments:

    .. sourcecode:: python

       >>> pymimic.basis.const()[0]([[0., 1.], [2., 3.]])
       array([1., 1.])

    """
    def fun(t):
        t = np.atleast_1d(t)
        return np.ones(len(t)).squeeze()
    
    return (fun,)

