What is PyMimic?
================

PyMimic is a pure-Python implementation of Gaussian-process emulation (O'Hagan
& Kingman, 1978, and Sacks et al., 1989), and efficient global optimization
(Jones et al., 1998) for real-valued functions of several real variables. It
also provides tools for validating the results of these methods, as well as
for data preparation.

Consider a function

.. math::
   f: \mathbf{X} \longrightarrow \mathbf{R},

where :math:`\mathbf{X}` is a compact region of :math:`\mathbf{R}^D`. Given a
set of data from this function, namely a set :math:`\{(\mathbf{x}_i,
f(\mathbf{x}_i)) | \mathbf{x}_i \in \mathbf{X} \text{ and } i = 1, 2, ... ,
n\}` Gaussian-process emulation (GPE) predicts the value :math:`f(\mathbf{x})`
for given :math:`\mathbf{x} \in \mathbf{X}`.

Using the same data, efficient global optimization (EGO) finds a global
minimum of :math:`f`. It does this by iteratively performing GPE and
augmenting the data where a new minimum in the data is most-likely to be
found.

GPE and EGO are most useful when the function :math:`f` is expensive to
evaluate. When this is the case they can dramatically reduce the computational
burden of function-evaluation and optimization.

PyMimic's emphasis is on ease of use. There are two principal commands:

    - :func:`pymimic.gpe`,
    - :func:`pymimic.ego`,

which perform GPE and EGO respectively. The first of these functions returns a
``pymimic.GpeResult`` object. The important attributes of this object are
``y``, the GPE prediction for :math:`f(x)`, and ``var``, the variance
associated with this prediction. The second of these functions returns an
``scipt.optimize.OptimizeResult`` object, as is used in Scipy's optimization
routines.

The GPE machinery must be tuned for each data set. The above commands automate
this tuning, though it may be done manually, or it may be overridden using
PyMimic's advanced features.

PyMimic also provides the following subpackages:
    - ``covfunc``, a suite of covariance functions [*]_,
    - ``plot``, a suite of plotting routines used to visualize the results of
      emulation,
    - ``prep``, a suite of data-preparation tools for preparing the data
      before emulation,
    - ``testfunc``, a suite of test functions that may be used to illustate
      emulation.

These are used with the syntax ``pymimic.subpackage.fun()``, where ``package``
and ``fun`` are the names of the subpackage and subpackage function
respectively. For example, we may plot the one- and two dimensional
projections of a multivariate function using the command

    :func:`pymimic.plot.marginal()`.

This is especially useful when the function in question is a probability
density function or likelihood.


Scientific applications
-----------------------

PyMimic accompanies the paper *The dynamical modelling of dwarf-spheroidal
galaxies using Gaussian-process emulation* by Gration and Wilkinson.

.. [*] Currently, ``covfunc`` contains only the squared-exponential covariance
       function. More will be added in future releases.

       
References
----------

Gration, A., and M.I. Wilkinson. In press. `Efficient global optimization of
expensive black-box functions' in *Monthly notices of the Royal Astronomical
Sociey*. https://arxiv.org/abs/1806.06614

Jones, D. R., Schonlau, M., and W. J. Welch. 1998. 'Efficient global optimization of
expensive black-box functions' in *Journal of global optimization*, 13 (4):
455--92. https://link.springer.com/article/10.1023/A:1008306431147

O'Hagan, A., and J. F. C. Kingman. 1978. 'Curve fitting and optimal design for
prediction' in *Journal of the Royal Statistical Society. Series B
(Methodological)*, 40 (1):
1--42. https://www.jstor.org/stable/2984861?seq=1#page_scan_tab_contents

Sacks, J., Welch, W. J., Mitchell, T.J., and H. P. Wynn. 1989. 'Design and
analysis of computer experiments' in *Statistical science*, 4 (4): 409--23. https://projecteuclid.org/euclid.ss/1177012413
