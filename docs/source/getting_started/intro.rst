.. _intro:

****************
What is PyMimic?
****************

PyMimic is a pure-Python implementation of linear prediction for
real-valued random variables [G62_]. Linear prediction may be used for
curve fitting, Kriging [C86_], DACE modelling [S89_], and
Gaussian-process emulation [RW06_]. These are the intended uses of
PyMimic.


The basics
##########

A linear predictor for a random variable :math:`Z` based on a random
process :math:`X := \{X_{t}\}_{t \in T}` is a linear combination of
the elements of :math:`X`. By denoting this predictor :math:`\hat{Z}`
we have that

.. math::

   \hat{Z} = \sum_{i = 1}^{n}a_{t}X_{t}.

We may form the *best linear predictor* (BLP) by choosing the set of
coefficients, :math:`(a_{t})_{t \in T}`, so as to minimize the
mean-squared error, :math:`\operatorname{E}((Z - \hat{Z})^{2})`.
      
We may further impose the constraint that the expected values of
:math:`Z` and :math:`\hat{Z}` are equal, i.e. that
:math:`\operatorname{E}(\hat{Z}) = \operatorname{E}(Z)`. In order to
do this, we work under the assumption that :math:`\operatorname{E}` is
an element of a finite-dimensional space of functions, :math:`F`. We
then minimize the mean-squared error :math:`\operatorname{E}((Z -
\hat{Z})^{2})` subject to the constraint that
:math:`\operatorname{E}_{f}(\hat{Z}) = \operatorname{E}_{f}(Z)` for
all :math:`\operatorname{E}_{f} \in F`. We call this the *best linear
unbiased predictor* (BLUP).

PyMimic provides the classes :class:`Blp` and :class:`Blup` for the
purposes of computing realizations of these predictors.

PyMimic also provides the following submodules:

    - :mod:`kernel`, a suite of positive-definite kernels, which
      may be used as second-moment or covariance kernels when
      specifying :math:`(Z, X)`;
    - :mod:`basis`, a suite of bases, which may be used to specify the
      space of mean-value functions when computing the BLUP;
    - :mod:`testfunc`, a suite of test functions, which may be used to
      illustrate linear prediction; and
    - :mod:`plot`, plotting routines that may be used to visualize the
      results of linear prediction and assist in their validation.


References
##########

.. [C86]

   Cressie, N. 1986. 'Kriging nonstationary data' in *Journal of the
   American Statistical Association*, 81 (395): 625--34. Available at
   https://www.doi.org/10.1080/01621459.1986.10478315.
   
.. [G62]

   Goldberger, A. S. 1962. \'Best linear unbiased prediction in
   the generalized linear regression model\' in *Journal of the
   American Statistical Association*, 57 (298): 369--75. Available
   at https://www.doi.org/10.1080/01621459.1962.10480665.

.. [RW06]

   Rasmussen, C.E., and C.K.I. Williams. 2006. *Gaussian processes for
   machine learning*. Cambridge: MIT Press.

.. [S89]

   Sacks, J., Welch, W. J., Mitchell, T.J.,
   and H. P. Wynn. 1989. 'Design and analysis of computer experiments'
   in *Statistical science*, 4 (4): 409--23. Available at
   https://www.doi.org/10.1214/ss/1177012413.
