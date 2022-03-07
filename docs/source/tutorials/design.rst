.. _design:

Experimental designs
====================

An experimental design is a sample of the domain of the curve we are
fitting. We may use the submodule :mod:`design` to generate
experimental designs.

.. _lhs:

Latin-hypercube sampling
------------------------

A popular method for generating a design is Latin-hypercube sampling
[M79_, ]. As a rule of thumb a sample size of :math:`10d`, where
:math:`d` is the dimension of the training inputs, is sufficient
[L09_]. We may generate a Latin-hypercube design of size :math:`10d`
for the three-dimensional region :math:`[0, 1] \times [0, 1] \times
[0, 1]` as follows.

.. sourcecode:: python

   >>> bounds = [[0., 1.], [0., 1.], [0., 1.]]
   >>> mim.design(bounds, method="lhs", n=10)
   array([[0.05, 0.55, 0.75],
          ...
	  [0.95, 0.85, 0.15]])

In fact, the function :func:`design` returns a Latin-hypercube
sample of size :math:`10d` by default. Such a design may be generated
more simply as follows.

.. sourcecode:: python

   >>> mim.design(bounds)
   array([[0.05, 0.55, 0.75],
          ...
	  [0.95, 0.85, 0.15]])

PyMimic generates Latin-hypercube designs using look-up tables
generated using the method of maximum projection [JB15_, BJ18_] and
then randomly rotated and reflected.

.. _gmlhs:

Generalized Latin-hypercube sampling
------------------------------------

An extension of Latin-hypercube sampling is generalized
Latin-hypercube sampling [DP10_]. This preferentially places training
inputs at the boundary of the sample region.

.. sourcecode:: python

   >>> bounds = [[0., 1.], [0., 1.], [0., 1.]]
   >>> mim.design(bounds, method="gmlhs", n=10)
   array([[0.00615583, 0.57821723, 0.85355339],
          ...
	  [0.99384417, 0.94550326, 0.05449674]])

PyMimic generates a generalized Latin-hypercube design by transforming
a Latin-hypercube design [lhs_].

.. _regular-lattice sampling:

Regular-lattice sampling
------------------------

We may also generate a design using a regular lattice.

.. sourcecode:: python

   >>> bounds = [[0., 1.], [0., 1.], [0., 1.]]
   >>> mim.design(bounds, method="regular", n=10)
   array([[0.        , 0.        , 0.        ],
          ...
	  [1.        , 1.        , 1.        ]])

.. _random_sampling:

Random sampling
---------------

We may also generate a design using random sampling.

.. sourcecode:: python

   >>> bounds = [[0., 1.], [0., 1.], [0., 1.]]
   >>> mim.design(bounds, method="random", n=10)
   array([[0.01865849, 0.457221  , 0.00652817],
          ...
	  [0.25116118, 0.46654406, 0.22595428]])

References
----------

.. [M79]

   McKay, M. D., Beckman, R. J., and W. J. Conover. 1979. \'A
   comparison of three methods for selecting values of input variables
   in the analysis of output from a computer code\' in
   *Technometrics*, 21 (2): 239--45.  Available at
   https://www.doi.org/10.2307/1268522.

.. [JB15]
   
   Joseph, V. R., Gul, E., and Ba, S. 2015. \'Maximum projection
   designs for computer experiments\' in *Biometrika*, 102: 371--80.
   Available at https://doi.org/10.1093/biomet/asv002.

.. [BJ18]
   
   Ba, S., and V.R. Joseph. 2018. MaxPro: maximum projection designs
   [software]. Available at
   https://cran.r-project.org/web/packages/MaxPro/index.html.
   
.. [DP10]

   Dette, H., and A. Pepelyshev. 2010. \'Generalized Latin hypercube
   design for computer experiment\' in *Technometrics*, 51 (4):
   421--9. Available at https://doi.org/10.1198/TECH.2010.09157.

.. [L09]

   Loeppky, J.L., Sacks, J., and W.J. Welch. 2009. \'Choosing the
   sample size of a computer experiment: a practical guide\' in
   *Technometrics* 51 (4): 366--76. Available at
   https://doi.org/10.1198/TECH.2009.08040.
