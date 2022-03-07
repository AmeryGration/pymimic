.. _basis:

*******************************************
Bases for the space of mean-value functions
*******************************************

When computing the BLUP we must specify a basis for the space of
mean-value functions. In ordinary kriging the random process
:math:`(Z, X)` is assumed to be weakly stationary, and hence to have
constant mean. In universal kriging the random process :math:`(Z, X)`
is assumed to be intrinsically stationary with polynomial mean.

Be aware that a basis of size comparable to the size of :math:`X` will
cause the BLUP to over-fit the data.


PyMimic and bases
#################

The basis must be specified using the keyword argument ``basis`` when
creating an instance of the :class:`Blup`. It cannot be changed
afterwards. The basis must be a tuple of functions ``func(x) ->
array_like`` with signature ``(n, d) -> (n,)``, where ``d`` is the
dimension of the index set :math:`T` (:ref:`The two flavours of linear
prediction <the_two_flavours_of_linear_prediction>`).


Built-in basis functions
------------------------

The module :mod:`basis` provides a suite of bases, which may be used
when specifying the BLUP using the class :class:`Blup`. The available
bases are described in the submodule's API documentation
(:ref:`pymimic.basis module <basis_module>`). Currently :mod:`basis`
contains only one basis, consisting of the constantly one
function. More will be added in future releases.


User-defined basis functions
----------------------------

Because Numpy and Scipy functions are vectorized they naturally have
the required signature. It is therefore convenient to use Numpy and
Scipy functions directly, or to construct basis functions using them.

Consider the case of one-dimensional index set :math:`T = \mathbf{R}`,
and the monomial basis for the space of second-degree polynomials in one
variable, namely :math:`(1, x, x^{2})`, as follows. We may construct
the basis as follows.

.. sourcecode:: python

   >>> import numpy as np
   >>> basis = (np.polynomial.Polynomial([1.]), np.polynomial.Polynomial([0., 1.]), np.polynomial.Polynomial([0., 0., 1.]))

..
   Or, use the orthogonal Legendre basis for the space of second-degree
   polynomials in one variable, namely :math:`(1, x, (3x^{2} - 1)/2)`, as
   follows.

   .. sourcecode:: python

	 >>> basis = (np.polynomial.legendre.Legendre([1.]), np.polynomial.legendre.Legendre([1., 0.]), np.polynomial.legendre.Legendre([1., 0., 0.]))
 
Once we have initiated a :class:`Blup` class the coefficients of the
best linear unbiased estimator of the mean are stored as the attribute
:attr:`Beta`. Let us return to the example of the Forrester function
(:ref:`Quick-start tutorial <quickstart_tutorial>`). Generate a sample
of this function and initialize a :class:`Blup` class.

.. sourcecode:: python

   >>> ttrain = np.linspace(0., 1., 10)
   >>> xtrain = mim.testfunc.forrester(ttrain) + 0.5*np.random.randn(10)
   >>> blup = mim.Blup(ttrain, xtrain, 0.5**2., basis=basis)
   >>> blup.opt()
      direc: array([[-0.01099084,  0.00122937],
                    [ 0.00069455, -0.00117173]])
	fun: 29.366313678967565
    message: 'Optimization terminated successfully.'
       nfev: 418
	nit: 11
     status: 0
    success: True
	  x: array([1.49828726, 0.738667  ])
   
Now compute the best linear unbiased estimators of the means.

.. sourcecode:: python

   >>> ttrain = np.linspace(0., 1.)
   >>> blup.Beta@[fun(ttrain) for fun in basis]
   array([ 4.41578258, 0.73007516, ... , 12.72544472])

To construct the basis from scratch we would instead define three
separate functions.

.. sourcecode:: python

   >>> def fun_1(t):
   ...     return np.ones(len(t))
   ...
   >>> def fun_2(t):
   ...     return t
   ...
   >>> def fun_3(t):
   ...     return t**2.
   ...
   >>> basis = (fun_1, fun_2, fun_3)

If we pass these functions :math:`n` arguments they return Numpy
arrays of shape ``(n,)`` as required.

.. sourcecode:: python

   >>> t = np.random.rand(4)
   >>> fun_1(t)
   array([1., 1., 1., 1.])

Check this as follows.

.. sourcecode:: python
		
   >>> t.shape
   (4,)
   >>> fun_1(t).shape
   (4,)
   

References
##########

.. [C86]

   Cressie, N. 1986. \'Kriging nonstationary data\' in *Journal of the
   American Statistical Association*, 81 (395), 625--34. Available at
   https://doi.org/10.2307/2288990.
