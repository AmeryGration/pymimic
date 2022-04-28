.. _kernels:

************************************
Second-moment and covariance kernels
************************************

Whereas the BLP requires a second-moment kernel :math:`r`, given by
:math:`r(s, t) = \operatorname{E}(X_{s}X_{t})`, the BLUP requires a
covariance kernel :math:`k`, given by :math:`k(s, t) =
\operatorname{cov}(X_{s}, X_{t})`. The two are related by the equation

.. math::

   \operatorname{cov}(X_{s}, X_{t}) = \operatorname{E}(X_{s}X_{t}) - \operatorname{E}(X_{s})\operatorname{E}(X_{t}).

Recall that a function, :math:`f`, is a second-moment kernel or
covariance kernel if and only if it is positive semidefinite, i.e. if

.. math::

   \sum_{i, j}f(t_{i}, t_{j})u_{i}u_{j} \ge 0

for all :math:`t_{i}, t_{j} \in T` and all :math:`u_{i}, u_{j} \in
\mathbf{R}`.

Real multiples, sums and products of positive-semidefinite kernels are
also positive-semidefinite kernels, i.e. if :math:`f` and :math:`g` are
both positive-semidefinite kernels and :math:`a` and :math:`b` are
real numbers then :math:`afg`, and :math:`af + bg` are also
positive-semidefinite kernels.


.. _pymimic_secondmoment_and_covariance_kernels:

PyMimic, second-moment and covariance kernels
#############################################

The second-moment kernel or covariance kernel must be specified using
the keyword argument ``covfunc`` when creating an instance of the
:class:`Blp` or :class:`Blup` classes. It cannot be changed
afterwards. The kernel must be a function ``func(x, y, *args) ->
array_like`` with signature ``(n, d), (m, d) -> (n, m)`` where ``d``
is the dimension of the index set :math:`T` (:ref:`The two flavours of
linear prediction
<the_two_flavours_of_linear_prediction>`). Additional arguments,
required to fully specify ``covfunc``, may be passed to :class:`Blp`
or :class:`Blup` as a tuple, using the keyword argument ``args``.


Built-in positive-definite kernels
----------------------------------

The module :mod:`kernel` provides a suite of positive-definite
kernels, which may be used as second-moment or covariance kernels. The
available kernels are described in the submodule's API documentation
(:ref:`pymimic.kernel module <kernel_module>`).

Recall that when we emulated the Branin function (:ref:`Curve fitting
using the BLP <curve_fitting_using_the_blp>`) we defined a covariance
kernel :func:`kernel`. This was the squared-exponential covariance
kernel with parameter :math:`(16000., 0.08, 0.009)`. Instead of
defining the function :func:`kernel`, we might instead have used
:func:`kernel.squared_exponential` as follows.

.. sourcecode:: python
   
   >>> import pymimic as mim
   >>> mim.Blup(ttrain, xtrain, 10.**2., mim.kernel.squared_exponential, (16000., 0.08, 0.009))

By default ``covfunc`` is set to :func:`kernel.squared_exponential`.

User-defined positive-definite kernels
--------------------------------------

Because Numpy and Scipy functions are vectorized they naturally have
the required signature. It is therefore convenient to construct
positive-semidefinite kernels using Numpy and Scipy.

Consider the case of two-dimensional index set :math:`T = \mathbf{R}
\times \mathbf{R}`, and the positive-semidefinite kernel :math:`k: T
\times T \longrightarrow \mathbf{R}` given by

.. math::

   k(s, t) = \exp\left(-\dfrac{1}{2}(t - s)^{2}\right).

This is the standard squared-exponential kernel. We can implement it
as follows

.. sourcecode:: python

   >>> import numpy as np
   >>> from scipy.spatial.distance import cdist
   >>> def kernel(s, t):
           return np.exp(-0.5*cdist(s, t)**2.)

If we pass this function :math:`n` first arguments and :math:`m`
second arguments it returns a Numpy array of shape ``(n, m)`` as
required.

.. sourcecode:: python

   >>> s = np.random.rand(3, 2)
   >>> t = np.random.rand(4, 2)
   >>> kernel(s, t)
   array([[0.99223368, 0.95303202, 0.93866327, 0.80759156],
          [0.9137875 , 0.96735599, 0.78265123, 0.71452666],
	  [0.98832842, 0.99021078, 0.91337   , 0.83891139]])

Check this as follows.

.. sourcecode:: python
		
   >>> s.shape
   (3, 2)
   >>> t.shape
   (4, 2)
   >>> kernel(s, t).shape
   (3, 4)

We may form sums and products of existing kernels by wrapping them in
a new function. For example, we may form a kernel from the sum of two
squared-exponential kernels, each with a different variance and
length-scale, as follows.

.. sourcecode:: python

   >>> def kernel(s, t, *args):
           k_0 = mim.kernel.squared_exponential(s, t, *args[:3])
	   k_1 = mim.kernel.squared_exponential(s, t, *args[3:])
           return k_0 + k_1

Now call this function.

.. sourcecode:: python

   >>> args = (1., 1., 1., 0.1, 100., 100.)
   >>> kernel(s, t, *args)
   array([[0.87047874, 0.97016041, 0.64963388, 1.08806028],
          [0.93122309, 0.62856795, 0.97246196, 0.74156457],
          [0.85529846, 1.26246453, 0.61968782, 1.37247391]])
