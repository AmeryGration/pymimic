.. _pymimic_and_ego:

EGO and PyMimic
===============

Efficient global optimization is implemeted in PyMimic using the function
:func:`pymimic.ego`.
      
Let us find a global minimum of the Branin function. First, generate the data.

.. sourcecode:: python

   >>> import pymimic as mim
   >>> bounds = ((-5., 10.), (0., 15.))
   >>> xtrain = mim.design(bounds)
   >>> ytrain = [mim.testfunc.branin(xi) for xi in xtrain]

Second, perform the optimization. Set the keyword ``display`` to ``True`` to
print the result of each iteration as it is computed.

.. sourcecode:: python

   >>> mim.ego(mim.testfuncs.branin, bounds, xtrain, ytrain, disp=True)
   fun: 4.047823681677371610e-01
     x: array([9.390722228904179403, 2.409990724199692291])

Note that :func:`pymimic.ego` has found one of the function's three global
minima. It does not find all of them.
