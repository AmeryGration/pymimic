.. _pymimic_and_preprocessing:

Data preparation and PyMimic
============================

The PyMimic subpackage ``prep`` contains a suite of data processing tools.

Let us use it to make a Box-Cox transformation of a sample of the Branin
function. First generate the sample.

.. sourcecode:: python

   >>> bounds = ((-5., 10.), (0., 15.))
   >>> xtrain = mim.design(bounds)
   >>> ytrain = [mim.testfunc.branin(xi) for xi in xtrain]

Now perform the transformation.

.. sourcecode:: python
		
   >>> mim.prep.box_cox(ytrain)
   array([162.28409869682289, 43.27857137756608, 121.0729371399516,
           30.73679747563637,  9.58385860076313,  38.6368661684143,
	   19.82863628443500, 40.88213754302706,  11.9057597361944,
	   30.43603987080041,  1.28778195558889,  16.1042992935732,
	   74.72494184747205, 17.90497224007153,  56.5308680945326,
	   19.18796696462430, 26.40563743102657,  58.5968468187025,
	    1.85273044462076, 13.556408148164241]

The function :func:`prep.box_cox` automatically determines the most-likely
parameters of the transformation. However, we may optimize these directly,
using the function :func:`prep.box_cox_lmbda_mle`.

.. sourcecode:: python
		
   >>> mim.prep.box_cox_lmbda_mle(ytrain)
	fun: 25.072731979435833
	jac: array([ 0.00098801,  0.00547671])
    message:'Optimization terminated successfully.'
       nfev: 109
	nit: 24
       njev: 24
     status: 0
    success: True
	  x: array([-0.19994269, -1.20608108])

We may round the result to :math:`(0, -1)` and then perform the
transformation. This is the logarithmic transform with an offset of minus
one.

.. sourcecode:: python

   >>> mim.prep.box_cox(ytrain, (0, -1))
   array([  5.0831674 ,  3.74428037,  4.78809937,  3.39238525,  2.14988353,
	    3.62798405,  2.93537892,  3.68592854,  2.38929107,  3.38221977,
	   -1.24555218,  2.71497942,  4.30034117,  2.82760779,  4.01693905,
	    2.90076022,  3.2349711 ,  4.05346782, -0.15931179,  2.53023114])

It is possible to invert the transformation using the function
:func:`pymimic.prep.box_cox_inv`.

.. sourcecode:: python

   >>> mim.prep.box_cox_inv(_, (0., -1))
   array([ 438.41563471,  114.92507231,  326.39208312,   80.83299621,
	    23.33334685,  102.30760939,   51.18153987,  108.41088976,
	    29.64492852,   80.01545228,    0.78227246,   41.0577423 ,
	   200.40516973,   45.95247885,  150.94854966,   49.4400201 ,
	    69.05968257,  156.56446208,    2.31796167,   34.1318561 ])
