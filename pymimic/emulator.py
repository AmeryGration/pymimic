#!/usr/bin/env python3

"""Emulator module for PyMimic

This module provides :class:`Blp` and :class:`Blup` classes for use in
computing the best linear predictor and best linear unbiased predictor
of a random variable based on a finite random process.

"""

import sys
import warnings
import numpy as np
import scipy.optimize as optimize
import scipy.linalg as linalg
import pymimic as mim

def _dot(x, y):
    r"""Return the column-wise dot product of two arrays"""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    try:
        # If x and y are not scalars
        return np.einsum("ji, ji -> i", x, y)
    except ValueError:
        # If x and y are scalars
        return np.einsum("i, i -> i", x, y)

def _opt(linear_predictor, opt_method="mle", n_start=None,
         method="Powell", bounds=None, tol=None, callback=None,
         options=None):
    r"""Optimize the parameter of the covariance kernel model"""
    class _Obj:
        def __init__(self, opt_method):
            if opt_method == "mle":
                self.obj = self._obj_lnL
            elif opt_method == "loocv":
                self.obj = self._obj_R2
            else:
                raise ValueError(
                    "unknown optimization method {}.".format(opt_method)
                )

        def __call__(self, x):
            return self.obj(x)

        def _obj_R2(self, x):
            r"""Return sum of squares of LOO residuals for given covfunc arg"""
            linear_predictor.args = tuple(
                [theta[i](x[i]) for i in range(len(x))]
            )
            return linear_predictor.R2

        def _obj_lnL(self, x):
            r"""Return negative of log-likelihood of covfunc arg"""
            # TO DO Be able to pass arbitrary probability distribution
            linear_predictor.args = tuple(
                [theta[i](x[i]) for i in range(len(x))]
            )
            return - linear_predictor.lnL

    class _Param:
        r"""Parameter transformation"""
        def __init__(self, inf_bound=False):
            if inf_bound:
                self.trans = np.tan
            else:
                self.trans = lambda x: np.sinh(10.*x)/10.

        def __call__(self, x):
            return self.trans(x)

    # Specify objective function
    obj = _Obj(opt_method)
    # Set bounds
    if bounds:
        linear_predictor.argbounds = bounds
    elif linear_predictor.argbounds is None:
        linear_predictor.argbounds = linear_predictor._argbounds()
    bounds = np.copy(linear_predictor.argbounds)
    # Transform infinite search bounds to finite using arctan.
    ind_infinite = np.isinf(bounds)
    bounds[ind_infinite] = np.pi/2.
    # Transform finite search bounds to finite using arcsinh
    row_infinite = np.any(ind_infinite, axis=1)
    bounds[~row_infinite] = np.arcsinh(10.*bounds[~row_infinite])/10.
    # Transform parameter
    theta = [_Param(val) for val in row_infinite]
    # Multistart optimization
    if n_start is None:
        n_start = 10*linear_predictor.ndim
    try:
        # If LHS design is available as a file (i.e. if self.n_start >
        # self.ndim)
        x0 = mim.design(bounds, "lhs", n=n_start)
    except ValueError:
        # If LHS design is not available as a file
        x0 = mim.design(bounds, "random", n=n_start)
    fun_best = np.inf
    for i, x0_i in enumerate(x0):
        try:
            res = optimize.minimize(obj, x0=x0_i, method=method,
                                    jac=None, hess=None, hessp=None,
                                    bounds=bounds, tol=tol,
                                    callback=callback,
                                    options=options)
            if res.success and res.fun < fun_best:
                res_best = res
                # args_best = linear_predictor.args
                vars_best = vars(linear_predictor)
            else:
                warnings.warn("multistart iteration failed.", UserWarning)
        except ValueError:
            pass
    # linear_predictor.args = args_best
    for key, val in vars_best.items():
        setattr(linear_predictor, key, val)
    return res_best

class _CovfuncWrapper:
    r"""Wrap positive-definite kernel so it can take additional arguments"""
    def __init__(self, f, args):
        self.f = f
        if args is None:
            self.args = ()
        elif np.isscalar(args):
            self.args = (args,)
        else:
            self.args = args
        
    def __call__(self, s, t):
        return self.f(s, t, *self.args)

class _LinearPredictor():
    r"""Base class for classes Blp and Blup"""
    # This class allows for dependent attributes to be recomputed if
    # and only if the independent attributes on which they depend are
    # changed. See: https://stackoverflow.com/questions/57439169.
    _independent_attribs = {"args"}
    _dependent_attribs = {"kernel", "K", "U_K"}
    def __init__(self, ttrain, xtrain, var_error=None,
                 covfunc=mim.kernel.squared_exponential, args=()):
        self._ttrain = ttrain
        self._xtrain = xtrain
        self._var_error = var_error
        self._covfunc = covfunc
        self._args = args
        self._kernel = None
        self._K = None
        self._U_K = None
        self.argbounds = None
        
    def __setattr__(self, name, value):
        # Update dependent attribute if and only if self.args changes
        if name in self._independent_attribs:
            name = f"_{name}"
            for var in self._dependent_attribs:
                super().__setattr__(f"_{var}", None)
        super().__setattr__(name, value)

    @property
    def ttrain(self):
        r"""Training inputs"""
        return self._ttrain

    @property
    def ntrain(self):
        r"""Number of training data"""
        return len(self._ttrain)

    @property
    def ndim(self):
        r"""Dimension of the training inputs"""
        try:
            return self._ttrain.shape[1]
        except IndexError:
            return 1

    @property
    def xtrain(self):
        r"""Training outputs"""
        return self._xtrain

    @property
    def var_error(self):
        r"""Variance of the error in the training outputs"""
        if isinstance(self._var_error, (list, tuple, np.ndarray)):
            # If var_error is array_like
            return self._var_error
        else:
            # If var_error is None or scalar
            if self._var_error is None:
                self._var_error = 0.
            return self._var_error*np.ones(self.ntrain)
        
    @property
    def args(self):
        r"""Additional arguments required to specify the `covfunc`"""
        return self._args
    
    @property
    def kernel(self):
        r"""Wrapper for `covfunc`"""
        if self._kernel is None:
            self._kernel = _CovfuncWrapper(self._covfunc, self.args)
        return self._kernel

    @property
    def K(self):
        r"""Covariance matrix"""
        if self._K is None:
            K = self.kernel(self.ttrain, self.ttrain)
            if self.var_error is not None:
                K = K + np.diag(self.var_error)
            self._K = K
        return self._K
        
    @property
    def U_K(self):
        r"""Cholesky decomposition of the covariance matrix"""
        if self._U_K is None:
            self._U_K = linalg.cho_factor(self.K)
        return self._U_K
        
    def _argbounds(self):
        """Return kernel parameter bounds"""
        domain = np.max(self.ttrain, axis=0) - np.min(self.ttrain, axis=0)
        l_min = (np.product(domain)/self.ntrain)**(1./self.ndim)
        if self._covfunc is mim.kernel.exponential:
            return np.vstack(
                ([0., np.inf],
                 [0., 2.    ],
                 np.repeat([[0., l_min**(-2.)]], self.ndim, axis=0))
            )
        elif self._covfunc is mim.kernel.gneiting:
            return np.vstack(
                ([0., np.inf],
                 [0., 2.    ],
                 np.repeat([[0., l_min**(-2.)]], self.ndim, axis=0))
            )
        elif self._covfunc is mim.kernel.matern:
            return np.vstack(
                ([0., np.inf],
                 [0., np.inf],
                 np.repeat([[0., l_min**(-2.)]], self.ndim, axis=0))
            )
        elif self._covfunc is mim.kernel.neural_network:
            return np.vstack(
                ([0., np.inf],
                 np.repeat([[0., l_min**(-2.)]], self.ndim + 1, axis=0))
            )
        elif self._covfunc is mim.kernel.periodic:
            return np.vstack(
                ([0., np.inf],
                 np.repeat([[l_min, np.inf]], self.ndim, axis=0),
                 np.repeat([[0., l_min**(-2.)]], self.ndim, axis=0))
            )
        elif self._covfunc is mim.kernel.rational_quadratic:
            return np.vstack(
                ([0., np.inf],
                 [0., np.inf],
                 np.repeat([[0., l_min**(-2.)]], self.ndim, axis=0))
            )
        elif self._covfunc is mim.kernel.squared_exponential:
            return np.vstack(
                ([0., np.inf],
                 np.repeat([[0., l_min**(-2.)]], self.ndim, axis=0))
            )
        else:
            raise TypeError(
                "unknown covariance kernel."
            )

class Blp(_LinearPredictor):
    r"""Best linear predictor of a random variable

    Let :math:`X := \{X_{t}\}_{t \in T}` be a random process
    (considered as a column vector) with finite index set :math:`T`
    and second-moment kernel :math:`k`. Let :math:`Z` be a random
    variable to which the second-moment kernel extends by writing
    :math:`Z := X_{t_{0}}`. The best linear predictor (BLP) of
    :math:`Z` based on :math:`X` is

    .. math::

       Z^{*} = \sigma{}^{\mathrm{t}}K^{-1}X

    with mean-squared error

    .. math::

       \operatorname{MSE}(Z^{*}) = k(t_{0}, t_{0}) -
       \sigma^{\mathrm{t}}K^{-1}\sigma,

    where :math:`\sigma := (k(t_{*}, t_{i}))_{i}` and :math:`K :=
    (k(t_{i}, t_{j}))_{ij}`.
       
    Given a realization of :math:`X`, namely :math:`x`, then we may
    compute the realization of :math:`Z^{*}`, namely :math:`z^{*}`, by
    making the substitution :math:`X = x`.

    Parameters
    ----------

    ttrain : (n, d) array_like

        Training input.

    xtrain : (n,) array_like

        Training output.

    var_error : scalar or (n,) array_like, optional

        Variance of error in training output.

    covfunc : callable, optional

        Second-Moment kernel, ``covfunc(s, t, *args) -> array_like``
        with signature ``(n, d), (m, d) -> (n, m)``, where ``args`` is
        a tuple needed to completely specify the function.

    args : (p,) tuple, optional

        Extra arguments to be passed to the second-moment kernel,
        ``covfunc``.

    Attributes
    ----------

    ttrain : (n, d) array_like

        Training inputs

    xtrain : (n,) array_like

        Training outputs

    ntrain : int

        Number of training data

    ndim : int

        Dimension of the training inputs

    var_error : bool, scalar, (n,) array_like, optional

        Variance of the error in the training outputs

    args : tuple

        Additional arguments required to specify ``covfunc``

    kernel : callable

        Second-moment kernel, additional argument spedified

    K : (n, n) ndarray

        Second-moment matrix for the training inputs.

    argbounds : (m, n) ndarray or None

        Bounds of the parameter space.
    
    loocv : (n,) ndarray

        Leave-one-out residuals.
    
    R2 : float

        Leave-one-out cross-validation score.
    
    lnL : float

        Log-likelihood of the parameter tuple under the assumption
        that training data are a realization of a Gaussian random
        process.

    Raises
    ------

    Warns
    -----

    Warnings
    --------

    If the matrix :math:`K` is singular then the BLP does not exist.
    This happens when any two elements of the random process :math:`X`
    are perfectly correlated (in this case the rows of :math:`K` are
    linearly dependent).  If the matrix :math:`K` is nearly-singular
    rather than singular then the BLP is numerically unstable. This
    happens when any two elements of the random process :math:`X` are
    highly correlated rather than perfectly correlated (in this case
    the rows of :math:`K` are nearly linearly dependent). There are
    several possible solutions to these problems:

    (1) remove all but one of the offending elements of the training data;

    (2) resample the random process using different training inputs;

    (3) use a different second-moment kernel;

    (4) add random noise to the the training output.

    Optimization of the BLP (implemented by the class method
    :func:`Blp.opt`) will also fail when the second-moment matrix
    becomes nearly singular.

    See also
    --------

    :class:`Blup`

        Best linear unbiased predictor.

    Notes
    -----

    The quantities :math:`z^{*}` and :math:`\operatorname{MSE}(Z^{*})`
    are computed using the class method :func:`Blp.xtest`.

    References
    ----------

    Examples
    --------

    One-dimensional index set:

    .. sourcecode:: python

       >>> ttrain = [0., 1.]
       >>> xtrain = [0., 1.]
       >>> mim.Blp(ttrain, xtrain, args=[1., 1.])
       <pymimic.emulator.Blp object at 0x7f87c9f8af28>

    Second-Moment kernel provided explicitly:

       >>> mim.Blp(ttrain, xtrain, covfunc=mim.kernel.squared_exponential, args=[1., 1.])
       <pymimic.emulator.Blp object at 0x7f1c519c0a20>

    Two-dimensional index set:

    .. sourcecode:: python

       >>> ttrain = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
       >>> xtrain = [0., 1., 1., 2.]
       >>> mim.Blp(ttrain, xtrain, args=[1., 1., 1.])
       <pymimic.emulator.Blp object at 0x7f1c519c0940>

    Second-Moment kernel provided explicitly:

    .. sourcecode:: python

       >>> mim.Blp(ttrain, xtrain, covfunc=mim.kernel.squared_exponential, args=[1., 1., 1.])
       <pymimic.emulator.Blp object at 0x7f970017fd68>

    """
    _independent_attribs = {"args"}
    _dependent_attribs = {"kernel", "K", "U_K", "Gamma", "loocv",
                          "R2", "lnL"}
    def __init__(self, ttrain, xtrain, var_error=None,
                 covfunc=mim.kernel.squared_exponential, args=()):
        _LinearPredictor.__init__(self, ttrain, xtrain, var_error,
                                  covfunc, args)
        self._Gamma = None
        self._loocv = None
        self._R2 = None
        self._lnL = None

    @property
    def Gamma(self):
        r"""Matrix Gamma 

        :meta private:

        """
        if self._Gamma is None:
            self._Gamma = linalg.cho_solve(self.U_K, self.xtrain)
        return self._Gamma

    @property
    def loocv(self):
        r"""LOO residuals and their variances 

        :meta private:

        """
        if self._loocv is None:
            var_residuals = 1./np.diag(np.linalg.inv(self.K))
            residuals = self.Gamma*var_residuals
            self._loocv = residuals, var_residuals
        return self._loocv

    @property
    def R2(self):
        r"""Leave-one-out cross-validation score 

        :meta private:

        """
        if self._R2 is None:
            try:
                self._R2 = np.sum(self.loocv[0]**2.)/self.ntrain
            except (linalg.LinAlgError, ValueError):
                self._R2 = np.inf
        return self._R2

    @property
    def lnL(self):
        r"""Log-likelihood of the covariance function parameter

        :meta private:

        """
        if self._lnL is None:
            try:
                log_det_K = 2.*np.sum(np.log(np.diag(self.U_K[0])))
                Q = linalg.cho_solve(self.U_K, self.xtrain)
                self._lnL = - self.xtrain.T@Q - log_det_K
            except (linalg.LinAlgError, ValueError):
                self._lnL = - np.inf
        return self._lnL
    
    def opt(self, opt_method="mle", n_start=None, method="Powell",
            bounds=None, tol=None, callback=None, options=None):
        r"""Optimize the parameter of the second-moment kernel model

        Set the second-moment kernel arguments, ``*args``, to their optimum
        values using multistart local optimization. 

        If the second-moment kernel is user defined ``*args`` must be
        a sequence of scalars.

        Parameters 
        ----------

        opt_method : str, optional

            Optimization method to be used. Should be either:

                - ``"mle"`` (maximum-likelihood method)

                - ``"loocv"`` (leave-one-out cross-validation method).

        n_start : int, optional

            Number of multistart iterations to be used. If ``None``
            then :math:`10d` iterations are used, where :math:`d` is
            the dimension of the parameter space.

        method : str or callable, optional

            Type of solver to be used by the minimizer.

        bounds : sequence or scipy.optimize.Bounds, optional

            Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
            Powell, and trust-constr methods. There are two ways to
            specify the bounds:

                - instance of scipy.optimize.Bounds class;

                - sequence of `(min, max)` pairs for each parameter.

            Bounds must be provided if the :func:`covfunc` is not
            included in the PyMimic module.

        tol : float, optional

            Tolerance for termination. When `tol` is specified, the
            selected minimization algorithm sets some relevant
            solver-specific tolerance(s) equal to `tol`.

        callback : callable, optional

            A function to be called after each iteration.

        options : dict, optional

            A dictionary of solver options.

        Returns 
        -------

        res : OptimizeResult

            The best of `n_start` optimization results, represented as
            an `OptimizeResult` object.

        Raises 
        ------

        Warns
        -----

        Warnings
        --------

        Suppose that the second-moment kernel model is of the form
        :math:`k(s, t) = a{}f(s, t)` for some constant :math:`a` and
        some positive-semidefinite kernel :math:`f`. In this case the
        method of leave-one-out cross-validation will fail to find
        :math:`a` if there are no errors on the training outputs
        (i.e. if ``var_error`` is ``None``, zero, or a tuple of
        zeros). This is because the BLP of a random variable is
        independent of :math:`a`.

        Typically, the likelihood and leave-one-out cross-validation
        score have multiple extrema. Use :func:`Blp.opt` in the
        knowledge that the global optimum may not be found, and that
        the global optimum may in any case result in over- or
        under-fitting of the data [RW06_].

        See also
        --------

        :func:`scipy.optimize.minimize`

           The Scipy function for the minimization of a scalar
           function of one or more variables.

        Notes
        -----

        The function optimizes the parameter of the second-moment kernel
        model using either (i) the method of maximum likelihood or
        (ii) the method of leave-one-out cross-validation.
        
        In the first case, the likelihood of the parameter is
        maximized under the assumption that training and test data are
        a realization of a centred Gaussian random process.

        In the second case, the parameter is chosen so as to minimize
        the leave-one-out cross-validation score (i.e. the sum of the
        squares of the leave-one-out residuals).

        The optimization is done using multiple starts of
        :meth:`scipy.optimize.minimize`. The keywords of
        :meth:`Blp.opt()` (``method``, ``bounds``, ``tol``,
        ``callback``, and ``options``) are passed directly to this
        function. The starting points are chosen using Latin
        hypersquare sampling, unless the training data are scalars or
        the number of training data is greater than the the number of
        starts, when uniform random sampling is used.

        The optimization is performed in a transformed parameter
        space. If either bound for an element of the parameter tuple
        is infinite (``np.inf``) then that element is transformed
        using :math:`\mathrm{tan}`. If both bounds for an element of
        the parameter tuple are finite then that element is
        transformed using :math:`\mathrm{sinh}(10\,\cdot)/10`.

        References
        ----------

        .. [RW06]

           Rasmussen, C.E., and C.K.I. Williams. 2006. *Gaussian
           process emulation*. Cambridge: MIT Press.

        Examples
        --------

        First create an instance of :class:`Blp`.

        .. sourcecode:: python

           >>> ttrain = np.linspace(0., np.pi, 10)
           >>> xtrain = np.sin(ttrain)
           >>> blp = mim.Blp(ttrain, xtrain)

        Now perform the optimization.

           >>> blp.opt() 
                fun: -52.99392000860139
           hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
                jac: array([276.77485929, 585.7648087 ])
            message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
               nfev: 9
                nit: 1
               njev: 3
             status: 0
            success: True
                  x: array([1.49225651, 0.23901602])

        Inspect the instance attribute ``args`` to see that it has
        been set to its maximum-likelihood estimate.

        .. sourcecode:: python

           >>> blp.args
           [12.706204736174696, 0.5411814081739366]

        See Notes for an explanation of the discrepancy between the
        values of ``x`` and ``blp.args``.

        """
        return _opt(self, opt_method, n_start, method, bounds, tol,
                    callback, options)

    def xtest(self, t):
        r"""Return the best linear predictor of a random variable.

        Returns the realization, :math:`z^{*}`, of the best linear
        predictor of a random variable based on the realization of a
        finite random process, :math:`x`.

        Parameters
        ----------

        t : (m, d) array_like

           Test input.

        Returns
        -------

        xtest : (m,) ndarray

           Test output.

        mse : (m,) ndarray, optional

           Mean-squared error in the test output.

        Raises
        ------

        Warns
        -----

        Warnings
        --------
        
        See also
        --------

        :func:`Blup.xtest`

           Best linear predictor

        Notes
        -----        

        Examples
        --------

        One-dimensional index set, one test input:

        .. sourcecode:: python

           >>> ttrain = [0., 1.]
           >>> xtrain = [0., 1.]
           >>> blp = mim.Blp(ttrain, xtrain, args=[1., 1.])
           >>> blp.xtest(0.62)
           (array(0.6800459), array(0.02680896))

        Two test inputs:

        .. sourcecode:: python

           >>> blp.xtest([0.62, 0.89])
           (array([0.6800459, 0.92670531]), array([0.02680896, 0.00425306]))

        Two-dimensional index set, one test input:

        .. sourcecode:: python

           >>> ttrain = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
           >>> xtrain = [0., 1., 1., 2.]
           >>> blp = Blp(ttrain, xtrain, args=[1., 1., 1.])
           >>> blp.xtest([[0.43, 0.27]])
           (array([0.82067282]), array([0.04697306]))

        Two test inputs at once:

        .. sourcecode:: python

           >>> blp.xtest([[0.43, 0.27], [0.16, 0.93]])
           (array([0.82067282, 1.17355541]), array([0.04697306, 0.0100251]))

        """
        if np.isscalar(t):
            # Allow for a single scalar index
            t = np.atleast_1d(t)
            scalar_input = True
        else:
            # Normally, there will be multiple scalar or vector indices
            scalar_input = False
        # BLP
        k = self.kernel(self.ttrain, t)
        # k = np.atleast_2d(self.kernel(self.ttrain, t))
        xtest = k.T@self.Gamma
        # MSE of BLP
        alpha = np.linalg.solve(self.K, k)
        mse = np.diag(self.kernel(t, t)) - _dot(alpha, k)
        # mse = ([self.kernel(t_i, t_i).item() for t_i in t]
        #        - _dot(alpha, k))
        # A very small MSE may be computed as slightly negative due
        # to a lack of numerical precision
        mse[mse<0.] = 0.
        if scalar_input:
            return np.squeeze(xtest), np.squeeze(mse)
        return xtest, mse

class Blup(_LinearPredictor):
    r"""Best linear unbiased predictor of a random variable [G62_].

    Let :math:`X := \{X_{t}\}_{t \in T}` be a random process
    (considered as a column vector) with finite index set :math:`T`
    and covariance kernel :math:`k`. Let :math:`Z` be a random
    variable to which the covariance kernel extends by writing
    :math:`Z := X_{t_{0}}`. Let :math:`(\phi_{i})_{i}` be a basis for
    the space of possible mean-value functions for :math:`Z`. The best
    linear unbiased predictor (BLUP) of :math:`Z` based on :math:`X` is


    .. math::

       Z^{\dagger} = \Theta^{\dagger} + \sigma^{\mathrm{t}}K^{-1}\mathrm{D}

    where
    :math:`\sigma := (k(t_{0}, t_{i}))_{i}`,
    :math:`K := (k(t_{i}, t_{j}))_{ij}`,
    :math:`\Theta^{\dagger} = \phi^{\mathrm{t}}\mathrm{B}`,
    :math:`\mathrm{D} := X - \Phi\mathrm{B}`, and where 
    :math:`\phi = (f_{i}(t_{0}))_{i}`,
    :math:`\Phi := (f_{i}(t_{j}))_{ij}`, and
    :math:`\mathrm{B} :=
    (\Phi^{\mathrm{t}}K^{\,-1}\Phi)^{-1}\Phi^{\mathrm{t}}K^{\,-1}X`.

    Given a realization of :math:`X`, namely :math:`x`, then we may
    compute the realization of :math:`Z^{\dagger}`, namely
    :math:`z^{\dagger}`, by making the substitution :math:`X = x`.

    Parameters
    ----------

    ttrain : (n, d) array_like

        Training input.

    xtrain : (n,) array_like

        Training output.

    var_error : scalar or (n,) array_like, optional

        Variance of error in training output.

    covfunc : callable, optional

        Covariance kernel, ``covfunc(s, t, *args) -> array_like`` with
        signature ``(n, d), (m, d) -> (n, m)``, where ``args`` is a
        tuple needed to completely specify the function.

    args : (q,) tuple, optional

        Extra arguments to be passed to the covariance kernel,
        ``covfunc``.

    basis: (p,) tuple, optional

        Basis for the space of mean-value functions. Each element of
        the tuple is a basis function, namely a callable object,
        ``fun(t) -> array_like`` with signature ``(n, d) -> (n,)``.
        

    Attributes
    ----------

    ttrain : (n, d) array_like

        Training inputs

    xtrain : (n,) array_like

        Training outputs

    ntrain : int

        Number of training data

    ndim : int

        Dimension of the training inputs

    var_error : bool, scalar, (n,) array_like, optional

        Variance of the error in the training outputs

    args : tuple

        Additional arguments required to specify ``covfunc```

    kernel : callable

        Covariance kernel, additional argument spedified

    K : (n, n) ndarray

        Covariance matrix for the training inputs.
    
    argbounds : (m, n) ndarray or None

        Bounds of the parameter space.

    Phi : (n, p) ndarray

        Basis matrix.

    Beta : (p,) ndarray

        Best linear unbiased estimator of the coefficients of the
        basis functions.

    Delta: (n,) ndarray

        Residuals of fit of BLUE to training data.

    loocv : (n,) ndarray

        Leave-one-out residuals.
    
    R2 : float

        Leave-one-out cross-validation score.
    
    lnL : float

        Log-likelihood of the parameter tuple under the assumption
        that training data are a realization of a Gaussian random
        process.


    Raises
    ------

    Warns
    -----

    Warnings
    --------

    If the matrix :math:`K` is singular then the BLUP does not exist.
    This happens when any two elements of the random process :math:`X`
    are perfectly correlated (in this case the rows of :math:`K` are
    linearly dependent).  If the matrix :math:`K` is nearly-singular
    rather than singular then the BLUP is numerically unstable. This
    happens when any two elements of the random process :math:`X` are
    highly correlated rather than perfectly correlated (in this case
    the rows of :math:`K` are nearly linearly dependent). There are
    several possible solutions to these problems:

    (1) remove all but one of the offending elements of the training data;

    (2) resample the random process using different training inputs;

    (3) use a different covariance kernel;

    (4) add random noise to the the training output.

    Optimization of the BLUP (implemented by the class method
    :func:`Blup.opt`) will also fail when the covariance matrix
    becomes nearly singular.

    See also
    --------

    :class:`Blp`

       Best linear predictor.

    Notes
    -----

    The quantities :math:`z^{\dagger}` and
    :math:`\operatorname{MSE}(Z^{\dagger})` are computed using the
    class method :func:`Blp.xtest`.

    References
    ----------

    .. [G62]

       Goldberger, A. S. 1962. \'Best linear unbiased prediction in
       the generalized linear regression model\' in *Journal of the
       American Statistical Association*, 57 (298): 369--75. Available
       at https://www.doi.org/10.1080/01621459.1962.10480665.

    Examples
    --------

    One-dimensional index set:

    .. sourcecode:: python

       >>> ttrain = [0., 1.]
       >>> xtrain = [0., 1.]
       >>> mim.Blup(ttrain, xtrain, args=[1., 1.])
       <pymimic.emulator.Blup object at 0x7f87c9f8af28>

    Covariance kernel function provided explicitly:

       >>> mim.Blup(ttrain, xtrain, covfunc=mim.kernel.squared_exponential, args=[1., 1.])
       <pymimic.emulator.Blup object at 0x7f1c519c0a20>

    Two-dimensional index set:

    .. sourcecode:: python

       >>> ttrain = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
       >>> xtrain = [0., 1., 1., 2.]
       >>> mim.Blup(ttrain, xtrain, args=[1., 1., 1.])
       <pymimic.emulator.Blp object at 0x7f1c519c0940>

    Covariance kernel function provided explicitly:

    .. sourcecode:: python

       >>> mim.Blup(ttrain, xtrain, covfunc=mim.kernel.squared_exponential, args=[1., 1., 1.])
       <pymimic.emulator.Blp object at 0x7f970017fd68>

    """
    _independent_attribs = {"args"}
    _dependent_attribs = {"kernel", "K", "U_K", "alpha", "beta",
                          "gamma", "G", "U_G", "Beta", "Delta",
                          "Gamma", "Q", "loocv", "R2", "lnL"}
    def __init__(self, ttrain, xtrain, var_error=None,
                 covfunc=mim.kernel.squared_exponential, args=(),
                 basis=mim.basis.const()):
        _LinearPredictor.__init__(self, ttrain, xtrain, var_error,
                                  covfunc, args)
        self._alpha = None
        self._beta = None
        self._gamma = None
        self._G = None
        self._U_G = None
        self._Beta = None
        self._Delta = None
        self._Gamma = None
        self._Q = None
        self._loocv = None
        self._R2 = None
        self._lnL = None
        self.basis = basis
        self.Phi = self._phi(ttrain)
        
    @property
    def alpha(self):
        r"""Matrix alpha

        :meta private:

        """
        if self._alpha is None:
            self._alpha = linalg.cho_solve(self.U_K, self.Phi)
        return self._alpha

    @property
    def beta(self):
        r"""Matrix beta

        :meta private:
        
        """
        if self._beta is None:
            self._beta = linalg.cho_solve(self.U_G, self.alpha.T)
        return self._beta

    @property
    def gamma(self):
        r"""Matrix gamma

        :meta private:

        """
        if self._gamma is None:
            self._gamma = np.eye(self.ntrain) - self.Phi@self.beta
        return self._gamma

    @property
    def Beta(self):
        r"""Coefficients of best linear unbiased estimator

        :meta private:

        """
        if self._Beta is None:
            self._Beta = self.beta@self.xtrain
        return self._Beta
        
    @property
    def G(self):
        r"""Gram matrix
        
        :meta private:

        """
        if self._G is None:
            self._G =  self.Phi.T@self.alpha
        return self._G

    @property
    def U_G(self):
        r"""Cholesky decomposition of the Gram matrix

        :meta private:

        """
        if self._U_G is None:
            self._U_G = linalg.cho_factor(self.G)
        return self._U_G

    @property
    def Delta(self):
        r"""Residuals of generalized least-square fit

        :meta private:

        """
        if self._Delta is None:
            self._Delta = self.gamma@self.xtrain
        return self._Delta

    @property
    def Gamma(self):
        r"""Matrix Gamma

        :meta private:

        """
        if self._Gamma is None:
            self._Gamma = self.Q@self.xtrain
        return self._Gamma

    @property
    def Q(self):
        r"""Matrix Q

        :meta private:

        """
        if self._Q is None:
            self._Q = linalg.cho_solve(self.U_K, self.gamma)
        return self._Q

    @property
    def loocv(self):
        r"""LOO residuals and their variances

        :meta private:

        """
        if self._loocv is None:
            var_residuals = 1./np.diag(self.Q)
            residuals = self.Gamma*var_residuals
            self._loocv = residuals, var_residuals
        return self._loocv

    @property
    def R2(self):
        r"""Leave-one-out cross-validation score

        :meta private:

        """
        if self._R2 is None:
            try:
                self._R2 = np.sum(self.loocv[0]**2.)/self.ntrain
            except (linalg.LinAlgError, ValueError):
                self._R2 = np.inf
        return self._R2

    @property
    def lnL(self):
        r"""Log-likelihood of the covariance function parameter

        :meta private:

        """
        # Assumes the random process is Gaussian
        if self._lnL is None:
            try:
                log_det_K = 2.*np.sum(np.log(np.diag(self.U_K[0])))
                Q = linalg.cho_solve(self.U_K, self.Delta)
                self._lnL = - self.Delta.T@Q - log_det_K
            except (linalg.LinAlgError, ValueError):
                self._lnL = - np.inf
        return self._lnL

    def _phi(self, t):
        r"""Return the basis function values for given argument"""
        return np.array([basis_j(t) for basis_j in self.basis]).T

    def opt(self, opt_method="mle", n_start=None, method="Powell",
            bounds=None, tol=None, callback=None, options=None):
        r"""Optimize the parameter of the covariance kernel model

        Set the covariance kernel arguments, ``*args``, to their
        optimum values using multistart local optimization.

        Parameters 
        ----------

        opt_method : str, optional

            Optimization method to be used. Should be either:

                - ``"mle"`` (maximum-concentrated likelihood method)

                - ``"loocv"`` (leave-one-out cross-validation method).

        n_start : int, optional

            Number of multistart iterations to be used. If ``None``
            then :math:`10d` iterations are used, where :math:`d` is
            the dimension of the parameter space.

        method : str or callable, optional

            Type of solver to be used by the minimizer.

        bounds : sequence or scipy.optimize.Bounds, optional

            Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
            Powell, and trust-constr methods. There are two ways to
            specify the bounds:

                - instance of scipy.optimize.Bounds class;

                - sequence of `(min, max)` pairs for each parameter.

            Bounds must be provided if the :func:`covfunc` is not
            included in the PyMimic module.

        tol : float, optional

            Tolerance for termination. When `tol` is specified, the
            selected minimization algorithm sets some relevant
            solver-specific tolerance(s) equal to `tol`.

        callback : callable, optional

            A function to be called after each iteration.

        options : dict, optional

            A dictionary of solver options.

        Returns 
        -------

        res : OptimizeResult

            The best of `n_start` optimization results, represented as
            an `OptimizeResult` object.

        Raises 
        ------

        Warns
        -----

        Warnings
        --------

        Suppose that the covariance kernel model is of the form
        :math:`k(s, t) = a{}f(s, t)` for some constant :math:`a` and
        some positive-semidefinite kernel :math:`f`. In this case the
        method of leave-one-out cross-validation will fail to find
        :math:`a` if there are no errors on the training outputs
        (i.e. if ``var_error`` is ``None``, zero, or a tuple of
        zeros). This is because the BLUP of a random variable is
        independent of :math:`a`.

        Typically, the likelihood and leave-one-out cross-validation
        score have multiple extrema. Use :meth:`Blup.opt()` in the
        knowledge that the global optimum may not be found, and that
        the global optimum may in fact result in over- or
        under-fitting of the data [RW06_].

        See also
        --------

        :func:`scipy.optimize.minimize`

           The Scipy function for the minimization of a scalar function of
           one or more variables.

        Notes
        -----

        The function optimizes the parameter of the covariance kernel
        model using either (i) the method of maximum-concentrated
        likelihood or (ii) the method of leave-one-out
        cross-validation.
        
        In the first case, the likelihood of the parameter is
        maximized under the assumption that training and test data are
        a realization of a Gaussian random process with mean given by
        its best linear unbiased estimator.

        In the second case, the parameter is chosen so as to minimize
        the leave-one-out cross-validation score (i.e. the sum of the
        squares of the leave-one-out residuals).

        The optimization is done using multiple starts of
        :meth:`scipy.optimize.minimize`. The keywords of
        :meth:`Blup.opt()` (``method``, ``bounds``, ``tol``,
        ``callback``, and ``options``) are passed directly to this
        function. The starting points are chosen using Latin
        hypersquare sampling, unless the training data are scalars or
        the number of training data is greater than the the number of
        starts, when uniform random sampling is used.

        The optimization is performed in a transformed parameter
        space. If either bound for an element of the parameter tuple
        is infinite (``np.inf``) then that element is transformed
        using :math:`\mathrm{tan}`. If both bounds for an element of
        the parameter tuple are finite then that element is
        transformed using :math:`\mathrm{sinh}(10\cdot)/10`.

        References
        ----------

        .. [RW06]

           Rasmussen, C.E., and C.K.I. Williams. 2006. *Gaussian
           process emulation*. Cambridge: MIT Press.

        Examples
        --------

        First create an instance of :class:`Blup`.

        .. sourcecode:: python

           >>> ttrain = np.linspace(0., np.pi, 10)
           >>> xtrain = np.sin(ttrain)
           >>> blup = mim.Blup(ttrain, xtrain)

        Now perform the optimization.

           >>> blup.opt() 
                fun: -52.99392000860139
           hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
                jac: array([276.77485929, 585.7648087 ])
            message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
               nfev: 9
                nit: 1
               njev: 3
             status: 0
            success: True
                  x: array([1.49225651, 0.23901602])

        Inspect the instance attribute ``args`` to see that it has
        been set to its maximum-likelihood estimate.

        .. sourcecode:: python

           >>> blup.args
           [12.706204736174696, 0.5411814081739366]

        See Notes for an explanation of the discrepancy between the
        values of ``x`` and ``blup.args``.

        """
        return _opt(self, opt_method, n_start, method, bounds, tol,
                    callback, options)

    def xtest(self, t):
        r"""Return the best linear predictor of a random variable.

        Returns the realization, :math:`z^{*}`, of the best linear
        predictor of a random variable based on the realization of a
        finite random process, :math:`x`.

        Parameters
        ----------

        t : (m, d) array_like

            Test input.

        Returns
        -------

        xtest : (m,) ndarray

            Test output.

        mse : (m,) ndarray, optional

            Mean-squared error in the test output.

        Raises
        ------

        Warns
        -----

        Warnings
        --------
        
        See also
        --------

        :func:`Blup.xtest`

           Best linear predictor

        Notes
        -----        

        Examples
        --------

        One-dimensional index set, one test input:

        .. sourcecode:: python

           >>> ttrain = [0., 1.]
           >>> xtrain = [0., 1.]
           >>> blup = mim.Blup(ttrain, xtrain, args=[1., 1.])
           >>> blup.xtest(0.62)
           (array(0.63368638), array(0.03371449))

        Two test inputs:

        .. sourcecode:: python

           >>> blup.xtest([0.62, 0.89])
           (array([0.63368638, 0.90790372]), array(0.03371449, 0.00538887]))

        Two-dimensional index set, one test input:

        .. sourcecode:: python

           >>> ttrain = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
           >>> xtrain = [0., 1., 1., 2.]
           >>> blup = Blup(ttrain, xtrain, args=[1., 1., 1.])
           >>> blup.xtest([[0.43, 0.27]])
           (array([0.63956746]), array([0.06813622]))

        Two test inputs at once:

        .. sourcecode:: python

           >>> blup.xtest([[0.43, 0.27], [0.16, 0.93]])
           (array([0.63956746, 1.09544711]), array([0.06813622, 0.01396161])

        """
        if np.isscalar(t):
            # Allow for a single scalar index
            t = np.atleast_1d(t)
            scalar_input = True
        else:
            # Normally, there will be multiple scalar or vector indices
            scalar_input = False
        # BLUP
        k = self.kernel(self.ttrain, t)
        # k = np.atleast_2d(self.kernel(self.ttrain, t))
        phi = self._phi(t)
        xtest = phi@self.Beta + k.T@self.Gamma
        # MSE of BLUP
        delta = linalg.cho_solve(self.U_K, k)
        epsilon = phi.T - self.Phi.T@delta
        zeta = linalg.cho_solve(self.U_G, epsilon)
        mse = (np.diag(self.kernel(t, t))
               - _dot(k, delta)
               + _dot(epsilon, zeta))
        # mse = ([self.kernel(t_i, t_i).item() for t_i in t]
        #        - _dot(k, delta)
        #        + _dot(epsilon, zeta))
        # A very small MSE may be computed as slightly negative due
        # to a lack of numercial precision
        mse[mse<0.] = 0.
        if scalar_input:
            return np.squeeze(xtest), np.squeeze(mse)
        return xtest, mse
