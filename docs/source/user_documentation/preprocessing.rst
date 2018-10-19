.. _preprocessing:

Data preparation
================

Recal that it is important to validate the result of Gaussian-process
emulation (:ref:`validating`. If the estimator fails validation then the
regression or covariance functions have been misspecified. In this case we may
do one of two things: choose different regression or covariance functions, or
transform the data so that the functions are better suited to their
tasks. Here we restrict ourselves to the second approach under the assumption
the mean function, :math:`r`, is the zero function and the covariance
function, :math:`k`, is a squared-exponential covariance function.

Crucially, the squared exponential covariance function is a function of the
difference of its arguments only (i.e. it is translationally invariant) and
the properties of the random process are therefore independent of the absolute
values of its arguments. In particular, the variance is constant, i.e. for all
:math:`\boldsymbol{x}` it is the case that :math:`k(\boldsymbol{x},
\boldsymbol{x}) = \sigma_\mathrm{SE}^2`. The fact that the variance is
constant means that elements of the sample, :math:`\{(y(\mathbf{x}_i)) \,|\,
\mathbf{x}_i \in \mathbf{X} \text{ and } i = 1, 2, ... , n\}`, are drawn from
identically normal distributions. Considered together, we expect them to be
distributed normally with variance :math:`\sigma_\mathrm{SE}^2`. If we do not
observe this distribution in our training data we may transform it to ensure
that we do. Such a transformation is said to be *variance stabilizing*.

One such variance-stabilizing transformation is the *Box-Cox
transformation*. Let :math:`\{(y(\mathbf{x}_i)) \,|\, \mathbf{x}_i \in
\mathbf{X} \text{ and } i = 1, 2, ... , n\}` be a sample. Then the Box-Cox
transformation of the observations (Box and Cox, 1964) is the function
:math:`g` such that

.. math::

   g(y(\boldsymbol{x}_i); \lambda_1, \lambda_2) =
   \dfrac{(y(\boldsymbol{x}_i) + \lambda_2)^{\lambda_1} - 1}{\lambda_1}

for real :math:`\lambda_1`, :math:`\lambda_2` such that :math:`\lambda_{2} > -
y(\boldsymbol{x}_{i})` for all :math:`i`. Note that this expression is just a
scaled power law, with the scaling chosen such that :math:`\lim_{\lambda_1
\longrightarrow 0} g(y(\boldsymbol{x}_i); \lambda_1, \lambda_2) =
\ln(y(\boldsymbol{x}_i) + \lambda_2)`.

Box and Cox give the following theorem concerning the choice of the parameters
:math:`\lambda_1` and :math:`\lambda_2`. Assume that each observation
:math:`y(\boldsymbol{x}_{i})` is a realization of the random variable
:math:`Y(\boldsymbol{x}_{i})`, and that each transformed observation,
:math:`g(y(\boldsymbol{x}_{i}))` is a realization of the random variable
:math:`G(\boldsymbol{x}_{i})`. Furthermore assume that
:math:`G(\boldsymbol{x}_{1}), \ldots, G(\boldsymbol{x}_{n})` are independent
and identically normal, i.e. assume that for all :math:`n`,

.. math::
   
   G(\boldsymbol{x}_{i}) \sim N(\mu_\mathrm{t}^{\phantom{2}},
   \sigma_\mathrm{t}^2).

for some mean, :math:`\mu_\mathrm{t}^{\phantom{2}}`, and variance
:math:`\sigma_\mathrm{t}^2`. Then the joint distribution of the (untransformed)
observations has PDF

.. math::

   f_{(G(\boldsymbol{x}_{1}), \ldots,
   G(\boldsymbol{x}_{n}))}(y(\boldsymbol{x}_{1}), \ldots,
   y(\boldsymbol{x}_{N}); \lambda_1, \lambda_2) = \prod_{i = 1}^{N}
   \dfrac{1}{\sqrt{2 \pi \sigma_\mathrm{t}^2}} \exp \left( \dfrac{\sum_{i =
   1}^{N} (g(y(\boldsymbol{x}_{i})) - \mu_\mathrm{t})^2}{2
   \sigma_\mathrm{t}^2} \right) J(y(\boldsymbol{x}_{i}); \lambda_1,
   \lambda_2),

where the Jacobian

.. math::
   
   J(y(\boldsymbol{x}_{i}); \lambda_1, \lambda_2)
   &= \left| \dfrac{\operatorname{d} g(y(\boldsymbol{x}_i))}{\operatorname{d}
   y(\boldsymbol{x}_i)} \right|\\
   &= (y(\boldsymbol{x}_i) + \lambda_2)^{(\lambda_1 - 1)}.

Thus the log-likelihood of the parameters :math:`\lambda_1` and
:math:`\lambda_2` is given by

.. math::

   \begin{split}
       \ln L_{(\lambda_1, \lambda_2)}(\lambda_1, \lambda_2;
       y(\boldsymbol{x}_{1}), \ldots, y(\boldsymbol{x}_i))
       &= - \frac{n}{2} \ln
       \sigma_\mathrm{t}^2 - \frac{n}{2}\log 2 \pi -
       \frac{1}{2\sigma_\mathrm{t}^2}\sum_{i = 1}^{N}(g(y(\boldsymbol{x}_i)) -
       \mu_\mathrm{t})^2\\
       &\qquad + (\lambda_1 - 1) \sum_{i = 1}^{n}
       \ln(y(\boldsymbol{x}_i) + \lambda_2)
   \end{split}

where we may substitute for :math:`\mu_\mathrm{t}^{\phantom{2}}` and
:math:`\sigma_\mathrm{t}^2` their maximum-likelihood estimates,

.. math::
      
   \hat{\mu}_{\mathrm{t}} = \frac{1}{N} \sum_{i = 1}^N g(y(\boldsymbol{x}_i))

and

.. math::
   
   \hat{\sigma}_\mathrm{t}^2 = \dfrac{1}{N} \sum_{i = 1}^N
   (g(y(\boldsymbol{x}_i)) - \hat{\mu}_{\mathrm{t}})^2.

The maximum-likelihood estimate of the transformation parameters has an
associated distribution, but it is common to consider transformation parameter
as known, and not propagate its uncertainty through the subsequent
analysis. It is also common to round off the value of :math:`\lambda_1` to the
nearest half-integer, e.g. to use one of the values :math:`2, 1, 1 / 2, 0, -1
/ 2, -1,` or :math:`-2` (the square, identity, square root, logarithm,
reciprocal square root, reciprocal, or reciprocal square) which give the
transformation a ready interpretation.

We must be careful about the direction of the implication in this theorem. It
is not the case that the maximum-likelihood estimates of :math:`\lambda_1` and
:math:`\lambda_2` ensure that the transformed variables are normal. We must
compute these maximum-likelihood estimates and then *check* that the
assumption of normality is met by inspecting a histogram of the transformed
values. In general the condition is not met, i.e. there exists no power
transformation such that :math:`G(\boldsymbol{x}_{i}) \sim N(\mu_\mathrm{t},
\sigma_\mathrm{t}^2)` for all :math:`i = 1, 2, \dots , n`. However we may
still use the power transformation to *regularize* the sample, i.e. to bring
the distribution of its elements closer to normal (Draper and Cox, 1969}.

When using the zero regression function the residuals :math:`(\boldsymbol{y} -
\boldsymbol{r})` in expression :eq:`gpe_mean` may be poorly behaved. They may
all have the same sign, or may contain outliers. The use of a transformation
overcomes these problems, and avoids the need to specify a non-zero regression
function.


References
----------
Box, G.E.P., and D.R. Cox. 1964. 'An analysis of transformations' in *Journal
of the Royal Statistical Society* Series B (Methodological) (26) 2, 211--252.

Draper, N.R., and D.R. Cox. 1969. 'On distributions and their transformation
to normality' in *Journal of the Royal Statistical Society* Series B
(Methodological) (31) 3, 472--476.
