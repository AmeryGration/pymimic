.. _ego:

Introduction to efficient global optimization
=============================================

Suppose that we are performing Gaussian-process emulation using training data
:math:`\{(\mathbf{x}_i, y(\mathbf{x}_i)) \,|\, \mathbf{x}_i \in \mathbf{X}
\text{ and } i = 1, 2, ... , n\}`. The minimum of the sample is
:math:`y_\mathrm{min} = \min(\{(y(\mathbf{x}_i)) \,|\, \mathbf{x}_i \in
\mathbf{X} \text{ and } i = 1, 2, ... , n\})`. We would like to know where to
sample in order to improve the accuracy of this minimum. To this end we define
the *improvement in the minimum*,

.. math::
   
   I_{Y(\boldsymbol{x})}(Y(\boldsymbol{x})) := \max(y_\mathrm{min} -
   Y(\boldsymbol{x}), 0).

This is a random variable, the PDF of which is

.. math::
   
   f_{I_{Y(\boldsymbol{x})}}(y(\boldsymbol{x}))
   &= \max(y_\mathrm{min} - y(\boldsymbol{x}), 0)\\
   &= \begin{cases}
       y_\mathrm{min} - y &\text{if $y(\boldsymbol{x}) < y_\mathrm{min}$,}\\
       0 &\text{otherwise}.
   \end{cases}

By definition the expected improvement is

.. math::
   
   \operatorname{E}(I_{Y(\boldsymbol{x})}(Y(\boldsymbol{x}))) :=
   \int_{\mathbf{R}} I_{Y(\boldsymbol{x})}(y(\boldsymbol{x}))
   f_{Y(\boldsymbol{x})}(y(\boldsymbol{x})) \mathrm{d} y(\boldsymbol{x}),

where :math:`f_{Y(\boldsymbol{x})}` is the PDF of
:math:`Y(\boldsymbol{x})`. In the case of GPE we know that
:math:`Y(\boldsymbol{x}) \sim N(\hat{r}(\boldsymbol{x}),
\hat{\sigma}(\boldsymbol{x})^2)`, i.e. :math:`f_{Y(\boldsymbol{x})}` is the
normal (i.e. Gaussian) PDF :math:`\phi(y; \hat{r}(\boldsymbol{x}),
\hat{\sigma}(\boldsymbol{x})^2)`. If :math:`\hat{\sigma}(\boldsymbol{x})^2 =
0` then the value :math:`y(\boldsymbol{x})` is known with certainty and we
cannot expect any improvement, hence
:math:`\operatorname{E}(I_{Y(\boldsymbol{x})}(Y(\boldsymbol{x}))) = 0`. If
:math:`\hat{\sigma}(\boldsymbol{x})^2 > 0` then we may make the change of
variables from :math:`y(\boldsymbol{x})` to :math:`u'(\boldsymbol{x}) =
(y(\boldsymbol{x}) - \hat{r}(\boldsymbol{x})) / \hat{\sigma}(\boldsymbol{x})`
to find that the expected improvement is

.. math::
   
   \operatorname{E}(I_{Y(\boldsymbol{x})}(Y(\boldsymbol{x}))) =
   \hat{\sigma}(\boldsymbol{x})
   \int_{-\infty}^{u(\boldsymbol{x})}(u(\boldsymbol{x}) - u'(\boldsymbol{x}))
   \phi(u'(\boldsymbol{x}); 0, 1) \mathrm{d} u'(\boldsymbol{x})

where :math:`u(\boldsymbol{x}) := (y_\mathrm{min} - \hat{r}(\boldsymbol{x})) /
\hat{\sigma}(\boldsymbol{x})`. Thus,

.. math::
   
   \operatorname{E}(I_{Y(\boldsymbol{x})}(Y(\boldsymbol{x}))) =
   \begin{cases}
       \hat{\sigma}(\boldsymbol{x}) (u(\boldsymbol{x}) \Phi(u(\boldsymbol{x});
       0, 1) + \phi(u(\boldsymbol{x}); 0, 1))
       &\text{ if $0 < \hat{\sigma}(\boldsymbol{x})$,}\\
       0 &\text{ otherwise}
   \end{cases}

where :math:`\Phi(u(\boldsymbol{x}); 0, 1)` is the normal cumulative
distribution function. We augment our training data, with the pair
:math:`(\boldsymbol{x}_{n + 1}, y(\boldsymbol{x}_{n + 1}))` where

.. math::
   
   \boldsymbol{x}_{n + 1} = \underset{\boldsymbol{x} \in
   \boldsymbol{X}}{\operatorname{argmax}}
   \operatorname{E}(I_{Y(\boldsymbol{x})}(Y(\boldsymbol{x}))),

and then iterate this procedure until
:math:`\operatorname{E}(I_{Y(\boldsymbol{x})}(Y(\boldsymbol{x})))` is smaller
than some threshold, :math:`\epsilon`.

The expected improvement for :math:`0 < \hat{\sigma}^2(\boldsymbol{x})` is the
sum of two terms in :math:`u(\boldsymbol{x})`. The first term dominates if
:math:`u(\boldsymbol{x})` is large, while the second term dominates if
:math:`u(\boldsymbol{x})` is small. For given :math:`\hat{r}(\boldsymbol{x})`
it is the case that :math:`u(\boldsymbol{x})` is large if
:math:`\hat{\sigma}(\boldsymbol{x})` is small (which will be the case close to
design points, including the current minimum) and :math:`u(\boldsymbol{x})` is
small if :math:`\hat{\sigma}(\boldsymbol{x})` is large (which will be the case
away from design points, including the current minimum). The expected
improvement is therefore a tradeoff between probable small improvements (near
to the current minimum) and improbable large improvements (remote from the
current minimum), or between local and global search.


References
----------

Jones, D. R., Schonlau, M., and W. J. Welch. 1998. 'Efficient global
optimization of expensive black-box functions' in *Journal of global
optimization*, 13 (4): 455--92.

