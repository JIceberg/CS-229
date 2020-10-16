import numpy as np

'''
With logistic regression, one of the things we might want
is for the hypothesis h_\theta(x) \in [0, 1].

We'll say that our hypothesis function is equal to
some function of g w.r.t \theta^{T}x which can be defined as:
\frac{1}{1 + e^{-\theta^{T}x}}. This is actually just a form
of the function g(z), where g(z) = \frac{1}{1 + e^{-z}}.

The graph of this function is a "sigmoid" or a "logistic" curve.
This forces the output values to only exist between 0 and 1.

We can then define the likelihood of the parameters L(\theta)
as:
\prod_{i=1}^{m} h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}

We then define the log likelihood to be l(\theta) = \log{L(\theta)}
'''