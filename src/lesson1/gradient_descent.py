import numpy

'''
Gradient descent is an optimization algorithm for finding the local minimum of a
differentiable function. We continue running the algorithm until our parameters
converge to an approximate solution.
'''

# these are our training examples
# size is our input, so that is a "feature"
# # of bedrooms is an input, so that is also a feature
# price is our output, so that is the "target"
sizes = [2104, 1416, 1534, 852, 1234]
bedrooms = [3, 2, 3, 2, 4]
prices = [400, 232, 315, 178, 231]

# `m` is the number of training examples, meaning the number of rows
# `n` is the number of features
m, n = len(prices), 2

# initialize our parameters at 0
params = [0, 0, 0]

# our learning rate
alpha = 1E-8

# let's express our features as one list
features = [[1 for _ in range(m)], sizes, bedrooms]

# our hypothesis function
h = lambda x, i: sum([params[j] * x[j][i] for j in range(n+1)])

# distance between hypothesis and targets
# iterate
while d := numpy.sqrt(sum([numpy.square(h(features, i) - prices[i]) for i in range(m)])) > 50:
    for i in range(m):
        for j in range(n+1):
            # this is stochastic gradient descent
            params[j] -= alpha / m * (h(features, i) - prices[i]) * features[j][i]

print(params)
print([h(features, i) for i in range(m)])

# let's reset...
params = [0, 0, 0]

# distance between hypothesis and targets
# iterate
while d := numpy.sqrt(sum([numpy.square(h(features, i) - prices[i]) for i in range(m)])) > 50:
    for j in range(n+1):
        # this is batch gradient descent
        pd = 0
        for i in range(m):
            pd += (h(features, i) - prices[i]) * features[j][i]
        params[j] -= alpha / m * pd

print(params)
print([h(features, i) for i in range(m)])

'''
As you can see above, these are the applications of the same algorithm but in different
loops. One is a batch, and the other is stochastic. They produce very similar results.
However, stochastic is much faster when working with a large number of training examples.

DISCLAIMER:
Note that these algorithms are pythonic interpretations of the pseudo-code, and there are other
methods of optimizing this. These samples are just to represent the algorithm at play.
'''