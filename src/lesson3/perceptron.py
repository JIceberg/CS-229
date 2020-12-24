import numpy as np
from random import randint
import matplotlib.pyplot as plt

x1_red = np.array([randint(0, 10) for _ in range(randint(5, 10))])
x2_red = np.array([randint(-10, 0) for _ in range(len(x1_red))])
plt.plot(x1_red, x2_red, 'ro')

x1_blue = np.array([randint(-10, 0) for _ in range(randint(5, 10))])
x2_blue = np.array([randint(0, 10) for _ in range(len(x1_blue))])
plt.plot(x1_blue, x2_blue, 'bo')

n = 2

reds = np.hstack((np.split(x1_red, len(x1_red)), np.split(x2_red, len(x2_red)))).reshape(len(x1_red), n)
blues = np.hstack((np.split(x1_blue, len(x1_blue)), np.split(x2_blue, len(x2_blue)))).reshape(len(x1_blue), n)

'''
Here's the important perceptron stuff.
'''

features = np.concatenate((reds, blues), axis=0).T

m, alpha = len(x1_red) + len(x1_blue), 0.1

y = np.concatenate((np.array([1] * len(x1_red)), np.array([0] * len(x1_blue))), axis=0)

w = [0 for _ in range(n)]

h = lambda x: np.heaviside(np.dot(w, x), 1)
def fit_model():
    fit, i = False, 0
    while not(fit) and i < 500:
        fit = True
        for i in range(m):
            for j in range(n):
                if (y[i] - h(features.T[i])) != 0:
                    fit = False
                # this is the actual algorithm
                w[j] += alpha * (y[i] - h(features.T[i])) * features[j][i]
            i += 1

'''
Starting here is just a way to show you how
it fits the model. This isn't anything actually related to the algorithm.
'''

fit_model()

x = np.linspace(-10, 10, 100)
f = lambda k: w[0] * k if w[1] == 0 else -w[0] / w[1] * k
plt.plot(x, f(x))
plt.show()
