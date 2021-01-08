'''
This visualizes a multivariate gaussian distribution.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

mu = np.array([0, 0])
sigma = np.array([[1, 0], [0, 1]])
rv = multivariate_normal(mu, sigma)

x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(X, Y, rv.pdf(pos), linewidth=0.5)
plt.show()
