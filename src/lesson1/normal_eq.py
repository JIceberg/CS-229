import numpy as np

'''
The normal equation is another method of solving for linear regression,
but can be done in one step as opposed to the iterative method of gradient descent.
However, this only works with linear regression.

Let's define the notation first.

Say we have a function f(A), where A is a matrix in \mathbb{R}^{m * n}.
Let's define the function as a mapping of the entries of matrix A
into some scalar output. An example could be f(A) = A_{1,1} + A_{1,2}^{2} where
A is a 2x2 matrix.

Now we define the gradient \nabla_{A} {f(A)} as a matrix with the same dimensions
as A where each element is the partial derivative of the function `f` with respect
to that element A_{i,j} at the i-th row and j-th column. If we apply this to our example,
then the result would be:
```
[ 1  2A_{1,2} ]
[ 0      0    ]
```

So now let's apply this to our cost function J(\theta). We want to set
\nabla_{\theta} {J(\theta)} to 0 since we want to minimize the change in the cost.
Then, we solve for the value of theta. Since the minimum or maximum of a function is where
the derivative is 0, this will allow us to find the minimum for our linear regression.

Our cost function is defined as:
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m} {\left ( h_{\theta}\left (x^{(i)} \right )-y^{(i)} \right )}^{2}
where theta is the set of parameters.

Now, we can simplify this a bit by defining the hypothesis function `h` to be the vector product
of x and theta, where x is our feature matrix. h_{\theta}(x) = x\theta.
Note that h_{\theta}(x) is actually a vector with elements that represent h_{\theta}\left (x^{(i)} \right ).

Now we can simplify the cost function into:
J(\theta) = \frac{1}{2m}(x\theta - y)^{T}(x\theta - y)

Now we can take the gradient of this (we will actually refer to this notation as the derivative)
which results in
\nabla_{\theta} {J(\theta)} = \frac{1}{m}\left( x^{T}x\theta - x^{T}y \right)

Set that to 0, and then solve for \theta results in
\theta = (x^{T}x)^{-1}x^{T}y
'''

x1 = np.array([2104, 1416, 1534, 852, 1234])
x2 = np.array([3, 2, 3, 2, 4])
y = np.array([400, 232, 315, 178, 231])

# https://medium.com/@dikshitkathuria1803/normal-equation-using-python-5993454fbb41

x_bias = np.ones((len(y),1))
x1 = np.reshape(x1,(len(y),1))
x = np.append(x_bias,x1,axis=1)
x2 = np.reshape(x2,(len(y),1))
x = np.append(x,x2,axis=1)
x_transpose = np.transpose(x)
x_transpose_dot_x = x_transpose.dot(x)
temp_1 = np.linalg.inv(x_transpose_dot_x)
temp_2 = x_transpose.dot(y)
theta = temp_1.dot(temp_2)

print(theta)
res = x.dot(theta)
print(res)