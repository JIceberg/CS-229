import numpy as np

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