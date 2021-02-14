import tensorflow as tf
import numpy as np


# 2-D tensor `a`
# [[1, 2, 3],
#  [4, 5, 6]]
X = np.matrix([[1, 2, 3], [4, 5, 6]])
print('X=\n', X)
# 2-D tensor `b`
# [[ 7,  8],
#  [ 9, 10],
#  [11, 12]]
Y = np.matrix([[7, 8], [9, 10], [11, 12]])
print('Y=\n', Y)

# Matrix Multiplication in Tensorflow
# 2-D a*b
# 2-D tensor `a`
# [[1, 2, 3],
#  [4, 5, 6]]
# a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
a = tf.constant(X, dtype=tf.int32, shape=[2, 3])

# 2-D tensor `b`
# [[ 7,  8],
#  [ 9, 10],
#  [11, 12]]
# b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
b = tf.constant(Y, dtype=tf.int32, shape=[3, 2])

# `a` * `b`
# [[ 58,  64],
#  [139, 154]]
print('==> Matrix Multiplication using tf.matlul(). Result=\n', tf.matmul(a, b))

#==============================

# Matrix dot Product using np.dot()
print('==> Matrix dot Product using np.dot(). Result=\n', np.dot(X, Y))

#==============================

# Element-wise Multiplication using np.multiply()
X1 = np.array([[1, 2], [3, 4]])
print('X1=\n', X1)
W1 = np.array([[5, 6], [7, 8]])
print('W1=\n', W1)
print('==> Element-wise Product using np.multiply(). Result =\n', np.multiply(X1, W1))

# Error: ValueError: operands could not be broadcast together with shapes (2,3) (3,2)
# print('==> Element-wise Product using Numpy. Result =\n', np.multiply(X, Y))

#==============================

# 3D Matrix Multiplication using Tensorflow
# 3-D a*b
# 3-D tensor `a`
# [[[ 1,  2,  3],
#   [ 4,  5,  6]],
#  [[ 7,  8,  9],
#   [10, 11, 12]]]
a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])

# 3-D tensor `b`
# [[[13, 14],
#   [15, 16],
#   [17, 18]],
#  [[19, 20],
#   [21, 22],
#   [23, 24]]]
b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])

# `a` * `b`
# [[[ 94, 100],
#   [229, 244]],
#  [[508, 532],
#   [697, 730]]]
print('==> 3D Tensor Matrix using tf.matmul() Result=\n', tf.matmul(a, b))

#==============================

# Tensorflow Maxtix Multiplication with Random numbers
rand_array = np.random.rand(1024, 1024)
print('==> Tensorflow Matrix Multiplication\n. Result=\n', rand_array)

#==============================
