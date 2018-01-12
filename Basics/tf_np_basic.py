import tensorflow as tf
import numpy as np

# Matrix Multiplication using Tensorflow
# 2-D a*b
# 2-D tensor `a`
# [[1, 2, 3],
#  [4, 5, 6]]
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

# 2-D tensor `b`
# [[ 7,  8],
#  [ 9, 10],
#  [11, 12]]
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])

# `a` * `b`
# [[ 58,  64],
#  [139, 154]]
c = tf.matmul(a, b)
# `a` * `b`
# [[ 58,  64],
#  [139, 154]]
with tf.Session() as sess:
    print('==> Matrix Multiplication using Tensorflow. Result=\n', sess.run(c))

#==============================

# Matrix Multiplication using Numpy
X1 = np.matrix([[1, 2, 3], [4, 5, 6]])
print('X1=', X1)
W1 = np.matrix([[7, 8], [9, 10], [11, 12]])
print('W1', W1)
print('==> Matrix Multiplication using Numpy. Result=\n', np.dot(X1, W1))

#==============================

print('==> Element-wise Product using Numpy')
X1 = np.array([[1, 2], [3, 4]])
print('X1=', X1)
W1 = np.array([[5, 6], [7, 8]])
print('W1', W1)
print('==> Element-wise Product using Numpy. Result =\n', np.multiply(X1, W1))

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
with tf.Session() as sess:
    print('==> 3D Tensor Matrix Multiplication Result=\n', sess.run(tf.matmul(a, b)))

#==============================

# Tensorflow Maxtix Multiplication with Random numbers
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
#  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print('==> Tensorflow Matrix Multiplication. Result=\n', sess.run(y, feed_dict={x: rand_array}))  # Will succeed.

#==============================

# X1:
# [[0, 0, 0],
#  [0, 0, 0],
#  [0, 2, 0]]
X1 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 2, 0]])
print('X1=\n', X1)
# W1:
# [[1, 0, -1],
#  [-1, -1, 0],
#  [1, 1, 0]]
W1 = np.matrix([[1, 0, -1], [-1, -1, 0], [1, 1, 0]])
print('W1=\n', W1)
print('Result=\n', np.dot(X1, W1))
