
# Basic Operations example using TensorFlow library.

import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2, tf.int32)
b = tf.constant(3, tf.int32)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

print("Addition with variables: ", add)
print("Multiplication with variables: ", mul)

# Matrix Multiplication from TensorFlow official tutorial

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
print("Result= ", tf.multiply(matrix1, matrix2))
print("Result= ", tf.matmul(matrix1, matrix2))

# ==> [[ 12.]]
