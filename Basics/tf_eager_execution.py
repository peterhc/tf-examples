# Tensorflow Eager Execution

import tensorflow as tf # version >= 1.50
import tensorflow.contrib.eager as tfe
import numpy as np

tfe.enable_eager_execution()

# Simple Execution
x = [[2.]]  # No need for placeholders!
m = tf.matmul(x, x)
print(m)  # tf.Tensor([[4.]], shape=(1, 1), dtype=float32)

# No more Lazy Loading
x = tf.random_uniform([2, 2])
for i in range(x.shape[0]):
  for j in range(x.shape[1]):
    print(x[i, j])

#  Tensors Act Like NumPy Arrays
x = tf.constant([1.0, 2.0, 3.0])
# Tensors are backed by NumPy arrays
assert type(x.numpy()) == np.ndarray
squared = np.square(x)  # Tensors are compatible with NumPy functions
# Tensors are iterable! for i in x:
print(i)


# Gradients - Automatic differentiation is built into eager execution
def square(x):
    return x ** 2

grad = tfe.gradients_function(square)
print(square(3.))   # tf.Tensor(9., shape=(), dtype=float32)
print(grad(3.))     # [tf.Tensor(6., shape=(), dtype=float32))]

# Gradients
x = tfe.Variable(2.0)  # Ue tfe.Variable when eager execution is enabled.

def loss(y):
    return (y - x ** 2) ** 2

grad = tfe.implicit_gradients(loss)   # Differentiate w.r.t. variables used to compute loss
print(loss(7.))  # tf.Tensor(9., shape=(), dtype=float32)
print(grad(7.))

# APIs
# for computing gradients work even when eager execution is not enabled
# ● tfe.gradients_function()
# ● tfe.value_and_gradients_function() ● tfe.implicit_gradients()
# ● tfe.implicit_value_and_gradients()



