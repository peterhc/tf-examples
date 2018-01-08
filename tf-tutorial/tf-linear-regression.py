# Tensorflow Core Tutorial
# Refer to: https://www.tensorflow.org/get_started/get_started

import tensorflow as tf
print ("Tensorflow version:",tf.__version__) # Tensorflow version info
LOGDIR='/tmp/my_linear_graph'


# Linear Regression Model
# In machine learning we will typically want a model that can take arbitrary inputs, such as the one above.
# To make the model trainable, we need to be able to modify the graph to get new outputs with the same input.
# Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

sess = tf.Session()

# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
init = tf.global_variables_initializer()
sess.run(init)

# It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables.
# Until we call sess.run, the variables are uninitialized.
# Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# to produce the output:
# [ 0.          0.30000001  0.60000002  0.90000004]


# We've created a model, but we don't know how good it is yet. To evaluate the model on training data,
# we need a y placeholder to provide the desired values, and we need to write a loss function.
# A loss function measures how far apart the current model is from the provided data. We'll use a standard loss model for
# linear regression, which sums the squares of the deltas between the current model and the provided data.
# linear_model - y creates a vector where each element is the corresponding example's error delta.
# We call tf.square to square that error. Then, we sum all the squared errors to create a single scalar that abstracts
# the error of all examples using tf.reduce_sum:
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# producing the loss value
# 23.66

# We could improve this manually by reassigning the values of W and b to the perfect values of -1 and 1.
# A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign.
# For example, W=-1 and b=1 are the optimal parameters for our model. We can change W and b accordingly:
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# The final print shows the loss now is zero.
# 0.0

# Tensorboard - use localhost:6006 to view the graphs
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
writer.close()
sess.close()