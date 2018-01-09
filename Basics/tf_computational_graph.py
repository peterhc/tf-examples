# Tensorflow Core Tutorial
# Refer to: https://www.tensorflow.org/get_started/get_started

import tensorflow as tf
print ("Tensorflow version:",tf.__version__) # Tensorflow version info
LOGDIR='/tmp/my_graph'

# The Computational Graph
# A computational graph is a series of TensorFlow operations arranged into a graph of nodes. Let's build a simple
# computational graph. Each node takes zero or more tensors as inputs and produces a tensor as an output.
# One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it
# stores internally. We can create two floating point Tensors node1 and node2 as follows:

# Tensors
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

# Computational Graph
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


# A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Tensorboard - use localhost:6006 to view the graphs
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)
writer.close()
sess.close()
