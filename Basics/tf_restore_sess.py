# Restoring variables
# The tf.train.Saver object not only saves variables to checkpoint files, it also restores variables.
# Note that when you restore variables from a file you do not have to initialize them beforehand.
# For example, the following snippet demonstrates how to call the tf.train.Saver.restore method to restore variables
# from a checkpoint file:

import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())