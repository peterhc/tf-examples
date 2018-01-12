# Python
import tensorflow as tf
print ("Tensorflow version:",tf.__version__) # Tensorflow version info

hello = tf.constant('Hello, TensorFlow!')

with tf.Session() as sess:
    print(sess.run(hello))