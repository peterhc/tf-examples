# Deep Learning with TensorFlow - Creating the Neural Network Model
# Video: https://pythonprogramming.net/tensorflow-deep-neural-network-machine-learning-tutorial/
# Youtube video: https://www.youtube.com/watch?v=BhpvH5DuVu8&index=46&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# mnist = input_data.read_data_sets("~peterhc/MyProjects/DataSet/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10  # 0-9
batch_size = 100  # number of features

x = tf.placeholder('float', [None, 784]) # MNIST 28*28 pixel image
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # the multiplication of the raw input data multipled by their unique weights (starting as random,
    # but will be optimized): tf.matmul(l1,hidden_2_layer['weights']). We then are adding, with tf.add the bias.
    # We repeat this process for each of the hidden layers, all the way down to our output, where we have the final
    # values still being the multiplication of the input and the weights, plus the output layer's bias values.

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # Cost Function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # Apply Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # For each epoch, and for each batch in our data, we're going to run our optimizer and cost against our batch
        # of data. To keep track of our loss/cost at each step of the way, we are adding the total cost per epoch up.
        # For each epoch, we output the loss, which should be declining each time. This can be useful to track,
        # so you can see the diminishing returns over time. The first few epochs should have massive improvements,
        # but after about 10 or 20 you will be seeing very small, if any, changes, or you may actually get worse.

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)







