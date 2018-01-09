# Convolutional Neural Network
# URL: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
# Youtube video: https://www.youtube.com/watch?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&time_continue=2&v=NMd7WjZiCzc

# The Convolutional Neural Network gained popularity through its use with image data, and is currently the state of
# the art for detecting what an image is, or what is contained in the image. CNNs even play an integral role in tasks
# like automatically generating captions for images.
#
# The basic CNN structure is as follows: Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer -> Output

# Convolution is the act of taking the original data, and creating feature maps from it.Pooling is down-sampling,
# most often in the form of "max-pooling," where we select a region, and then take the maximum value in that region,
# and that becomes the new valgit mv tf-tuue for the entire region. Fully Connected Layers are typical neural networks, where all
# nodes are "fully connected." The convolutional layers are not fully connected like a traditional neural network.

# TFLearn
# pip install --upgrade tflearn

import tflearn  # no need to import Tensorflow
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

# Load MNist data
X, Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])  # Mnist data image is in 28x28 pixel
test_x = test_x.reshape([-1, 28, 28, 1])

# Input Layer
convnet = input_data(shape=[None, 28, 28, 1], name='input')

# ConvNet Layer
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# ConvNet Layer
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# Fully Connected Layer
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# Fully Connected Layer
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id='mnist')

model.save('/tmp/quicktest.model')

# Once you've saved, you can load the model with:
#model = tflearn.DNN(convnet)
#model.load('quicktest.model')
# You still need to setup the structure of the model. The .load method just simply restores the weights,
# so you also obviously need the model to have the same layers and neurons.

# After this load, you could do something like this to test predictions:
#import numpy as np
#print( np.round(model.predict([test_x[1]])[0]) )
#print(test_y[1])

# If you wanted to continue training? Just simply run .fit on your new data, save it, and continue.