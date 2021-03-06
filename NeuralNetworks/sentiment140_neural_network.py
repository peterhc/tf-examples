# 10K samples compared to 1.6 million samples with Deep Learning
# Using More Data - Deep Learning with Neural Networks and TensorFlow part 8
# URL: https://pythonprogramming.net/data-size-example-tensorflow-deep-learning-tutorial/
# Youtube video: https://www.youtube.com/watch?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&time_continue=241&v=JeamFbHhmDo

# NOTE: Based on your CPU/GUP power, this code may take a long time to run.

import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2
# 200 data? 52% (10 epochs)
# 2000 data? 62% (10 epochs) ~9-10 seconds on GPU; ~14 seconds on CPU
# 2000 data? 63% (15 epochs)
# 200,000 data? 74.3% (15 epochs)

batch_size = 32
total_batches = int(1600000/batch_size)
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum' :n_nodes_hl1,
                  'weight' :tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                  'bias' :tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum' :n_nodes_hl2,
                  'weight' :tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias' :tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum' :None,
                'weight' :tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias' :tf.Variable(tf.random_normal([n_classes])) ,}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data ,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1 ,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2 ,output_layer['weight']) + output_layer['bias']
    return output

# Add ops to save and restore Tensoflow model
saver = tf.train.Saver()  # No specific variable included so save all the variables
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction ,y) )
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log ,'r').read().split('\n')[-2] ) +1
            print('STARTING:' ,epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess ,"/tmp/model.ckpt")
            epoch_loss = 1
            with open('lexicon.pickle' ,'rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                      y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:' ,batches_run ,'/' ,total_batches ,'| Epoch:' ,epoch ,'| Batch Loss:' ,c ,)

            save_path = saver.save(sess, "/tmp/model.ckpt")
            print ("Model saved in file: %s" % save_path)
            print ('Epoch', epoch, 'completed out of' ,hm_epochs ,'loss:' ,epoch_loss)
            with open(tf_log ,'a') as f:
                f.write(str(epoch ) +'\n')
            epoch += 1

train_neural_network(x)

def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess ,"model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested' ,counter ,'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:' ,accuracy.eval({x :test_x, y :test_y}))


test_neural_network()

