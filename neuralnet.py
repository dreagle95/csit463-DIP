import cv2
import numpy as np
from numpy import ndarray
import os
from random import shuffle
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# # load tensorflow dataset
# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
def normalize(im):
    dim = (128, 128)
    resized = cv2.resize(im, dim)
    return resized


def create_sign_featureset():
    warn_featureset = []
    stop_featureset = []
    hog = cv2.HOGDescriptor()
    warn_path = os.path.join(os.getcwd(), 'warnClassification')
    stop_path = os.path.join(os.getcwd(), 'stopClassification')

    for image in os.listdir(warn_path):
        image = cv2.imread(os.path.join(warn_path, image))
        image = normalize(image)
        features = hog.computeGradient(image)
        warn_featureset.append([ndarray.flatten(np.array(features, dtype=np.float32)), [1,0]])

    for image in os.listdir(stop_path):
        image = cv2.imread(os.path.join(stop_path, image))
        image = normalize(image)
        features = hog.computeGradient(image)
        stop_featureset.append([ndarray.flatten(np.array(features, dtype=np.float32)), [0, 1]])

    return ndarray.flatten(np.array(warn_featureset)), \
           ndarray.flatten(np.array(stop_featureset))


def split_sign_dataset():
    # featureset = create_sign_featureset()
    warn_features, stop_features = create_sign_featureset()
    features = []
    features += warn_features
    features += stop_features


    print("warn feats: ", len(warn_features))
    shuffle(warn_features)
    shuffle(stop_features)

    # reserve 10% of the dataset for testing
    test_size = int(0.10*max(len(warn_features), len(stop_features)))
    # print(test_size)

    w_train = ndarray.flatten(warn_features[:-test_size])
    w_test = ndarray.flatten(warn_features[-test_size:])
    s_train = ndarray.flatten(stop_features[:-test_size])
    s_test = ndarray.flatten(stop_features[-test_size:])

    return  w_train, w_test, s_train, s_test

w_train, w_test, s_train, s_test = split_sign_dataset()
print(len(w_train))

# define constants
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 25
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

# initialize random weights and biases to start
hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(w_train[0]), n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):
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
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        # OLD:
        sess.run(tf.initialize_all_variables())
        # NEW:
        # sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            i=0
            epoch_loss = 0

            while i < len(w_train):
                start = i
                end = i+batch_size
                w_batch = np.array(w_train[start:end], dtype=object)
                s_batch = np.array(s_train[start:end], dtype=object)

                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                print(type(w_batch))

                _, c = sess.run([optimizer, cost], feed_dict={x: w_batch, y: s_batch})
                epoch_loss += c
                i+=batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:w_test, y:s_test}))


train_neural_network(x)