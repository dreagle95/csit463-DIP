import cv2
import numpy as np
from numpy import ndarray
import os
from os.path import join
# from random import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from sklearn.utils import shuffle

ncols=64
nrows=64

def normalize(im):
    dim = (ncols,nrows)
    resized = cv2.resize(im, dim)
    return resized

"""
    This was me being stupid
"""
def create_sign_featureset():
    warn_featureset = []
    stop_featureset = []
    hog = cv2.HOGDescriptor()
    warn_path = os.path.join(os.getcwd(), 'warnClassification')
    stop_path = os.path.join(os.getcwd(), 'stopClassification')

    for image in os.listdir(warn_path):
        image = cv2.imread(os.path.join(warn_path, image))
        image = normalize(image)
        features = [np.array(hog.computeGradient(image), dtype=np.float32).flatten()]
        warn_featureset.append(features)

    for image in os.listdir(stop_path):
        image = cv2.imread(os.path.join(stop_path, image))
        image = normalize(image)
        features = [np.array(hog.computeGradient(image), dtype=np.float32).flatten()]
        stop_featureset.append(features)

    warn_labels=np.zeros(len(warn_featureset))
    stop_labels=np.ones(len(stop_featureset))
    features = np.append(warn_featureset, stop_featureset)
    labels = np.append(warn_labels, stop_labels)

    return features, labels

def create_sign_set():
    warn_path = os.path.join(os.getcwd(), 'warnClassification')
    stop_path = os.path.join(os.getcwd(), 'stopClassification')

    warn_size, stop_size = 0,0

    warn_features = np.array([np.array(
        normalize(cv2.imread(join(warn_path, im)))) for im in os.listdir(warn_path)])

    stop_features = np.array([np.array(
        normalize(cv2.imread(join(stop_path, im)))) for im in os.listdir(stop_path)])

    print(warn_features.shape, stop_features.shape)

    warn_labels = np.zeros(len(warn_features))
    # print("warn label size: ", len(warn_labels))
    stop_labels = np.ones(len(stop_features))
    # print("stop label size: ", len(stop_labels))
    labels = np.append(warn_labels, stop_labels)

    features = np.append(warn_features, stop_features,axis=0)
    print("features shape: ", features.shape)

    return features, labels


def split_sign_dataset():
    features, labels = create_sign_set()
    print("features len: ", len(features), "Labels len: ", len(labels))

    data, labels = shuffle(features, labels, random_state=2)
    whole_data = [data, labels]
    # print("whole data: ",len(whole_data))

    test_size = int(0.10*len(data))
    print("test_size:",test_size)

    train, l_train = data[:-test_size], labels[:-test_size]
    test, l_test = data[-(test_size-10):], labels[-(test_size-10):]

    print("Train len:", len(train))
    print("Test len:", len(test))

    return train, l_train, test, l_test, whole_data


train, l_train, test, l_test, whole_data = split_sign_dataset()
train, test = train.astype('float32'), test.astype('float32')

print("train len outside:",len(train))

model = Sequential()

model.add(Convolution2D(32, 3, 3,
                        border_mode='valid',
                        input_shape=(ncols, nrows, 3)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(32,3,3))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

# calculate decay of learning rate
epochs = 20
l_rate = 0.015
decay = l_rate/epochs
# sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='sparse_categorical_crossentropy', decay=decay,
              optimizer="adadelta", metrics=["accuracy"])
#
# hist = model.fit(train, l_train, batch_size=32,
#                  verbose=1, nb_epoch=epochs, validation_data=(test, l_test))
#
# hist = model.fit(train, l_train, batch_size=32, nb_epoch=20,
#                   verbose=True, validation_split=0.2)
#
# score = model.evaluate(test, l_test, verbose=0)
# print("Train score: ", score[0])
# print("Train accuracy: ", score[1]*100)
#
# model.save_weights("64-soft-20-CNN2.hdf5", True)

fname = "64-soft-20-CNN2.hdf5"
model.load_weights(fname)

# n_test, n_label = whole_data[0][-10:], whole_data[1][-10:]
#
# warn_path = os.path.join(os.getcwd(), 'warnClassification2')
# stop_path = os.path.join(os.getcwd(), 'stopClassification2')
#
# warn_features = np.array([np.array(
#         normalize(cv2.imread(join(warn_path, im)))) for im in os.listdir(warn_path)])
# stop_features = np.array([np.array(
#         normalize(cv2.imread(join(stop_path, im)))) for im in os.listdir(stop_path)])
# warn_labels = np.zeros(len(warn_features))
# # print("warn label size: ", len(warn_labels))
# stop_labels = np.ones(len(stop_features))
# # print("stop label size: ", len(stop_labels))
# labels = np.append(warn_labels, stop_labels)
#
# features = np.append(warn_features, stop_features,axis=0)
# print("features shape: ", features.shape)
#
# data, labels = shuffle(features, labels, random_state=2)
stop1 = cv2.imread(os.path.join(os.getcwd(), "stop1.jpg"))
stop2 = cv2.imread(os.path.join(os.getcwd(), "stop2.jpg"))

features = np.array([np.array(normalize(im)) for im in [stop1, stop2]])
labels = [1.0,1.0]
predictions = model.predict_classes(features)

for i, prediction in enumerate(predictions):
    print("correct: ", "stop" if labels[i] == 1.0 else "warn")
cv2.imshow("stop1", features[0])
cv2.imshow("stop2", features[1])
cv2.waitKey(0)

# print(model.predict_classes(test))
# print(l_test[:])

# predictions = model.predict_classes(data)
# for i, prediction in enumerate(predictions):
#     if prediction == int(labels[i]):
#         print("correct: ", "stop" if labels[i] == 1.0 else "warn")
#         cv2.imshow("correct", data[i].astype('uint8'))
#         cv2.waitKey(0)
#     else:
#         print("FAIL")
#         print("should be: ", "stop" if labels[i] == 1.0 else "warn")
#         cv2.imshow("fail", data[i].astype('uint8'))
#         cv2.waitKey(0)

# for i, im in enumerate(n_test):
#     cv2.imshow("image", im)
#     print("prediction: ", model.predict_classes(im.reshape((1,)+im.shape)))
#     print("base: ", n_label[i])
#     cv2.waitKey(0)