import cv2
import numpy as np
from numpy import ndarray
import os
from os.path import join
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

ncols=64
nrows=64


def normalize(im):
    dim = (ncols, nrows)
    resized = cv2.resize(im, dim)
    return resized


def create_sign_set():
    # hog = cv2.HOGDescriptor()
    warn_path = os.path.join(os.getcwd(), 'warnClassification')
    stop_path = os.path.join(os.getcwd(), 'stopClassification')
    false_path = os.path.join(os.getcwd(), 'falsePositives')

    warn_size, stop_size, false_size = 0,0,0

    warn_features = np.array([np.array(
        normalize(cv2.imread(join(warn_path, im)))) for im in os.listdir(warn_path)])

    stop_features = np.array([np.array(
        normalize(cv2.imread(join(stop_path, im)))) for im in os.listdir(stop_path)])

    false_features = np.array([np.array(
        normalize(cv2.imread(join(false_path, im)))) for im in os.listdir(false_path)])

    print(warn_features.shape, stop_features.shape, false_features.shape)

    # warn_labels = np.zeros(len(warn_features))
    # # print("warn label size: ", len(warn_labels))
    # stop_labels = np.ones(len(stop_features))
    # # print("stop label size: ", len(stop_labels))
    # labels = np.append(warn_labels, stop_labels)

    labels = np.zeros(((len(stop_features)+len(warn_features)+len(false_features)),3))
    for i, label in enumerate(labels):
        if i < len(stop_features):
            label[0] = 1
        elif i < len(warn_features)+len(stop_features):
            label[1] = 1
        else:
            label[2] = 1

    features = np.append(stop_features, warn_features, axis=0)
    features = np.append(features, false_features,axis=0)

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
    test, l_test = data[-(test_size-50):], labels[-(test_size-50):]

    print("Train len:", len(train))
    print("Test len:", len(test))

    return train, l_train, test, l_test, whole_data

train, l_train, test, l_test, whole_data = split_sign_dataset()
# train, test = train.astype('float32'), test.astype('float32')
#
epochs = 50
l_rate = 0.025
decay = l_rate/epochs
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
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dense(3))
model.add(Activation('softmax'))

# calculate decay of learning rate
epochs = 35
l_rate = 0.015
decay = l_rate/epochs
# sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', decay=decay,
              optimizer="adam", metrics=["accuracy"])

hist = model.fit(train, l_train, batch_size=32,
                 verbose=1, nb_epoch=epochs, validation_data=(test, l_test))

first_predict = model.predict_classes(whole_data[0][-50:])
score1 = model.evaluate(test, l_test, verbose=0)

hist = model.fit(train, l_train, batch_size=32, nb_epoch=epochs,
                  verbose=True, validation_split=0.2)

score = model.evaluate(test, l_test, verbose=0)
print("Test2 score: ", score[0])
print("Test2 accuracy: ", score[1]*100)
print("Test1 accuracy: ", score1[1]*100)
model.save_weights("64-soft-50-DNN.hdf5", True)

# model.load_weights("64-soft-50-CNN18.hdf5")

n_test, n_label = whole_data[0][-50:], whole_data[1][-50:]
print(model.predict_classes(n_test.astype('float32')))
print(first_predict)
for i, im in enumerate(n_test):
    # cv2.imshow("image", im)
    print("prediction: ", model.predict_classes(im.reshape((1,)+im.shape)))
    print("base: ", n_label[i])
    # cv2.waitKey(0)