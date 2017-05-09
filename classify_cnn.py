import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense


def load_model():
    model = Sequential()

    model.add(Dense(256, input_dim=16384, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=['accuracy'])
    model.load_weights("64-soft-50-CNN18.hdf5")
    return model


def get_features(potentials):
    hog = cv2.HOGDescriptor()

    pots = np.array([np.array(hog.computeGradient(
        potential)).flatten() for potential in potentials])

    print(pots[0].shape)
    return pots

def classify_DNN(pot_signs):
    model = load_model()
    print(pot_signs[0].shape)
    features = get_features(pot_signs)
    print(features[0].shape)

    predictions = model.predict_classes(features, batch_size=len(pot_signs))
    return predictions
