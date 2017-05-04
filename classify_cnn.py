from keras.models import Sequential

fname = "64-soft-20-CNN.hdf5"

model = Sequential()
model.load_weights(fname)

