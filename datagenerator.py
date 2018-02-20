from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

warn_path = os.path.join(os.getcwd(), 'falsePositives')

for image in os.listdir(warn_path):
    img = load_img(os.path.join(warn_path, image))
    print("Working on image: ", image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i=0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='falsePositives', save_format='jpeg'):
        i+=1
        if i > 5:
            break
