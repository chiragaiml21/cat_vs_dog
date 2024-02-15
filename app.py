import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import load_img, img_to_array

async def predict(name):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.load_weights("D://Study Material//sem 6//Workshop//Project1//models//dogs_vs_cat.h5")

    test_image = load_img(f'prediction_data/{name}', target_size=(256, 256))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    test_image = test_image / 255.0

    result = model.predict(test_image)

    if result[0][0] < 0.5:
        prediction = 'cat'
    else:
        prediction = 'dog'

    print(prediction)

    return prediction

# predict("dog.4003.jpg")