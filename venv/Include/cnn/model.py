from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras import layers

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

ROWS = 128
COLS = 128
CHANNELS = 3

def catdog():
    model = Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))


    model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


