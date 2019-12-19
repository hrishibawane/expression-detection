
## Data loading n preprocessing

import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_path = 'dataset/images/train'
validation_path = 'dataset/images/validation'
emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']

train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
validation_datagen = ImageDataGenerator()

train_batches = train_datagen.flow_from_directory(train_path, class_mode='categorical', target_size=(48,48), batch_size=64)
validation_batches = validation_datagen.flow_from_directory(validation_path, class_mode='categorical', target_size=(48,48), batch_size=32)

from PIL import Image
from skimage import transform

def load_img(filename):
    np_img = Image.open(filename)
    np_img = np.array(np_img).astype('float32')
    np_img = transform.resize(np_img, (48,48,3))
    np_img = np.expand_dims(np_img, axis=0)
    return np_img

img = Image.open(train_path + '/angry/0.jpg')


## Building CNN
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48,48,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(rate=0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.3))
model.add(Dense(7, activation='softmax'))

model.compile(Adam(lr=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(train_batches,
                              validation_data=validation_batches,
                              validation_steps=220,
                              steps_per_epoch=450,
                              epochs=50)

model.save('model.h5')

## Testing on random image
img = load_img('test.jpg')
pred = model.predict(img)
result = emotions[np.argmax(pred)]






