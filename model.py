#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import numpy as np

# input:

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 1)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


