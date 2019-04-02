#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from DataLoader import DataLoader
import sys

folder = sys.argv[1]
input_data, output_data = DataLoader(folder, 4).load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=6),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(input_data, output_data, epochs=100)
