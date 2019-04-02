#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from DataLoader import DataLoader
import sys

folder = sys.argv[1]
volume = int(sys.argv[2])
volume_step = int(sys.argv[3])
time_horizon = int(sys.argv[4])
time_step = int(sys.argv[5])
max_action = int(sys.argv[6])
min_action = int(sys.argv[7])
action_step = int(sys.argv[8])

input_data, output_data = DataLoader(folder, 4).load_data()

# normalisation of output
normalized_output = (output_data - min_action) / action_step

model = keras.Sequential([
    keras.layers.Dense(6, activation=tf.nn.sigmoid),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense((max_action - min_action) / action_step + 1, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(input_data, normalized_output, epochs=100)
