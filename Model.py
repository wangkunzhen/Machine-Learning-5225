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

output_count = (max_action - min_action) / action_step + 1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(output_count, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(input_data, normalized_output, epochs=100)
