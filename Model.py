#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from DataLoader import DataLoader
from ModelEvaluator import ModelEvaluator
import numpy as np
from os.path import join
import sys

train_folder = sys.argv[1]
test_folder = sys.argv[9]
volume = int(sys.argv[2])
volume_step = int(sys.argv[3])
time_horizon = int(sys.argv[4])
time_step = int(sys.argv[5])
max_action = int(sys.argv[6])
min_action = int(sys.argv[7])
action_step = int(sys.argv[8])

window_size = 1
input_data, output_data = DataLoader(train_folder, 4).load_data(window_size)
test_input, test_output = DataLoader(test_folder, 4).load_data(window_size)

# normalisation of output
normalized_output = (output_data - min_action) / action_step
normalized_output_test = (test_output - min_action) / action_step

output_count = (max_action - min_action) / action_step + 1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2+4*window_size,)),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(output_count, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(input_data, normalized_output, epochs=10)

res = model.evaluate(test_input, normalized_output_test)
print(res)

possible_actions = range(max_action, min_action, -action_step)
evaluation_result = ModelEvaluator.evaluate(model, test_folder, volume, volume_step, time_horizon, time_step, int(9.5*60*60), 16*60*60, possible_actions, action_step, 30)
print(evaluation_result)