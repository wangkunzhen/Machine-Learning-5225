import tensorflow as tf
from tensorflow import keras
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd


class Model:
    def __init__(self, private_folder, market_folder, volume, volume_step, time_horizon, time_step, max_action,
                 min_action, action_step, window_size):
        self.private_folder = private_folder
        self.market_folder = market_folder
        self.volume = volume
        self.volume_step = volume_step
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.max_action = max_action
        self.min_action = min_action
        self.action_step = action_step
        self.window_size = window_size

    def fit_model(self, epochs):
        training_market_folder = join(self.market_folder, "Train")
        training_private_folder = join(self.private_folder, "Train")
        input_data, output_data, private_variable_size, market_variable_size = self.load_data(training_market_folder,
                                                                                              training_private_folder)
        model = self.model(private_variable_size, market_variable_size)
        model.fit(input_data, output_data, epochs=epochs)

        print("Evaluating model accuracy")
        test_market_folder = join(self.market_folder, "Test")
        test_private_folder = join(self.private_folder, "Test")
        test_input, test_output, private_variable_size, market_variable_size = self.load_data(test_market_folder,
                                                                                              test_private_folder)
        loss, accuracy = model.evaluate(test_input, test_output)
        print("Loss " + str(loss) + " Accuracy " + str(accuracy))
        return model, loss, accuracy

    def model(self, private_variable_size, market_variable_size):
        output_count = (self.max_action - self.min_action) / self.action_step + 1
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(private_variable_size + market_variable_size * self.window_size,)),
            keras.layers.Dense(256, activation=tf.nn.leaky_relu),
            keras.layers.Dense(256, activation=tf.nn.sigmoid),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation=tf.nn.leaky_relu),
            keras.layers.Dense(256, activation=tf.nn.sigmoid),
            keras.layers.Dense(output_count, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def load_data(self, market_folder, private_folder):
        time_step_per_episode = int(self.time_horizon / self.time_step)
        market_rows = self.load_market_data_folder(market_folder)
        private_rows = self.load_private_data_folder(private_folder)
        assert len(market_rows) == len(private_rows)
        repeat_count = int(market_rows.shape[0] / time_step_per_episode)
        repeated_remaining_count = np.repeat(np.asarray(range(time_step_per_episode, 0, -1)), repeat_count)
        remaining_times = repeated_remaining_count.reshape(repeat_count, time_step_per_episode, order='F')\
            .reshape(market_rows.shape[0], 1)
        input_data = np.hstack((market_rows, private_rows[:, 1:], remaining_times))
        output_data = private_rows[:, 0]
        normalized_output = (output_data - self.min_action) / self.action_step
        return input_data, normalized_output, private_rows.shape[1], market_rows.shape[1]

    def load_market_data_folder(self, folder):
        market_files = np.asarray(
            [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.startswith("market")])
        market_files.sort()
        print("Loading market data folder. Found " + str(len(market_files)) + " market files")
        market_rows = np.concatenate([self.load_market_data(f) for f in market_files])
        print("Loaded market data folder. Total " + str(len(market_rows)) + " rows")
        return market_rows

    def load_market_data(self, file):
        """
        Give a market data file, load all market data input into a flat 2-D numpy array
        :return: 2D numpy array with each row representing a decision point's market data
        """
        rows = np.asarray(pd.read_csv(file, header=None))
        return rows

    def load_private_data_folder(self, folder):
        private_files = np.asarray(
            [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.startswith("private")])
        private_files.sort()
        print("Loading private data folder. Found " + str(len(private_files)) + " private files")
        private_rows = np.concatenate([self.load_private_data(f) for f in private_files])
        private_rows = Model.window_stack(private_rows, 1, self.window_size)
        print("Loaded private data folder. Total " + str(len(private_rows)) + " rows")
        return private_rows

    def load_private_data(self, file):
        rows_without_cost = np.asarray(pd.read_csv(file, header=None))[1:, :-1]
        time_step_per_episode = int(self.time_horizon / self.time_step)
        private_len = int(rows_without_cost.shape[1] / time_step_per_episode)
        return rows_without_cost.reshape(time_step_per_episode * rows_without_cost.shape[0], private_len)

    @staticmethod
    def window_stack(arr, step_size, width):
        return np.hstack(arr[i:1 + i - width or None:step_size] for i in range(0, width))
