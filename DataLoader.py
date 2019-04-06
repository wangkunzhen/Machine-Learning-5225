from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, path, time_step_per_episode):
        self.path = path
        self.time_step_per_episode = time_step_per_episode

    def load_market_data(self):
        """
        Load all market data input into a flat 2-D numpy array
        :return: 2D numpy array with each row representing a decision point's market data
        """
        market_files = np.asarray(
            [join(self.path, f) for f in listdir(self.path) if isfile(join(self.path, f)) and f.startswith("market")])
        market_files.sort()
        market_rows = np.asarray([np.asarray(pd.read_csv(f)) for f in market_files])
        shape = market_rows.shape
        flatten_rows = market_rows.reshape(shape[0]*shape[1], shape[2])
        return flatten_rows
        # return self.reshape_market_data(flatten_rows)

    def reshape_market_data(self, market_rows):
        market_variable_len = int(market_rows.shape[1] / self.time_step_per_episode)
        return np.asarray(market_rows).reshape(self.time_step_per_episode*market_rows.shape[0], market_variable_len)

    def load_private_data(self):
        private_files = np.asarray(
            [join(self.path, f) for f in listdir(self.path) if isfile(join(self.path, f)) and f.startswith("private")])
        private_files.sort()
        rows = np.asarray([np.asarray(pd.read_csv(f)) for f in private_files])
        shape = rows.shape
        flatten_rows = rows.reshape(shape[0] * shape[1], shape[2])
        return self.reshape_private_data(flatten_rows)

    def reshape_private_data(self, private_rows):
        private_variable_len = int((private_rows.shape[1] - 1) / self.time_step_per_episode)
        return np.asarray(private_rows[:, :-1]).reshape(self.time_step_per_episode * private_rows.shape[0], private_variable_len)

    def combine_market_and_private_data(self, market, private):
        repeat_count = int(market.shape[0] / self.time_step_per_episode)
        repeated_remaining_count = np.repeat(np.asarray(range(self.time_step_per_episode, 0, -1)), repeat_count)
        remaining_times = repeated_remaining_count.reshape(repeat_count, self.time_step_per_episode, order='F')\
                                                  .reshape(market.shape[0], 1)
        input_data = np.hstack((market, private[:, 1:], remaining_times))
        output_data = private[:, 0]
        return input_data, output_data

    def window_stack(self, arr, step_size, width):
        return np.hstack(arr[i:1 + i - width or None:step_size] for i in range(0, width))

    def load_data(self, window_size):
        market_rows = self.load_market_data()
        market_rows = self.window_stack(market_rows, 1, window_size)
        private_rows = self.load_private_data()[window_size-1:]
        assert market_rows.shape[0] == private_rows.shape[0]
        input, output = self.combine_market_and_private_data(market_rows, private_rows)
        return input, output


