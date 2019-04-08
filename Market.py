import pandas as pd
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, exists
from ModelUtil import ModelUtil


class Market:
    def __init__(self, data_folder, output_folder, time_horizon, time_step, moving_window, trade_start, trade_end):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.moving_window = moving_window
        self.trade_start = trade_start
        self.trade_end = trade_end
        if not exists(output_folder):
            mkdir(output_folder)

    def load(self):
        msg_book_files = [f for f in listdir(self.data_folder) if
                          isfile(join(self.data_folder, f)) and f.endswith("message_5.csv")]
        order_book_files = [f for f in listdir(self.data_folder) if
                            isfile(join(self.data_folder, f)) and f.endswith("orderbook_5.csv")]
        msg_book_files.sort()
        order_book_files.sort()

        assert len(msg_book_files) == len(order_book_files)

        for (msg_book_file, order_book_file) in zip(msg_book_files, order_book_files):
            msg_book = np.asarray(pd.read_csv(join(self.data_folder, msg_book_file), header=None))
            order_book = np.asarray(pd.read_csv(join(self.data_folder, order_book_file), header=None))

            assert len(msg_book) == len(order_book)
            print("Loaded " + msg_book_file + " + " + order_book_file)

            trade_start_time = self.trade_start
            trade_end_time = self.trade_end
            daily_result = []
            moving_average = np.zeros([order_book.shape[0], 1])
            for iline in range(order_book.shape[0]):
                if iline < self.moving_window:
                    continue
                else:
                    moving_average[iline] = np.mean(order_book[iline - 30:iline - 1, 0])
            moving_average[0:self.moving_window - 1] = moving_average[self.moving_window]
            order_book = np.column_stack((order_book, moving_average))

            for start_time in range(trade_start_time + self.time_horizon, trade_end_time, self.time_horizon):
                end_time = start_time + self.time_horizon
                decision_points = range(start_time, end_time, self.time_step)
                for decision_pt in decision_points:
                    if decision_pt == trade_end_time:
                        continue

                    daily_result_entry = ModelUtil.calculate_market_input(msg_book,
                                                                          order_book,
                                                                          decision_pt - self.time_step,
                                                                          decision_pt, self.moving_window)
                    print("Time " + str(decision_pt) + " " + str(daily_result_entry))
                    daily_result.append(daily_result_entry)
            date_string = msg_book_file.split("_")[1]
            formatted_date_string = ''.join(date_string.split("-"))
            output_filename = "market_new_" + formatted_date_string + ".csv"
            np.savetxt(join(self.output_folder, output_filename), daily_result, delimiter=",")
