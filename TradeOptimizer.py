import time
import csv
import pandas as pd
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import isfile, join, exists
from OptimizationEngine import OptimizationEngine
from ExecutionEngine import ExecutionEngine


class TradeOptimizer:
    """
    Given message books & order books, compute the optimal strategy to invest for a training episode assuming all information is known at the beginning.
    Writes the optimal strategy to the output folder as per provided.
    """

    def __init__(self, data_folder, output_folder, volume, volume_step, time_horizon, time_step, max_action, min_action, action_step, trade_start, trade_end):
        """
        :param data_folder: Path to data folder
        :param output_folder: Path to output folder
        :param volume: Target Volume
        :param volume_step: Volume fragmentation
        :param time_horizon: Target time horizon
        :param time_step: Time fragmentation
        :param max_action: Maximum action price relative to lowest-bid to be considered when optimizing
        :param min_action: Minimum action price relative to lowest-bid to be considered when optimizing
        :param action_step: Action fragmentation
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.volume = volume
        self.volume_step = volume_step
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.max_action = max_action
        self.min_action = min_action
        self.action_step = action_step
        self.trade_start = trade_start
        self.trade_end = trade_end
        TradeOptimizer.reset_directory_if_needed(self.output_folder)

    @staticmethod
    def reset_directory_if_needed(path):
        if exists(path):
            rmtree(path)
        mkdir(path)

    def optimize_trade_execution(self):
        actions = range(self.max_action, self.min_action, self.action_step)
        trade_start_time = self.trade_start
        trade_end_time = self.trade_end

        msg_book_files = [f for f in listdir(self.data_folder) if
                          isfile(join(self.data_folder, f)) and f.endswith("message_5.csv")]
        order_book_files = [f for f in listdir(self.data_folder) if
                            isfile(join(self.data_folder, f)) and f.endswith("orderbook_5.csv")]
        assert len(msg_book_files) == len(order_book_files)

        msg_book_files.sort()
        order_book_files.sort()

        for (msg_book_file, order_book_file) in zip(msg_book_files, order_book_files):
            msg_book = np.asarray(pd.read_csv(join(self.data_folder, msg_book_file), header=None))
            order_book = np.asarray(pd.read_csv(join(self.data_folder, order_book_file), header=None))

            assert len(msg_book) == len(order_book)
            print("Loaded " + msg_book_file + " + " + order_book_file)

            daily_result = []
            for start_time in range(trade_start_time, trade_end_time, self.time_horizon):
                end_time = start_time + self.time_horizon
                msg_book_episode = np.asarray([msg for msg in msg_book if start_time <= msg[0] < end_time])
                order_book_episode = np.asarray([order_book[i][:]
                                                 for i in range(0, msg_book.shape[0])
                                                 if start_time <= msg_book[i][0] < end_time])
                optimize_engine = OptimizationEngine(order_book_episode,
                                                     msg_book_episode,
                                                     start_time,
                                                     self.time_step,
                                                     int(self.time_horizon / self.time_step),
                                                     self.volume_step,
                                                     int(self.volume / self.volume_step),
                                                     actions)
                start = time.time()
                execution_engine = ExecutionEngine()
                strategy = optimize_engine.compute_optimal_solution(self.volume, execution_engine)
                elapse_time = time.time() - start
                print("Done execution for " + str(start_time) + " using " + str(elapse_time))
                print("Strategy: " + str(strategy))
                if len(strategy.actions) == 0:
                    daily_result_entry = [0] * int(2 * (self.time_horizon / self.time_step) + 1)
                else:
                    daily_result_entry = [item for tup in zip(strategy.actions, strategy.inventory) for item in
                                          [tup[0], tup[1]]] + [strategy.cost]
                daily_result.append(daily_result_entry)

            date_string = msg_book_file.split("_")[1]
            formatted_date_string = ''.join(date_string.split("-"))
            output_filename = "private_" + formatted_date_string + ".csv"
            with open(join(self.output_folder, output_filename), "w") as f:
                writer = csv.writer(f)
                writer.writerows(daily_result)
