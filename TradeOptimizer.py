#!/usr/bin/env python

import pandas as pd
import sys
from os import listdir
from os.path import isfile, join
from OptimizationEngine import OptimizationEngine
from all_cost import ExecutionEngine

data_folder = sys.argv[1]
output_folder = data_folder + "/output/"

volume = sys.argv[2]
volume_step = sys.argv[3]
time_horizon = sys.argv[4]
time_step = sys.argv[5]

actions = range(2000, -300, -50)

msg_book_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) & f.endswith("message_5.csv")]
order_book_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) & f.endswith("orderbook_5.csv")]

assert len(msg_book_files) == len(order_book_files)

for (msg_book_file, order_book_file) in zip(msg_book_files, order_book_files):
    with pd.read_csv(msg_book_file, header=None) as msg_book, pd.read_csv(order_book_file, header=None) as order_book:
        assert len(msg_book) == len(order_book)
        print("Loaded " + msg_book_file + " + " + order_book_file)

        trade_start_time = int(9.5 * 60 * 60)
        trade_end_time = int(16 * 60 * 60)
        for start_time in range(trade_start_time, trade_end_time, time_horizon):
            end_time = start_time + time_horizon
            msg_book_episode = [msg for msg in msg_book if start_time <= msg[0] < end_time]
            order_book_episode = [tup[0] for tup in zip(order_book, msg_book) if start_time <= tup[1] < end_time]
            optimize_engine = OptimizationEngine(order_book_episode,
                                                 msg_book_episode,
                                                 time_step,
                                                 int(time_horizon / time_step),
                                                 volume_step,
                                                 int(volume / volume_step),
                                                 actions)
            execution_engine = ExecutionEngine()
            strategy = optimize_engine.compute_optimal_solution(volume, execution_engine)
            print(strategy)
