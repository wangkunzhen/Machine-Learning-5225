#!/usr/bin/env python

"""
Created on Sun Mar 31
@author: Wang Kunzhen
"""

import sys
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from OptimizationEngine import OptimizationEngine
from ExecutionEngine import ExecutionEngine

data_folder = sys.argv[1]
output_folder = data_folder + "/output/"

volume = int(sys.argv[2])
volume_step = int(sys.argv[3])
time_horizon = int(sys.argv[4])
time_step = int(sys.argv[5])

actions = range(2000, -300, -50)

msg_book_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith("message_5.csv")]
order_book_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith("orderbook_5.csv")]

assert len(msg_book_files) == len(order_book_files)

for (msg_book_file, order_book_file) in zip(msg_book_files, order_book_files):
    msg_book = np.asarray(pd.read_csv(join(data_folder, msg_book_file), header=None))
    order_book = np.asarray(pd.read_csv(join(data_folder, order_book_file), header=None))

    assert len(msg_book) == len(order_book)
    print("Loaded " + msg_book_file + " + " + order_book_file)

    trade_start_time = int(9.5 * 60 * 60)
    trade_end_time = int(16 * 60 * 60)
    for start_time in range(trade_start_time, trade_end_time, time_horizon):
        end_time = start_time + time_horizon
        msg_book_episode = np.asarray([msg for msg in msg_book if start_time <= msg[0] < end_time])
        order_book_episode = np.asarray([order_book[i][:]
                                        for i in range(0, msg_book.shape[0])
                                        if start_time <= msg_book[i][0] < end_time])
        optimize_engine = OptimizationEngine(order_book_episode,
                                             msg_book_episode,
                                             start_time,
                                             time_step,
                                             int(time_horizon / time_step),
                                             volume_step,
                                             int(volume / volume_step),
                                             actions)
        execution_engine = ExecutionEngine()
        strategy = optimize_engine.compute_optimal_solution(volume, execution_engine)
        print("Done execution for " + str(start_time))
        print(strategy)
