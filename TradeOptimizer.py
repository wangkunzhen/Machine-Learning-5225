#!/usr/bin/env python

"""
Created on Sun Mar 31
@author: Wang Kunzhen
"""

import time
import sys
import csv
import pandas as pd
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import isfile, join, exists
from OptimizationEngine import OptimizationEngine
from ExecutionEngine import ExecutionEngine

data_folder = sys.argv[1]
output_folder = data_folder + "/output"

if exists(output_folder):
    rmtree(output_folder)
mkdir(output_folder)

volume = int(sys.argv[2])
volume_step = int(sys.argv[3])
time_horizon = int(sys.argv[4])
time_step = int(sys.argv[5])
max_action = int(sys.argv[6])
min_action = int(sys.argv[7])
action_step = int(sys.argv[8])

actions = range(max_action, min_action, action_step)

msg_book_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith("message_5.csv")]
order_book_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and f.endswith("orderbook_5.csv")]

msg_book_files.sort()
order_book_files.sort()

assert len(msg_book_files) == len(order_book_files)

for (msg_book_file, order_book_file) in zip(msg_book_files, order_book_files):
    msg_book = np.asarray(pd.read_csv(join(data_folder, msg_book_file), header=None))
    order_book = np.asarray(pd.read_csv(join(data_folder, order_book_file), header=None))

    assert len(msg_book) == len(order_book)
    print("Loaded " + msg_book_file + " + " + order_book_file)

    trade_start_time = int(9.5 * 60 * 60)
    trade_end_time = int(16 * 60 * 60)
    daily_result = []
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
        start = time.time()
        execution_engine = ExecutionEngine()
        strategy = optimize_engine.compute_optimal_solution(volume, execution_engine)
        elapse_time = time.time() - start
        print("Done execution for " + str(start_time) + " using " + str(elapse_time))
        print(strategy)
        if len(strategy.actions) == 0:
            daily_result_entry = [0] * int(2 * (time_horizon / time_step) + 1)
        else:
            daily_result_entry = [item for tup in zip(strategy.actions, strategy.inventory) for item in
                                  [tup[0], tup[1]]] + [strategy.cost]
        daily_result.append(daily_result_entry)

    date_string = msg_book_file.split("_")[1]
    formatted_date_string = ''.join(date_string.split("-"))
    output_filename = "private_" + formatted_date_string + ".csv"
    with open(join(output_folder, output_filename), "w") as f:
        writer = csv.writer(f)
        writer.writerows(daily_result)
