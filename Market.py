#!/usr/bin/env python

"""
Created on Sun Mar 31
@author: Wang Kunzhen
"""

import sys
import pandas as pd
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, exists
from math import floor

data_folder = sys.argv[1]
output_folder = data_folder + "/output"

if not exists(output_folder):
    mkdir(output_folder)

time_horizon = int(sys.argv[2])
time_step = int(sys.argv[3])

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
    for start_time in range(trade_start_time + time_horizon, trade_end_time, time_horizon):
        end_time = start_time + time_horizon
        decision_points = range(start_time, end_time, time_step)
        for decision_pt in decision_points:
            order_book_step = np.asarray([order_book[i][:]
                                          for i in range(0, msg_book.shape[0])
                                          if decision_pt - time_step < msg_book[i][0] <= decision_pt])
            if decision_pt == trade_end_time:
                continue

            relative_order_book = [(np.average(order_book_step[:-1, 0]) - order_book_step[-1, 0]) / order_book_step[-1, 0],
                                   (np.average(order_book_step[:-1, 2]) - order_book_step[-1, 2]) / order_book_step[-1, 2]] # %
            order_book_spread = np.average((order_book_step[:, 0] - order_book_step[:, 2]) / order_book_step[:, 0]) # %
            order_book_trend = [np.average((order_book_step[1:, 0] - order_book_step[:-1, 0]) / order_book_step[-1, 0]),
                                np.average((order_book_step[1:, 2] - order_book_step[:-1, 2]) / order_book_step[-1, 2])]
            order_book_trend = [floor(x * 1e6) for x in order_book_trend]
            relative_order_book = [floor(x * 1e4) for x in relative_order_book]
            order_book_spread = floor(order_book_spread * 1e4)
            daily_result_entry = [relative_order_book[0], relative_order_book[1], order_book_spread, order_book_trend[0], order_book_trend[1]]
            print("Time " + str(decision_pt) + " " + str(daily_result_entry))
            daily_result.append(daily_result_entry)
    date_string = msg_book_file.split("_")[1]
    formatted_date_string = ''.join(date_string.split("-"))
    output_filename = "market_new_" + formatted_date_string + ".csv"
    np.savetxt(join(output_folder, output_filename), daily_result, delimiter=",")
