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
moving_window = int(sys.argv[4])

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
    moving_average = np.zeros([order_book.shape[0], 1])
    for iline in range(order_book.shape[0]):
        if iline < moving_window:
            continue
        else:
            moving_average[iline] = np.mean(order_book[iline - 30:iline - 1, 0])
    moving_average[0:moving_window - 1] = moving_average[moving_window]

    order_book = np.column_stack((order_book, moving_average))

    for start_time in range(trade_start_time + time_horizon, trade_end_time, time_horizon):
        end_time = start_time + time_horizon
        decision_points = range(start_time, end_time, time_step)
        for decision_pt in decision_points:
            order_book_step = np.asarray([order_book[i][:]
                                          for i in range(0, msg_book.shape[0])
                                          if decision_pt - time_step < msg_book[i][0] <= decision_pt])
            if decision_pt == trade_end_time:
                continue

            order_book_moving_avg = (order_book_step[0, -1] - order_book_step[0, 0]) / order_book_step[0, 0]
            order_book_mismatch = abs(order_book_step[0, 1] - order_book_step[0, 3])
            order_book_spread = order_book_step[0, 0] - order_book_step[0, 2]
            order_book_trend = (order_book_step[0, 0] - order_book_step[0, -1]) / order_book_step[0, -1]

            # normalization
            order_book_moving_avg = floor(order_book_moving_avg * 1e5)
            order_book_mismatch = floor(order_book_mismatch / 100)
            order_book_spread = floor(order_book_spread / 100)
            order_book_trend = floor(order_book_trend * 1e5)

            daily_result_entry = [order_book_spread, order_book_trend, order_book_mismatch, order_book_moving_avg]
            print("Time " + str(decision_pt) + " " + str(daily_result_entry))
            daily_result.append(daily_result_entry)
    date_string = msg_book_file.split("_")[1]
    formatted_date_string = ''.join(date_string.split("-"))
    output_filename = "market_new_" + formatted_date_string + ".csv"
    np.savetxt(join(output_folder, output_filename), daily_result, delimiter=",")
