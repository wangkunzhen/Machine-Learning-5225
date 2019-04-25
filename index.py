#!/usr/bin/env python

import sys
from os.path import join, exists
from os import mkdir
from Market import Market
from TradeOptimizer import TradeOptimizer
from Model import Model
from ModelEvaluator import ModelEvaluator

# arguments
if not len(sys.argv) <= 1:
	print("Please specify a folder containing the data. The folder should have two subfolders named Train and Test correspondingly, each containing message book and order book files.")
	sys.exit(1)

data_folder = sys.argv[1] #

volume = 1000
if len(sys.argv) >= 2:
	volume = int(sys.argv[2]) # 1000

volume_step = 50
if len(sys.argv) >= 3:
	volume_step = int(sys.argv[3]) # 50

time_horizon = 120
if len(sys.argv) >= 4:
	time_horizon = int(sys.argv[4]) # 120

time_step = 30
if len(sys.argv) >= 5:
	time_step = int(sys.argv[5]) # 30

max_action = 3000
if len(sys.argv) >= 6:
	max_action = int(sys.argv[6]) # 3000

min_action = -3000
if len(sys.argv) >= 7:
	min_action = int(sys.argv[7]) # -3000

action_step = 50
if len(sys.argv) >= 8:
	action_step = int(sys.argv[8]) # 50

moving_window = 50
if len(sys.argv) >= 9:
	moving_window = int(sys.argv[9]) # 

output_folder = join(data_folder, "output")
market_folder = join(output_folder, "Market")
private_folder = join(output_folder, "Private")
train_data_folder = join(data_folder, "Train")
train_market_output = join(market_folder, "Train")
train_private_output = join(private_folder, "Train")
test_data_folder = join(data_folder, "Test")
test_market_output = join(market_folder, "Test")
test_private_output = join(private_folder, "Test")
model_output = join(output_folder, "Model")

if not exists(output_folder):
    mkdir(output_folder)

if not exists(market_folder):
    mkdir(market_folder)

if not exists(private_folder):
    mkdir(private_folder)

if not exists(model_output):
    mkdir(model_output)

trade_start = int(9.5*60*60)
trade_end = int(16*60*60)
window_size = 1


# compute market variable
Market(train_data_folder, train_market_output, time_horizon, time_step, moving_window, trade_start, trade_end).load()
Market(test_data_folder, test_market_output, time_horizon, time_step, moving_window, trade_start, trade_end).load()

# compute optimal strategy
train_opt = TradeOptimizer(train_data_folder, train_private_output, volume, volume_step, time_horizon, time_step,
                           max_action, min_action, -action_step, trade_start, trade_end)
train_opt.optimize_trade_execution()

private_opt = TradeOptimizer(test_data_folder, train_private_output, volume, volume_step, time_horizon, time_step,
                             max_action, min_action, -action_step, trade_start, trade_end)
private_opt.optimize_trade_execution()


# fit model
model_path = join(model_output, "model.dat")

model_trainer = Model(private_folder, market_folder, volume, volume_step, time_horizon, time_step,
                      max_action, min_action, action_step, window_size)
model, loss, accuracy = model_trainer.fit_model(500)

# evaluate model
possible_actions = range(max_action, min_action - action_step, -action_step)
evaluation_result = ModelEvaluator.evaluate(model, test_data_folder, volume, volume_step, time_horizon, time_step,
                                            trade_start, trade_end, possible_actions, action_step, moving_window)
with open(join(model_output, "evaluation.csv")) as file:
    file.writelines(evaluation_result)
