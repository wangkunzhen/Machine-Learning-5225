"""
Created on Sun Apr 7
@author: Wang Kunzhen
"""

from ExecutionEngine import ExecutionEngine
from OptimizationEngine import OptimizationEngine
from ModelUtil import ModelUtil
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np


class ModelEvaluator:
    @staticmethod
    def evaluate(model, test_folder, volume, volume_step, time, time_step, start_time, end_time, possible_actions, action_step, window_size):
        msg_book_files = [join(test_folder, f) for f in listdir(test_folder)
                          if isfile(join(test_folder, f)) and f.endswith("message_5.csv")]
        order_book_files = [join(test_folder, f) for f in listdir(test_folder)
                            if isfile(join(test_folder, f)) and f.endswith("orderbook_5.csv")]

        msg_book_files.sort()
        order_book_files.sort()

        evaluation_result = np.zeros((len(msg_book_files), int((end_time - time - start_time) / time_step), 5))

        exe_engine = ExecutionEngine()
        for day_idx in range(0, len(msg_book_files)):
            msg_book = np.asarray(pd.read_csv(msg_book_files[day_idx], header=None))
            order_book = np.asarray(pd.read_csv(order_book_files[day_idx], header=None))
            for start in range(start_time, end_time-time, time_step):
                end = start + time
                msg_book_episode = np.asarray([msg for msg in msg_book if start <= msg[0] < end])
                order_book_episode = np.asarray([order_book[i][:]
                                                 for i in range(0, msg_book.shape[0])
                                                 if start <= msg_book[i][0] < end])

                row_idx = int((start - start_time) / time_step)

                # Model Cost
                model_cost = 0
                total_steps = int(time / time_step)
                remaining_inventory = volume
                for t in range(start, end, time_step):
                    market_variables = ModelUtil.calculate_market_input(msg_book_episode, order_book_episode, t,
                                                                        t + time_step, window_size)
                    remaining_steps = total_steps - (t - start) / time_step
                    model_input = ModelUtil.model_input(market_variables, remaining_inventory, remaining_steps)
                    predictions = model.predict(np.array([model_input]))
                    action = list(possible_actions)[np.argmax(predictions)]
                    msg_book_step = np.asarray([m for m in msg_book_episode if t < m[0] <= t + time_step])
                    order_book_step = np.asarray([order_book_episode[i] for i in range(0, msg_book_episode.shape[0]) if
                                                  t < msg_book_episode[i, 0] <= t + time_step])
                    cost, remaining = exe_engine.cost_other(msg_book_step, order_book_step, remaining_inventory,
                                                            [action])
                    model_cost += cost[0]
                    remaining_inventory = remaining[0]
                    if remaining_inventory is 0:
                        break
                last_period_cost = exe_engine.cost_T(order_book_step, remaining_inventory)
                evaluation_result[day_idx, row_idx, 0] = model_cost + last_period_cost

                # Optimal Cost
                opt_engine = OptimizationEngine(order_book_episode,
                                                msg_book_episode,
                                                start,
                                                time_step,
                                                int(time / time_step),
                                                volume_step,
                                                int(volume / volume_step),
                                                possible_actions)
                optimal_cost = opt_engine.compute_optimal_solution(volume, exe_engine).cost
                evaluation_result[day_idx, row_idx, 1] = optimal_cost

                percentage = (model_cost - optimal_cost) / optimal_cost
                evaluation_result[day_idx, row_idx, 2] = percentage

                # Mid-spread Strategy
                mid_spread_price = (order_book_episode[0, 0] + order_book_episode[0, 2]) / 2
                mid_spread_action = (order_book_episode[0, 0] - mid_spread_price) / action_step
                mid_spred_opt_engine = OptimizationEngine(order_book_episode,
                                                          msg_book_episode,
                                                          start,
                                                          time_step,
                                                          int(time / time_step),
                                                          volume_step,
                                                          int(volume / volume_step),
                                                          [mid_spread_action])
                mid_spread_cost = mid_spred_opt_engine.compute_optimal_solution(volume, exe_engine).cost
                evaluation_result[day_idx, row_idx, 3] = mid_spread_cost

                mid_spread_improvement = (model_cost - mid_spread_cost) / mid_spread_cost
                evaluation_result[day_idx, row_idx, 4] = mid_spread_improvement

                print("Evaluated " + str(start) + " Optimal: " + str(percentage * 100)
                      + "%, Mid-spread: " + str(mid_spread_improvement * 100) + "% " + str(evaluation_result[day_idx, row_idx, :]))

        return np.asarray(evaluation_result)

