"""
Created on Sun Mar 31
@author: Wang Kunzhen
"""

from math import floor
from functools import reduce
import numpy as np


class Strategy:
    def __init__(self, cost, actions, inventory):
        self.cost = cost
        self.actions = actions
        self.inventory = inventory

    def __str__(self):
        return "Cost: " + str(self.cost) + " Actions: " + str(self.actions) + " Inventories: " + str(self.inventory)


class OptimizationEngine:
    """
    Given order book & message book entries for an episode, calculate the optimal strategy.
    Output is a Strategy object with an array of actions and its associated cost.
    """

    def __init__(self, order_book, msg_book, start_time, time_step, time_count, inv_step, inv_count, actions):
        self.order_book = order_book
        self.message_book = msg_book
        self.start_time = start_time
        self.time_step = time_step
        self.time_count = time_count
        self.inventory_step = inv_step
        self.inventory_count = inv_count
        self.actions = actions

    def mid_spread_from_order_book(self, order_book):
        return (order_book[0, 0] + order_book[0, 2]) / 2

    def cost_update(self, costs, inventories, next_results):
        next_period_results = [next_results[floor(i / self.inventory_step)] for i in inventories]
        tuples = zip(self.actions, costs, next_period_results)
        total_results = [Strategy(c + r.cost, [a] + r.actions, r.inventory) for (i, (a, c, r)) in enumerate(tuples)]
        return reduce(lambda x, y: x if x.cost == max(x.cost, y.cost) else y, total_results)

    def order_book_entries(self, time_index):
        if time_index == self.time_count:
            lower = self.start_time + (time_index - 1) * self.time_step
            upper = self.start_time + time_index * self.time_step
            order_book = [self.order_book[i]
                          for i in range(0, self.message_book.shape[0])
                          if lower <= self.message_book[i][0] < upper]
            if len(order_book) == 0:
                return np.asarray([[]])
            else:
                return np.asarray([order_book[-1]])
        else:
            lower = self.start_time + time_index * self.time_step
            upper = self.start_time + (time_index + 1) * self.time_step
            return np.array([self.order_book[i]
                            for i in range(0, len(self.order_book))
                            if lower <= self.message_book[i][0] < upper])

    def message_book_entries(self, time_index):
        lower = self.start_time + time_index * self.time_step
        upper = self.start_time + (time_index + 1) * self.time_step
        return np.array([m for m in self.message_book if lower <= m[0] < upper])

    def compute_optimal_solution(self, total_inventory, exe_engine):
        time = self.time_count
        while time >= 0 and (self.order_book_entries(time).shape[0] == 0 or self.order_book_entries(time).shape[1] == 0):
            time -= 1

        if time < 0:
            return Strategy(0, [], [])

        # Initialize for time T
        last_order_book = self.order_book_entries(time)
        results = [Strategy(exe_engine.cost_T(last_order_book, idx*self.inventory_step), [], [])
                   for idx in range(0, self.inventory_count + 1)]

        # Back Propagation
        for t in range(time-1, -1, -1):
            order_book = self.order_book_entries(t)
            message_book = self.message_book_entries(t)

            if message_book.shape[0] == 0:
                print("Skipping " + str(t))
                continue

            curr_results = []
            for idx in range(0, self.inventory_count + 1):
                executions = exe_engine.cost_other(message_book,
                                                   order_book,
                                                   idx * self.inventory_step,
                                                   self.actions)
                updated_result = self.cost_update(executions[0], executions[1], results)
                curr_results.append(Strategy(updated_result.cost,
                                             updated_result.actions,
                                             [idx * self.inventory_step] + updated_result.inventory))

            results = curr_results

        # Extract solution
        result = results[floor(total_inventory / self.inventory_step)]
        if len(result.actions) != self.time_count or len(result.inventory) != self.time_count:
            return Strategy(0, [], [])

        idx = 0
        while self.order_book_entries(idx).shape[0] == 0 or self.order_book_entries(idx).shape[1] == 0:
            idx += 1
        first_order_book = self.order_book_entries(idx)
        mid_spread = self.mid_spread_from_order_book(first_order_book)
        result = results[floor(total_inventory / self.inventory_step)]
        return Strategy(result.cost - mid_spread * total_inventory, result.actions, result.inventory)
