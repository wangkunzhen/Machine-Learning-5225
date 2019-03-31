import numpy as np
from all_cost import ExecutionEngine
from math import floor
from functools import  reduce


class Result:
    def __init__(self, cost, actions):
        self.cost = cost
        self.actions = actions


class OptimizationEngine:
    def __init__(self, order_book, message_book, possible_inventories):
        self.order_book = order_book
        self.message_book = message_book
        self.possible_inventories = possible_inventories

    def order_book_entries(self, time):
        return []

    def message_book_entries(self, time):
        return []

    @staticmethod
    def mid_spread_from_order_book(order_book):
        return ((order_book[:, 0] + order_book[:, 2]) / 2)[0]

    @staticmethod
    def cost_function(actions, costs, inventories, next_results, unit_inventory):
        next_period_results = [next_results[floor(i / unit_inventory)] for i in inventories]
        tuples = zip(actions, costs, next_period_results)
        total_results = [Result(c + r.cost, [a] + r.actions) for (a, c, r) in tuples]
        return reduce(lambda x, y: x if x.cost == min(x.cost, y.cost) else y, total_results)

    def calculate_optimal_solution(self, actions, periods_count, total_inventory):
        inventories_count = len(self.possible_inventories)
        results = np.array(inventories_count)

        # Initialize for time T
        for idx in range(0, inventories_count):
            order_book = self.order_book_entries(periods_count)
            mid_spread = OptimizationEngine.mid_spread_from_order_book(order_book)
            inventory = self.possible_inventories[idx]
            cost = ExecutionEngine.cost_T(order_book, mid_spread, inventory)
            results[idx] = Result(cost, [])

        # Back Propagation
        for t in range(periods_count-1, 0, -1):
            order_book = self.order_book_entries(t)
            message_book = self.message_book_entries(t)
            mid_spread = OptimizationEngine.mid_spread_from_order_book(order_book)
            curr_results = np.array(inventories_count)

            for idx in range(0, inventories_count):
                inventory = self.possible_inventories[idx]
                costs, remaining_inventories = ExecutionEngine.cost_other(message_book,
                                                                          order_book,
                                                                          inventory,
                                                                          mid_spread,
                                                                          actions)
                curr_results[idx] = self.cost_function(costs, remaining_inventories, results)

            results = curr_results

        # Extract solution
        return results[total_inventory]
