import numpy as np
from math import floor


class ModelTrainer:
    @staticmethod
    def calculate_market_input(msg_book, order_book, start, end):
        order_book_step = np.asarray([order_book[i][:]
                                      for i in range(0, msg_book.shape[0])
                                      if start < msg_book[i][0] <= end])
        relative_order_book = [(np.average(order_book_step[:-1, 0]) - order_book_step[-1, 0]) / order_book_step[-1, 0],
                               (np.average(order_book_step[:-1, 2]) - order_book_step[-1, 2]) / order_book_step[
                                   -1, 2]]  # %
        order_book_spread = np.average((order_book_step[:, 0] - order_book_step[:, 2]) / order_book_step[:, 0])  # %
        order_book_trend = [np.average((order_book_step[1:, 0] - order_book_step[:-1, 0]) / order_book_step[-1, 0]),
                            np.average((order_book_step[1:, 2] - order_book_step[:-1, 2]) / order_book_step[-1, 2])]
        order_book_trend = [floor(x * 1e6) for x in order_book_trend]
        relative_order_book = [floor(x * 1e4) for x in relative_order_book]
        order_book_spread = floor(order_book_spread * 1e4)
        return np.asarray([relative_order_book[0],
                           relative_order_book[1],
                           order_book_spread,
                           order_book_trend[0],
                           order_book_trend[1]])

    @staticmethod
    def model_input(market_variables, inventory, remaining_time):
        return np.append(market_variables, [inventory, remaining_time])
