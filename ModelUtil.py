import numpy as np
from math import floor


class ModelUtil:
    @staticmethod
    def calculate_market_input(msg_book, order_book, start, end, moving_window):
        order_book_step = np.asarray([order_book[i][:]
                                      for i in range(0, msg_book.shape[0])
                                      if start < msg_book[i][0] <= end])

        if not order_book_step.shape[0]:
            return np.asarray([0, 0, 0, 0])

        order_book_window = np.asarray([order_book[i][0] for i in range(0, msg_book.shape[0]) if msg_book[i][0] <= start])
        if order_book_window.shape[0] < moving_window:
            moving_avg = np.mean(order_book[:moving_window][0])
        else:
            moving_avg = np.mean(order_book_window[-moving_window:])
        order_book_moving_avg = (moving_avg - order_book_step[0, 0]) / order_book_step[0, 0]
        order_book_mismatch = abs(order_book_step[0, 1] - order_book_step[0, 3])
        order_book_spread = order_book_step[0, 0] - order_book_step[0, 2]
        order_book_trend = (order_book_step[0, 0] - moving_avg) / moving_avg

        # normalization
        order_book_moving_avg = floor(order_book_moving_avg * 1e5)
        order_book_mismatch = floor(order_book_mismatch / 100)
        order_book_spread = floor(order_book_spread / 100)
        order_book_trend = floor(order_book_trend * 1e5)

        return np.asarray([order_book_spread, order_book_trend, order_book_mismatch, order_book_moving_avg])

    @staticmethod
    def running_mean(x, n):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    @staticmethod
    def model_input(market_variables, inventory, remaining_time):
        return np.append(market_variables, [inventory, remaining_time])
