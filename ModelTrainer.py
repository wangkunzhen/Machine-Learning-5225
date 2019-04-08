import numpy as np
from math import floor


class ModelTrainer:
    @staticmethod
    def calculate_market_input(msg_book, order_book, start, end, moving_window):
        moving_average = np.zeros([order_book.shape[0], 1])
        for iline in range(order_book.shape[0]):
            if iline < moving_window:
                continue
            else:
                moving_average[iline] = np.mean(order_book[iline - 30:iline - 1, 0])
        moving_average[0:moving_window - 1] = moving_average[moving_window]

        order_book_padded = np.column_stack((order_book, moving_average))

        order_book_step = np.asarray([order_book_padded[i][:]
                                      for i in range(0, msg_book.shape[0])
                                      if start < msg_book[i][0] <= end])
        order_book_moving_avg = (order_book_step[0, -1] - order_book_step[0, 0]) / order_book_step[0, 0]
        order_book_mismatch = abs(order_book_step[0, 1] - order_book_step[0, 3])
        order_book_spread = order_book_step[0, 0] - order_book_step[0, 2]
        order_book_trend = (order_book_step[0, 0] - order_book_step[0, -1]) / order_book_step[0, -1]

        # normalization
        order_book_moving_avg = floor(order_book_moving_avg * 1e5)
        order_book_mismatch = floor(order_book_mismatch / 100)
        order_book_spread = floor(order_book_spread / 100)
        order_book_trend = floor(order_book_trend * 1e5)

        return np.asarray([order_book_spread, order_book_trend, order_book_mismatch, order_book_moving_avg])

    @staticmethod
    def model_input(market_variables, inventory, remaining_time):
        return np.append(market_variables, [inventory, remaining_time])
