from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np


class FileUtil:
    @staticmethod
    def load_msg_book(test_folder):
        msg_book_files = [f for f in listdir(test_folder)
                          if isfile(join(test_folder, f)) and f.endswith("message_5.csv")]
        msg_book_files.sort()
        if len(msg_book_files) is 0:
            return np.array([])

        msg_book = np.asarray(pd.concat([pd.read_csv(join(test_folder, f), header=None) for f in msg_book_files]))
        return msg_book

    @staticmethod
    def load_order_book(test_folder):
        order_book_files = [f for f in listdir(test_folder)
                            if isfile(join(test_folder, f)) and f.endswith("orderbook_5.csv")]
        order_book_files.sort()
        if len(order_book_files) is 0:
            return np.array([])

        order_book = np.asarray(pd.concat([pd.read_csv(join(test_folder, f), header=None) for f in order_book_files]))
        return order_book
