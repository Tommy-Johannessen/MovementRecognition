import pickle

import numpy as np
from sklearn.model_selection import train_test_split


class DataBatcher:
    def __init__(self, path, split_size=.5, do_shuffle=True):
        with open(path, 'rb') as bf:
            features, labels = pickle.load(bf)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features,
            labels,
            test_size=split_size,
            random_state=42,
            shuffle=do_shuffle
        )
        self.train_index = 0
        self.test_index = 0

        self.shape = features.shape
        self.n_labels = len(set(labels))

    def get_info(self):
        return self.shape[1], self.shape[2], self.n_labels

    @staticmethod
    def make_labels_one_hot(values):
        one_hot = np.zeros((values.size, values.max() + 1))
        one_hot[np.arange(values.size), values] = 1

        return one_hot

    def next_batch_train(self, size):
        if self.train_index + size > len(self.x_train):
            remaining = size - (len(self.x_train) - self.train_index)

            first_part_x = self.x_train[self.train_index:]
            first_part_y = self.y_train[self.train_index:]

            self.train_index = remaining

            rem_x = self.x_train[:self.train_index]
            rem_y = self.y_train[:self.train_index]

            x = np.concatenate((first_part_x, rem_x))
            y = np.concatenate((first_part_y, rem_y))

        else:
            end_index = self.train_index + size

            x = self.x_train[self.train_index:end_index]
            y = self.y_train[self.train_index:end_index]

            self.train_index = end_index

        return x, y

    def get_testing_data(self):
        return self.x_test, self.y_test

    def reset(self):
        self.train_index = 0
        self.test_index = 0
