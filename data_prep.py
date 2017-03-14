from __future__ import division

from math import floor
import os

import cv2
import numpy as np
import pandas as pd

def image_dir_to_np_array(directory):
  return np.array([cv2.imread(directory + '/' + file) for file in os.listdir(directory)])

def scale(data, lower=0, upper=1, data_min=None, data_max=None):
  data_min = data_min if data_min is not None else data.min()
  data_max = data_max if data_max is not None else data.max()
  return np.nan_to_num(((upper - lower) * (data - data_min)) / (data_max - data_min) + lower)


def scale_dataframe(data, excluded_columns):
  for column in data.columns:
    if column not in excluded_columns:
      data[column] = scale(data[column], 0, 1)

  return data


def nominal_to_numeric(array):
  mapper = {name: i for i, name in enumerate(pd.unique(array))}
  return np.array([mapper[name] for name in array])


def nominal_to_one_hot(array):
  return pd.get_dummies(array).as_matrix()

def split(array, first_proportion):
  split_row = floor(len(array) * first_proportion)
  return array[:split_row], array[split_row:]

def shuffle_rows(array):
  copy = np.copy(array)
  np.random.shuffle(copy)
  return copy

def n_fold_train_test(dataset, n):
  size = len(dataset)

  test_amount = size // n
  train_amount = size - test_amount

  for test_start in range(0, size-1, test_amount):
    test_end = min(size, test_start + test_amount)

    test_indices = range(test_start, test_end)
    train_indices = range(0, test_start) + range(test_end, size)

    yield dataset[train_indices], dataset[test_indices]

def batch_generator(data, labels, batch_size=25):

  starting_point = 0

  while starting_point < len(data):
    data_batch = data[starting_point:starting_point+batch_size]
    label_batch = labels[starting_point:starting_point+batch_size]

    starting_point += batch_size

    yield data_batch, label_batch


if __name__ == '__main__':
  pass

if __name__ == '__main__':
  pass
