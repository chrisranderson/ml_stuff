from __future__ import division

from math import floor

import pandas as pd
import numpy as np

def scale(data, lower=0, upper=1):
  return ((upper - lower) * (data - data.min())) / (data.max() - data.min()) + lower

def nominal_to_numeric(array):
  mapper = {name: i for i, name in enumerate(pd.unique(array))}
  return np.array([mapper[name] for name in array])

def nominal_to_one_hot(array):
  return pd.get_dummies(array).as_matrix()

def n_fold_train_test(dataset, n):
  size = len(dataset)

  test_amount = size // n
  train_amount = size - test_amount

  for test_start in range(0, size-1, test_amount):
    test_end = min(size, test_start + test_amount)

    test_indices = range(test_start, test_end)
    train_indices = range(0, test_start) + range(test_end, size)

    yield dataset[train_indices], dataset[test_indices]

if __name__ == '__main__':
  pass
