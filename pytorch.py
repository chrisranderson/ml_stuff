import numpy as np
import torch

def to_scalar(x):
  return x.cpu().float().data[0]

def nparray(x):
  return x.float().cpu().data.numpy()

def rotate_axis(x):
  return np.moveaxis(x, 0, 2)
