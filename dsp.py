from __future__ import division

import cv2
import numpy as np

def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold(nparray, thresh=None):
  def otsu_threshold(nparray):
    flattened = nparray.flatten()
    highest_variance = 0
    best_threshold = -1

    for T in range(0, 256):
      below_threshold = flattened[flattened < T]
      above_equal_threshold = flattened[flattened >= T]

      n1 = len(below_threshold)
      n2 = len(above_equal_threshold)
      mu_1 = np.mean(below_threshold)
      mu_2 = np.mean(above_equal_threshold)

      between_class_variance = n1 * n2 * (mu_1 - mu_2)**2

      if between_class_variance > highest_variance:
        highest_variance = between_class_variance
        best_threshold = T

    return best_threshold

  thresh = otsu_threshold(nparray) if thresh is None else thresh
  output = np.copy(nparray)
  output[output < thresh] = 0
  output[output >= thresh] = 255
  return output

def image_normalize(nparray):
  output = np.copy(nparray)
  output -= np.min(output)
  output /= np.max(output)
  output *= 255
  return output

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
  return result
 
def get_neighbors(image, x, y):
  # starts at top left, goes clockwise
  width = image.shape[1]
  height = image.shape[0]

  neighbor_locations = [
    (max(0, y-1), max(0, x-1)),
    (max(0, y-1), x),
    (max(0, y-1), min(width-1, x+1)),
    (y, min(width-1, x+1)),
    (min(height-1, y+1), min(width-1, x+1)),
    (min(height-1, y+1), x),               
    (min(height-1, y+1), max(0, x-1)),     
    (y, max(0, x-1)),     
  ]

  return np.array([
    image[neighbor_locations[0]],
    image[neighbor_locations[1]],
    image[neighbor_locations[2]],
    image[neighbor_locations[3]],
    image[neighbor_locations[4]],
    image[neighbor_locations[5]],
    image[neighbor_locations[6]],
    image[neighbor_locations[7]]
  ])

def resize_to_width(image, width, square=False):
  ratio = width / image.shape[1]
  dimensions = (width, int(image.shape[0] * ratio)) if not square else (width, width)
  return cv2.resize(image, dimensions)

def intersection_over_union(a, b):
  (a_x_start, a_y_start, a_window_width, a_window_height) = a
  (b_x_start, b_y_start, b_window_width, b_window_height) = b

  a_x_end = a_x_start + a_window_width
  a_y_end = a_y_start + a_window_height
  b_x_end = b_x_start + b_window_width
  b_y_end = b_y_start + b_window_height

  a_area = a_window_width * a_window_height
  b_area = b_window_width * b_window_height

  x_overlap = max(min(a_x_end, b_x_end) - max(a_x_start, b_x_start), 0)
  y_overlap = max(min(a_y_end, b_y_end) - max(a_y_start, b_y_start), 0)

  intersection_area = x_overlap * y_overlap
  union_area = a_area + b_area - intersection_area

  return intersection_area / union_area

def fft(data, power_spectrum=False):
  k = 2 if power_spectrum else 1
  return np.absolute(np.fft.fft(data))[:len(data)/2]**k

if __name__ == '__main__':
  from finch.viz import scatter
  from finch.datasets import noisy_sin

  print('noisy_sin()', noisy_sin())

  scatter(fft(noisy_sin(), power_spectrum=True))
