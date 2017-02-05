import cv2
import numpy as np

from finch.dsp import grayscale

def harris_corners(image):
  i = 2
  j = 11
  k = 4
  copy = np.copy(image)
  corners = cv2.cornerHarris(grayscale(copy), i, j, k/20)
  # dst = cv2.dilate(dst, None)
  # copy[dst > 0.01 * dst.max()] =  [0, 0, 255]
  # cv2.imwrite('{}-{}-{}.png'.format(i, j, k), copy)
  return [cv2.KeyPoint(corner[0], corner[1], 10) for corner in corners]

def random_points(image, n=10):
  xs = np.random.random_integers(image.shape[0], size=(n, 1))
  ys = np.random.random_integers(image.shape[1], size=(n, 1))

  return np.hstack([xs, ys])

################################################################################

if __name__ == '__main__':
  pass
