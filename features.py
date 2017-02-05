import cv2
import numpy as np

from finch.dsp import grayscale, get_neighbors
from finch.cv2_util import keypoints_to_points

def orb_features(image, keypoints):
  if isinstance(keypoints, np.ndarray):
    size = 10
    keypoints = [cv2.KeyPoint(p[0], p[1], size) for p in keypoints]

  orb = cv2.ORB()
  new_keypoints, descriptors = orb.compute(image, keypoints)
  return new_keypoints, descriptors

def lbp_features(image, keypoints):
  image = grayscale(image)
  keypoints = keypoints_to_points(keypoints)
  features = []
  for x, y in keypoints:
    bitstring = (get_neighbors(image, x, y) <= image[y, x]).astype(int)
    features.append(sum(1<<i for i, b in enumerate(bitstring) if b))

  features = np.atleast_2d(features).T
  return features

def hog_features(image):
  hog_computer = cv2.HOGDescriptor()
  return hog_computer.compute(image)


if __name__ == '__main__':
  pass
