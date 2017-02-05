from math import sin

import numpy as np
import cv2

def single_image(grayscale=False):
  image = cv2.imread('data/grace.png')

  if grayscale:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return image

def random_set(rows=20, columns=5):
  return np.random.standard_normal((rows, columns)),\
         np.random.standard_normal(rows)

def noisy_sin():
  return [sin(x) + sin(5*x) + np.random.standard_normal(1)[0]/5 for x in np.arange(0, 10, .01)]

if __name__ == '__main__':
  print('random_set(20, 5)', random_set(20, 5))
