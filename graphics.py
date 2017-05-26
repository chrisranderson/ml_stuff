import cv2
import numpy as np

def pad(image, amount):
  return np.pad(image, ((amount, amount), (amount, amount), (0, 0)), mode='constant')

def put_text(image, text_string, row, column, size=2, color=(255, 255, 255)):
  cv2.putText(image,
              text_string, 
              (column-10, row), 
              cv2.FONT_HERSHEY_SIMPLEX, 
              0.5, 
              color)

def angle_to_point(angle):
  radius = IMAGE_SIZE / sqrt(2)
  x = radius * cos(angle)
  y = radius * sin(angle)
  return x, y
