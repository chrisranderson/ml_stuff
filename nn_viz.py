from __future__ import division, print_function

from collections import deque

import tensorflow as tf
import numpy as np
import cv2
cv2_font = cv2.FONT_HERSHEY_SIMPLEX

from finch.data_prep import scale

WIDTH = 1800
HEIGHT = 1000
PADDING = 20
TEXT_SIZE = 0.3

global_step = 0
global_mins = {}
global_maxes = {}
sections_over_time = deque([], 10)

def visualize_parameters(sess, 
                         limit=None, 
                         save_images=False, 
                         bottom_text='', 
                         headless=False, 
                         display='variance'):  

  def get_parameter_variances():
    sections_over_time.append(sections)
    time_sections_matrix = np.array(sections_over_time) # [time, section, height, width]
    section_variances = np.var(time_sections_matrix, axis=0)

    final_sections = []

    names = [var.name for var in variables]
    for name, variance in zip(names, section_variances):

      if name not in global_maxes:
        global_maxes[name] = variance.max()
        global_mins[name] = variance.min()

      if variance.max() > global_maxes[name]:
        global_maxes[name] = variance.max()

      if variance.min() < global_mins[name]:
        global_mins[name] = variance.min()

      variance = scale(variance, 
                       data_min=global_mins[name] / 5,
                       data_max=global_maxes[name] / 5)

      final_sections.append(variance)

    return final_sections


  def get_resized_parameters(display_width):
    arrays = [np.ravel(x)[:limit] for x in sess.run(variables)]
    display_width = WIDTH // len(arrays)

    reshaped_matrices = []

    for array in arrays:
      element_count = len(array)
      columns = int(np.ceil(np.sqrt((display_width * element_count) / HEIGHT)))
      rows = int(element_count // columns)
      reshaped_matrices.append((np.reshape(array[:rows*columns], (rows, columns))))

    return [cv2.resize(x, 
                       (display_width, HEIGHT), 
                       interpolation=cv2.INTER_NEAREST) 
            for x in reshaped_matrices]
 

  def add_variable_names(image, display_width):
    cv2.rectangle(image, (0, 0), (WIDTH, 30), 0.1, -1)

    for i, variable in enumerate(variables):
      x_offset = (i * display_width) + 5
      cv2.putText(image, 
                  str(variable.get_shape()), 
                  (x_offset, PADDING), 
                  cv2_font, 
                  TEXT_SIZE, 
                  255, 
                  1)


  def add_bottom_text(image):
    cv2.rectangle(image, (0, HEIGHT - 30), (WIDTH, HEIGHT), 0.1, -1)
    cv2.putText(image, bottom_text, (5, HEIGHT - 10), cv2_font, 0.5, 255, 1)

################################################################################

  global global_step
  global_step += 1

  variables = tf.trainable_variables()
  display_width = WIDTH // len(variables)

  if limit is None:
    limit = WIDTH * HEIGHT / len(variables)

  sections = get_resized_parameters(display_width)

  if display == 'variance':
    sections = get_parameter_variances()
  elif display == 'both':
    #stack the two on top of each other for each one, then hstack
    variance_sections = get_parameter_variances()
    sections = [cv2.resize(np.vstack([scale(orig), var]), (display_width, HEIGHT))
                for orig, var in zip(sections, variance_sections)]
  else:
    sections = [scale(section) for section in sections]

  final_image = np.hstack(sections)

  add_variable_names(final_image, display_width)
  add_bottom_text(final_image)

  if save_images:
    cv2.imwrite('gif/'+str(global_step).zfill(5)+'.png', final_image*255)

  if not headless:
    cv2.imshow('Parameters', final_image)
    cv2.waitKey(1)
  
  # def online_variance(data):
  #   global global_mean
  #   global global_step
  #   global global_M2

  #   if global_mean is None:
  #     global_mean = np.zeros(data.shape)
  #     global_M2 = np.zeros(data.shape)

  #   delta = data - global_mean
  #   global_mean += delta / global_step
  #   delta2 = data - global_mean
  #   global_M2 += delta * delta2

  #   if global_step < 2:
  #     return np.zeros(data.shape)
  #   else:
  #     return global_M2 / (global_step - 1)


  # def scale_sections(image, display_width):
  #   starting_point = 0

  #   while starting_point < WIDTH - 10:
  #     section = image[:, starting_point:starting_point+display_width]
  #     section -= global_min
  #     section /= (global_max ** 2.9)
  #     # image[:, starting_point:starting_point+display_width] = scale(section)
  #     starting_point += display_width

  #   return image
  #


  # def split_into_sections(image, display_width):
  #   starting_point = 0
  #   sections = []

  #   while starting_point < WIDTH:
  #     section = image[:, starting_point:starting_point+display_width]
  #     if section.shape[1] != 0:
  #       sections.append(section)
  #     starting_point += display_width

  #   return sections
  #
