from __future__ import division, print_function

from collections import deque

import tensorflow as tf
import numpy as np
import cv2
cv2_font = cv2.FONT_HERSHEY_SIMPLEX

from finch.data_prep import scale
from finch.error_handling import print_error

WIDTH = 1800
HEIGHT = 1000
PADDING = 20
TEXT_SIZE = 0.3

global_step = 0
temporal_section_mins = {}
temporal_section_maxes = {}
sections_over_time = deque([], 5)

def visualize_parameters(sess, 
                         bottom_text='', 
                         display='variance',
                         headless=False, 
                         ignore_patterns=None,
                         parameter_limit=None, 
                         save_images=False, 
                         vars_and_values=None):
  '''
  Visualize the parameters of a neural network.


  ## Parameters

  ### Required
    sess: an instance of tf.Session

  ### Optional
    bottom_text (string): text to add to the bottom of every frame.
    display {'variance', 'both', 'normal'}:
      'variance': shows a rolling variance of each individual parameter.
      'both': stacks 'variance' and 'normal' on top of each other.
      'normal': shows the parameter values.
    headless (boolean): if True, don't show the images. Useful with `save_images`.
    ignore_patterns (list of strings): if one of these strings is in the variable
      name, it will not be displayed.
    parameter_limit (integer > 0): upper bound of how many parameters per variable to show.
    save_images (boolean): whether or not to save each image in a gifs folder.
    vars_and_values (TensorFlow ops, np arrays): two lists of variables - the first list
      is the TF ops, the second list are the corresponding np arrays.

  ### Example usage:
    variables_to_evaluate = [train_step] + tf.trainable_variables() + [predictions]
    evaluated_variables = sess.run(variables_to_evaluate, feed_dict=feed_dict)[1:]

    visualize_parameters(sess, 
                         vars_and_values=(variables_to_evaluate, evaluated_variables),
                         bottom_text='Step: {} || Test accuracy: {}'.format(i, np.mean(test_accuracies)),
                         ignore_patterns=['Batch', 'b:'])
  '''  


  def get_filtered_variables(ignore_patterns):
    ignore_patterns = ignore_patterns if ignore_patterns is not None else []
    variables = tf.trainable_variables()

    for pattern in ignore_patterns:
      variables = [x for x in variables if pattern not in x.name]

    return variables


  def get_parameter_variances():
    sections_over_time.append(sections)
    time_sections_matrix = np.array(sections_over_time) # [time, section, height, width]
    section_variances = np.var(time_sections_matrix, axis=0)

    final_sections = []

    names = [var.name for var in tf_variables]

    for name, variance in zip(names, section_variances):

      if name not in temporal_section_maxes:
        temporal_section_maxes[name] = variance.max()
        temporal_section_mins[name] = variance.min()

      if variance.max() > temporal_section_maxes[name]:
        temporal_section_maxes[name] = variance.max()

      if variance.min() < temporal_section_mins[name]:
        temporal_section_mins[name] = variance.min()

      variance = scale(variance, 
                       data_min=temporal_section_mins[name] / 5,
                       data_max=temporal_section_maxes[name] / 5)

      # band-aid fix for when there is no variance. Hopefully this magic number
      # is low enough... 
      if np.abs(temporal_section_maxes[name] - temporal_section_mins[name]) < 1e-10:
        variance = variance * 0

      final_sections.append(variance)

    return final_sections


  def get_resized_parameters(display_width):
    if variables_already_evaluated:
      arrays = [np.ravel(x) for x in np_arrays]
    else:
      arrays = [np.ravel(x)[:parameter_limit] for x in sess.run(tf_variables)]

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

    for i, shape in enumerate(shapes):
      x_offset = (i * display_width) + 5
      cv2.putText(image, 
                  str(shape), 
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

  if vars_and_values is not None:
    variables_already_evaluated = True
    tf_variables, np_arrays = vars_and_values

    try:
      shapes = [x.shape for x in np_arrays]
    except AttributeError as e:
      print_error('It looks like you have the wrong order for the variables tuple. TensorFlow ops should come first.',
                  e)
  else:
    tf_variables = get_filtered_variables(ignore_patterns)
    variables_already_evaluated = False
    shapes = [x.get_shape() for x in tf_variables]

  display_width = WIDTH // len(tf_variables)

  if parameter_limit is None:
    parameter_limit = int(WIDTH * HEIGHT / len(tf_variables))

  sections = get_resized_parameters(display_width)

  if display == 'variance':
    sections = get_parameter_variances()
  elif display == 'both':
    #stack the two on top of each other for each one, then hstack
    variance_sections = get_parameter_variances()
    sections = [cv2.resize(np.vstack([scale(orig), var]), 
                           (display_width, HEIGHT))
                for orig, var 
                in zip(sections, variance_sections)]
  else:
    sections = [scale(section) for section in sections]

  final_image = np.hstack(sections)

  add_variable_names(final_image, display_width)
  add_bottom_text(final_image)

  if save_images:
    cv2.imwrite('gif/' + str(global_step).zfill(5)+'.png', final_image*255)

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
