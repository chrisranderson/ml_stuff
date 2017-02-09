import tensorflow as tf
import numpy as np
import cv2

from finch.nn import fully_connected_layer
from finch.data_prep import batch_generator

BATCH_SIZE = 10
EPOCH_LIMIT = 10

LEARNING_RATE = 0.001

# graph #######################################################################

sess = tf.Session()

input_placeholder = tf.placeholder(tf.float32, shape=...)
labels_placeholder = tf.placeholder(tf.float32, shape=...)
learning_rate_placeholder = tf.placeholder(tf.float32)


cost_function = ...
train_step = tf.train.AdamOptimizer(learning_rate_placeholder).minimize(cost_function)

sess.run(tf.initialize_all_variables())


# train #######################################################################

data =
labels =

for epoch_number in range(EPOCH_LIMIT):
  for batch, batch_labels in batch_generator(data, labels, batch_size=BATCH_SIZE):
    loss, _ = sess.run([cost_function, train_step], feed_dict={
      input_placeholder: batch,
      labels_placeholder: batch_labels,
      learning_rate_placeholder: LEARNING_RATE
    })
