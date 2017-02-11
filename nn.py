import numpy as np
import tensorflow as tf

''' Notes
For TensorBoard: writer = tf.train.SummaryWriter('./logs', graph=tf.get_default_graph())
'''

def fully_connected_layer(inputs, output_size, scope, nonlinearity=True):
  xavier = tf.contrib.layers.xavier_initializer

  with tf.variable_scope(scope):
    input_size = inputs.get_shape()[1].value

    weights = tf.get_variable('w', 
                              [input_size, output_size],
                              initializer=xavier())

    bias = tf.get_variable('b', 
                           [output_size],
                           initializer=tf.constant_initializer(0.0))

    multiplied = tf.matmul(inputs, weights)

    return tf.nn.elu(multiplied + bias) if nonlinearity else multiplied + bias

def conv2d(x, shape, scope, stride=None):
  xavier = tf.contrib.layers.xavier_initializer
  stride = [1, 1, 1, 1] if stride is None else stride

  with tf.variable_scope(scope):
    W = tf.get_variable('conv-w', shape, initializer=xavier())
    return tf.nn.conv2d(x, W, strides=stride, padding='VALID')

def flatten_final_conv_layer(batch):
  '''
  For converting the final conv layer for use in a FC layer.
  '''
  batch_shape = batch.get_shape().as_list()[1:]
  size = np.prod(batch_shape)
  return tf.reshape(batch, [-1, size])
