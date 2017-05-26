import os

import cv2
import numpy as np

from finch.data_prep import batch_generator
from finch.dsp import grayscale, get_neighbors
from finch.graphics import pad
from finch.cv2_util import keypoints_to_points
from finch.vgg.vgg16 import vgg16

cv2.ocl.setUseOpenCL(False)

ORB_DESCRIPTOR_SIZE = 5



THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

def _xy_grid(image):
  crop_size = ORB_DESCRIPTOR_SIZE*3+1
  cropped = image[crop_size:-crop_size, 
                  crop_size:-crop_size]
  return np.roll(np.vstack(np.where(np.sum(cropped, axis=2) != 0.131231312)).T, 1, axis=1) + crop_size

def dense_orb_features(image):
  keypoints = _xy_grid(image)
  return orb_features(image, keypoints)

def orb_features(image, keypoints):
  if isinstance(keypoints, np.ndarray):
    # takes in x, y coordinates. size is the diameter of the descripted area
    keypoints = [cv2.KeyPoint(p[0], p[1], ORB_DESCRIPTOR_SIZE) for p in keypoints]

  orb = cv2.ORB_create()
  new_keypoints, descriptors = orb.compute(np.mean(image, axis=2).astype(np.uint8), keypoints)

  print('len(keypoints)', len(keypoints))
  print('len(new_keypoints)', len(new_keypoints))
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


def CNNDescriptor(image_size=224, return_tf=False):
  print('Initializing CNN...')
  import tensorflow as tf
  sess = tf.Session()

  image_placeholder = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])
  vgg_network = vgg16(image_placeholder)
  final_pool = vgg_network.convlayers()

  sess.run(tf.initialize_all_variables())

  def load_weights():
    vgg_network.load_weights( '{}/vgg/vgg16_weights.npz'.format(THIS_DIRECTORY), sess)

  def get_features(patches, batch_size=100, embedding_size=9999999):
    print('Getting features...')
    results = []

    for start, end in batch_generator(patches, batch_size=batch_size):
      print('len(results)/len(patches)', len(results)/len(patches))
      embeddings = sess.run(final_pool, feed_dict={
        image_placeholder: patches[start:end]
      })

      reshaped = embeddings.reshape((-1, np.prod(embeddings.shape[1:])))[:, :embedding_size]
      results += list(reshaped)



    return results


  if return_tf:
    return sess, image_placeholder, final_pool, load_weights
  else:
    return get_features



if __name__ == '__main__':
  pass
