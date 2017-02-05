import numpy as np

def keypoints_to_points(keypoints):

  def keypoint_to_point(keypoint):
    return [int(keypoint.pt[0]), int(keypoint.pt[1])]

  if isinstance(keypoints, np.ndarray):
    return keypoints

  return [keypoint_to_point(x) for x in keypoints]
