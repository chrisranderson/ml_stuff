from math import floor

def sliding_window_iterator(image):
  image_width = image.shape[1]
  image_height = image.shape[0]

  scale = 1

  window_height = int(floor(image_height/scale))
  window_width = int(window_height/3)

  while window_width >= 35:

    for x_start in range(int(scale), image_width, 8):
      for y_start in range(int(scale), image_height, 8):
        window = image[y_start:y_start+window_height, x_start:x_start+window_width]

        if window.shape[1] == window_width and window.shape[0] == window_height:
          yield window, x_start, y_start, window_width, window_height

    scale *= 1.3

    window_height = int(floor(image_height/scale))
    window_width = int(window_height/3)


if __name__ == '__main__':
  from finch.datasets import single_image

  test_image = single_image()

  for x in sliding_window_iterator(test_image):
    print('x', x[0].shape)
