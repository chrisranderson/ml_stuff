def batch_generator(data, labels, batch_size=25):

  starting_point = 0

  while starting_point < len(data):
    data_batch = data[starting_point:starting_point+batch_size]
    label_batch = labels[starting_point:starting_point+batch_size]

    starting_point += batch_size

    yield data_batch, label_batch


if __name__ == '__main__':
  pass
