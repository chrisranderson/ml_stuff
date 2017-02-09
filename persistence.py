from sklearn.externals import joblib
import os

CACHE_FOLDER = 'pickles'

def make_cache_folder():
  if not os.path.exists(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

def save_pickle(data, name):
  joblib.dump(data, CACHE_FOLDER + '/' + name + '.pkl')

def load_pickle(name):
  return joblib.load(CACHE_FOLDER + '/' + name + '.pkl')

if __name__ == '__main__':
  save_pickle('data', 'data')
  print('load_pickle()', load_pickle('data'))
