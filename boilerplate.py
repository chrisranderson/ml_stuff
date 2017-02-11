#! /usr/bin/python

import os

def make_directory(name):
  if not os.path.exists(name):
    os.makedirs(name)

def make_file(name):
  if not os.path.exists(name):
    file = open(name, 'w')
    file.close()

def setup():
  make_directory('cache')
  make_directory('data')
  make_directory('plots')
  make_directory('output')
  make_file('main.py')
  make_file('notes.md')

if __name__ == '__main__':
  setup()
