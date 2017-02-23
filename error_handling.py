from __future__ import print_function

from textwrap import fill

def print_error(message, exception):
  print('\n'+'='*80)
  print(fill(message, 80))
  print('='*80)
  print()
  raise(exception)
