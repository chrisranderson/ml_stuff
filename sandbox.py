import pandas as pd

from data_prep import scale_dataframe
from viz import scatter_plot

data = pd.read_csv('data/glass.csv')
data = scale_dataframe(data, ['Type'])
