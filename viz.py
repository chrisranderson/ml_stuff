import plotly
from plotly.graph_objs import Scatter, Histogram

def scatter_plot(ys, filename='scatter.html'):
  plotly.offline.plot([
    Scatter(y=ys, mode='markers')
  ])

def line_plot(ys, filename='scatter.html'):
  plotly.offline.plot([
    Scatter(y=ys)
  ])

def histogram(data):
  plotly.offline.plot([
    Histogram(x=data)
  ])
