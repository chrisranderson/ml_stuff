import numpy as np
import plotly
from plotly.graph_objs import Scatter, Histogram, Heatmap, Layout, Figure, Marker

from datasets import random_set 

COLORS = [ # maximally distinct colors
    '#FFB300', # Vivid Yellow
    '#803E75', # Strong Purple
    '#FF6800', # Vivid Orange
    '#A6BDD7', # Very Light Blue
    '#C10020', # Vivid Red
    '#CEA262', # Grayish Yellow
    '#817066', # Medium Gray
    # The following don't work well for people with color blindness
    '#007D34', # Vivid Green
    '#F6768E', # Strong Purplish Pink
    '#00538A', # Strong Blue
    '#FF7A5C', # Strong Yellowish Pink
    '#53377A', # Strong Violet
    '#FF8E00', # Vivid Orange Yellow
    '#B32851', # Strong Purplish Red
    '#F4C800', # Vivid Greenish Yellow
    '#7F180D', # Strong Reddish Brown
    '#93AA00', # Vivid Yellowish Green
    '#593315', # Deep Yellowish Brown
    '#F13A13', # Vivid Reddish Orange
    '#232C16', # Dark Olive Green
]

def scatter_plot(xs, ys=None, xlabel='', ylabel='', title='', lines=False):
  layout = Layout(
    title=title,
    xaxis=dict(title=xlabel),
    yaxis=dict(title=ylabel)
  )

  if ys is None:
    ys = xs
    xs = range(len(xs))

  data = [
    Scatter(x=xs, 
      y=ys, 
      mode='lines' if lines else 'markers'
    )
  ]

  figure = Figure(data=data, layout=layout)
  plotly.offline.plot(figure, filename='plots/' + title + '.html')

def histogram(data, filename):
  plotly.offline.plot([
    Histogram(
      x=data,
    )
  ], filename=filename)

def heatmap(data, x=None, y=None):
  plotly.offline.plot([
    Heatmap(
      x=None, 
      y=None,
      z=data,
      colorscale='Viridis'
    )
  ])

def exploratory_plot(data, filename='exploratory', use_labels=True):

  with_labels = (data)[:600, :]
  labels = np.unique(with_labels[:, -1])
  data = data[:, :-1]

  non_normalized = np.copy(data)

  traces = []
  column_count = data.shape[1]
  figure = plotly.tools.make_subplots(rows=column_count, cols=column_count,
    shared_yaxes=True, shared_xaxes=True)
  
  # data = scale_down(data)
  # with_labels = scale_down(with_labels)

  for first_column in range(column_count):
    for second_column in range(column_count):
      if use_labels:
        for label_index, label in enumerate(labels):
          label_data = with_labels[with_labels[:, -1] == label]
          
          figure.append_trace(Scatter({
            'x': label_data[:, first_column],
            'y': label_data[:, second_column],
            'mode': 'markers',
            'marker': Marker(
              size=3,
              color=COLORS[label_index]
            ),
            'showlegend': False
          }), first_column + 1, second_column + 1)
      else:
        figure.append_trace(Scatter({
          'x': data[:, first_column],
          'y': data[:, second_column],
          'mode': 'markers',
          'marker': Marker(
            size=2,
            color=COLORS[first_column % len(COLORS)]
          ),
          'showlegend': False
        }), first_column + 1, second_column + 1)

  figure['layout'].update(height=1000, width=1000, title='megachart of awesome')
  plotly.offline.plot(figure, filename=filename+'.html')

  traces = []

  for x in range(column_count):
    column = non_normalized[:, x]

    traces.append(Scatter({
      'x': column,
      'y': np.zeros(len(column)) + x + 1,
      'mode': 'markers',
      'marker': Marker(
        size=3
      )
    }))

  plotly.offline.plot({
    'data': traces
  }, filename = 'exploratory-range.html')
