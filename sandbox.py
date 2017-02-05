import pandas as pd
from data_prep import nominal_to_numeric

# can you predict the district by other information?



data = pd.read_csv('data/school-grades.csv')
data['district'] = nominal_to_numeric(data['district'])
labels = data['district']
train_data = data.iloc[:, [3, 4]]
print('train_data', train_data)
print('data.columns', data.columns)
