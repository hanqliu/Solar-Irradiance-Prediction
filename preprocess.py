'''
parameters
'''
import pandas as pd
from sklearn import preprocessing
import numpy as np

Station_num = 1

df = pd.read_csv(r'C:\Users\23971\Desktop\数学\水文气象\南开光伏课题数据集及说明\NK2_GF\训练数据\train\Station_' + str(Station_num) + '.csv')
dataset = df.iloc[:, 0:7]

'''
dataset['Time'] = pd.to_datetime(dataset['Time'])
dataset['Hours'] = dataset['Time'].dt.hour
dataset['Mins'] = dataset['Time'].dt.minute
dataset['Time'] = dataset['Hours'] * 4 + dataset['Mins'] / 15
dataset.drop(['Hours', 'Mins'], axis = 1, inplace = True)
'''

dataset.drop(['Time'], axis = 1, inplace = True)

scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))

datas = scaler.fit_transform(dataset)
data_min = scaler.data_min_
data_scale = scaler.scale_

De = 12

def split_y(dataset, width):
    
    data_y = []
    for i in range(len(dataset) - width):
        
        data_y.append(dataset[i:i + width, 0])
        
    return np.array(data_y)

y_series = split_y(dataset.values, De)
