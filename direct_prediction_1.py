# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 21:23:41 2021

@author: Lenovo
"""
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import plot_model

import numpy as np

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences)):
        end_element_index = i + n_steps_in
        out_end_index = end_element_index + n_steps_out - 1
        
        if out_end_index > len(sequences): 
            break
        
        sequence_x, sequence_y = sequences[i:end_element_index,:-1], sequences[end_element_index-1:out_end_index,-1]
        X.append(sequence_x)
        y.append(sequence_y)

    return np.array(X).astype(np.float32), np.array(y).astype(np.float32)

def multi_step_output_model(n_input, n_steps_out, X, y, epochs_num):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_input))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=epochs_num, verbose=0)
    return model

if __name__ == '__main__':
        epochs_num = 500
        data = np.array(pd.read_csv('D:\\e\\南开光伏课题数据集及说明\\NK2_GF\\模拟结果\\Station_1.csv'))
        dataset=data[:,3:8]
        outcome=data[:,2]
        dataset=np.insert(dataset,5,outcome,axis=1)
        test= np.array(pd.read_csv('D:\\e\\南开光伏课题数据集及说明\\NK2_GF\\评测数据\\气象数据\\Station_1.csv'))
        testset=test[:,2:7]
        n_steps_in, n_steps_out = 8, 1
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        n_input = X.shape[1] * X.shape[2]
        X = X.reshape((X.shape[0], n_input))
    
        print(X.shape, y.shape)
        model = multi_step_output_model(n_input, n_steps_out, X, y, epochs_num)
        plot_model(model ,show_shapes=True,to_file=r'C:\Users\Lenovo\Desktop\气象水文\A1model.png')

