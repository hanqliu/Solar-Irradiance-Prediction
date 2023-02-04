# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 23:15:23 2022

@author: Liu kp
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

N_ITER = 5000
BATCH_SIZE = 64
DATA_PATH = 'Station1.xlsx'
NUM_CLASSES = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
SEQ_LENGTH = 5  # time step
INPUT_SIZE = 5 # input size
mm_x = MinMaxScaler()
mm_y = MinMaxScaler()

# 读取数据
def read_data(data_path):
    data = pd.read_excel(data_path)
    feature = data
    label = data.iloc[:, [1]]
    return feature, label


# 标准化数据
def normalization(x, y):
    # print(x.values)
    x = mm_x.fit_transform(x.values)
    y = mm_y.fit_transform(y)
    return x, y


# 建立滑动窗口
def sliding_windows(data,irra):
    x = []
    y = []
    for i in range(len(data) - SEQ_LENGTH - 1):
        _x = data[i:i + SEQ_LENGTH, :]
        _y = irra[i + SEQ_LENGTH, -1]
        x.append(_x)
        y.append(_y)
    x = np.array(x)
    y = np.array(y)
    return x, y


# 建立DataLoader
def data_generator(x_train, y_train, x_test, y_test):
    train_dataset = TensorDataset(torch.from_numpy(x_train).to(torch.float32),
                                  torch.from_numpy(y_train).to(torch.float32))
    test_dataset = TensorDataset(torch.from_numpy(x_test).to(torch.float32), torch.from_numpy(y_test).to(torch.float32))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
    test_Loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
    return train_loader, test_Loader


feature, label = read_data(DATA_PATH)
feature=feature.iloc[:,2:7]
feature, label = normalization(feature, label)

x, y = sliding_windows(feature,label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

train_loader, test_loader = data_generator(x_train, y_train, x_test, y_test)


# 建立 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = SEQ_LENGTH

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # h_0 = torch.zeros(
        #     self.num_layers,
        #     BATCH_SIZE, self.hidden_size
        # )
        # c_0 = torch.zeros(
        #     self.num_layers, BATCH_SIZE, self.hidden_size
        # )
        output, (h_n, c_n) = self.lstm(x, None)
        h_out = output[:, -1, :]
        # h_n.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


model = LSTM(num_classes=NUM_CLASSES, input_size=INPUT_SIZE,
             hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练模型
def train():
    iter = 0
    for epoch in range(NUM_EPOCHS):
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_y = Variable(torch.reshape(batch_y, (len(batch_y), 1)))
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1
            if iter % 100 == 0:
                print("iter: %d,    loss: %1.5f" % (iter, loss.item()))


def eval(test_x, test_y):
    model.eval()
    test_x = Variable(torch.from_numpy(test_x).to(torch.float32))
    test_y = Variable(torch.from_numpy(test_y).to(torch.float32))
    train_predict = model(test_x)
    data_predict = train_predict.data.numpy()
    y_data_plot = test_y.data.numpy()
    y_data_plot = np.reshape(y_data_plot, (-1, 1))
    data_predict = mm_y.inverse_transform(data_predict)
    y_data_plot = mm_y.inverse_transform(y_data_plot)

    plt.figure(figsize=(30,10),dpi=200)
    plt.plot(y_data_plot)
    plt.plot(data_predict)
    plt.legend(('real', 'predict'), fontsize='15')
    plt.show()

    print('MAE/RMSE')
    print(mean_absolute_error(y_data_plot, data_predict))
    print(np.sqrt(mean_squared_error(y_data_plot, data_predict)))
    print(y_data_plot.flatten()[:20])
    print(data_predict.flatten()[:20])
    err=[]
    for i in range(len(data_predict)):
        err.append(abs(data_predict[i]-y_data_plot[i]))
    return err

train()
sadasd=eval(x_test, y_test)
sadasd.mean()
afagps=[]
for i in range(len(sadasd)):
    afagps.append(sadasd[i][0])
np.array(afagps).mean()