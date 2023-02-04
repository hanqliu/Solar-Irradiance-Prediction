import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import preprocess
import time

time_start = time.time()

Station_num = preprocess.Station_num

dataset = preprocess.datas
x = preprocess.datas[:, 1:]
x_min = preprocess.data_min[1:]
x_scale = preprocess.data_scale[1:]

H_size = 16
epochs = 50
train_size = 2000
pred_size = 96

txt_name = 'Staion_{} Train_size-{} Epoch-{}.txt'.format(Station_num, train_size, epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = dataset[:, 1:]

class LSTM(nn.Module):
    
    def __init__(self, ):
        
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=H_size,
            num_layers=1,
            batch_first=True
            )
        
        self.linear = nn.Linear(H_size, 5)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        
        out, _ = self.lstm(x)
        out = self.linear(out.view(-1, H_size))
        out = self.activation(out)
        
        return out[-1]
    
model = LSTM().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)

model.train()

for epoch in range(epochs):
    
    for step in range(0, train_size, 10):
    
        x = torch.tensor(dataset[:step + 1]).float().unsqueeze(0).to(device)
        y = torch.tensor(dataset[step + 1]).float().to(device)
        
        optimizer.zero_grad()
        output = model(x)
        
        
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        
        if step % 100 ==0:
            print('Epoch: {} Step: {}'.format(epoch, step))
            print(loss.cpu().data.numpy())
            print('Prediction: {}'.format(output.cpu().data.numpy()))
            print('y: {}'.format(y.cpu().data.numpy()))
        

   
model.eval()

pred_series = dataset[:train_size + 1, :]
    
for step in range(pred_size):
    
    seq = torch.FloatTensor(pred_series).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(seq)
        pred_series = np.vstack((pred_series, prediction.cpu().data.numpy()))
        loss = loss_func(prediction, torch.FloatTensor(dataset[train_size + 1 + step]).to(device))
        
        with open(txt_name, 'a') as f:
            print('Step: {} Loss: {}'.format(step, loss.cpu().data.numpy()))
            f.write('Step: {} Loss: {}'.format(step, loss.cpu().data.numpy()))
        
plt.figure()
for i in range(5):
    
    plt.subplot(5, 1, i+1)
    plt.plot(pred_series[-pred_size:, i])
    plt.plot(dataset[train_size + 1 : train_size + pred_size + 1].reshape(pred_size, -1)[:, i])

plt.show()
pd.DataFrame(pred_series).to_csv('D:\\e\\南开光伏课题数据集及说明\\NK2_GF\\评测结果(无插值)\\5S_tation_1.csv')


time_end = time.time()

Time = time_end - time_start
print(Time)