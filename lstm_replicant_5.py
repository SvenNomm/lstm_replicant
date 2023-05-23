import pandas as pd
import os
import datetime
import preprocessing_module as ppm
import matplotlib.pyplot as plt
import pickle as pkl

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import platform
import time
from pathlib import Path

from torcheval.metrics import R2Score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#dev = 'mps'
dev = 'cpu'
katse_nr = 3
order = 2


#path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/'
path = 'C:/Users/Sven/Puu/Data_files/AIRSCS/wave/data_v2/'
winx=256

#path = path + "katse_0" + str(katse_nr) + "/"
path = path + "/katse_0" + str(katse_nr)

initial_data_file_1 = Path(path + '\sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_co_clean.csv')
initial_data_file_2 = path + '\sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_cro_clean.csv'
target_data_file = path + '\wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'

initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
target_data = pd.read_csv(target_data_file, sep=',')

input_data_1_train, input_data_2_train, target_data_train, input_data_1_test, input_data_2_test, target_data_test,\
        test_data_indexes = ppm.time_based_splitter(initial_data_1, initial_data_2, target_data)

input_data_1_train, input_data_2_train, target_data_train, input_data_1_valid, input_data_2_valid, target_data_valid,\
        valid_data_indexes = ppm.time_based_splitter(input_data_1_train, input_data_2_train, target_data_train)

input_data_1_train, input_data_2_train, target_data_train = ppm.initial_formatting_old_data(input_data_1_train, input_data_2_train, target_data_train)
input_data_1_test, input_data_2_test, target_data_test = ppm.initial_formatting_old_data(input_data_1_test,input_data_2_test, target_data_test)
input_data_1_valid, input_data_2_valid, target_data_valid = ppm.initial_formatting_old_data(input_data_1_valid,input_data_2_valid, target_data_valid)

device = torch.device(dev)
#input_tensor_1_train = torch.tensor(input_data_1_train.values).float()
#input_tensor_2_train = torch.tensor(input_data_2_train.values).float()
#target_tensor_train = torch.tensor(target_data_train.values).float()

#input_tensor_1_test = torch.tensor(input_data_1_test.values).float()
#input_tensor_2_test = torch.tensor(input_data_2_test.values).float()
#target_tensor_test = torch.tensor(target_data_test.values).float()

#input_tensor_1_valid = torch.tensor(input_data_1_valid.values).float()
#input_tensor_2_valid = torch.tensor(input_data_2_valid.values).float()
#target_tensor_valid = torch.tensor(target_data_valid.values).float()


def create_dataset(X_train, y_train):
    X, y = [], []
    nx, mx = X_train.shape
    ny, my = y_train.shape
    for i in range(0, nx):
        ax = np.transpose(X_train[i,:]).reshape([mx, 1])
        ay = np.transpose(y_train[i, :]).reshape([my, 1])
        X.append(ax)
        y.append(ay)

    X = np.asarray(X)
    y = np.asarray(y)
    return torch.tensor(X).float(), torch.tensor(y).float()

X_train = input_data_1_train.values
y_train = target_data_train.values
X_test = input_data_1_test.values
y_test = target_data_test.values
X_valid = input_data_1_valid.values
y_valid = target_data_valid.values

X_train, y_train = create_dataset(X_train, y_train)
X_test, y_test = create_dataset(X_test, y_test)
X_valid, y_valid = create_dataset(X_valid, y_valid)

X_train, y_train = X_train.to(dev), y_train.to(dev)
X_test, y_test = X_test.to(dev), y_test.to(dev)
X_valid, y_valid = X_valid.to(dev), y_valid.to(dev)


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=3, batch_first=True)
        #self.norm1 = torch.nn.LayerNorm([1, 16,16,1])
        #self.lstm_1 = nn.LSTM(32, 32, 1)
        #self.lstm_2 = nn.LSTM(8, 8, 1)
        #self.lstm_3 = nn.LSTM(8, 8, 1)
        #self.lstm_4 = nn.LSTM(8, 8, 1)
        self.linear = nn.Linear(128, 1)


    def forward(self, x):
        x, _ = self.lstm(x)
        #x, _ = self.norm1(x)
        #x, _ = self.lstm_1(x)
        #x, _ = self.lstm_2(x)
        #x, _ = self.lstm_3(x)
        #x, _ = self.lstm_4(x)
        x = self.linear(x)
        return x


model = AirModel()
model.to(dev)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.MSELoss()
print(type(X_train), torch.Tensor.size(X_train), type(y_train), torch.Tensor.size(y_train))
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=200)
metric = R2Score()

rmse_train = []
rmse_test = []

n_epochs = 50

for epoch in range(n_epochs):
    start = time.time()
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    #if epoch % 100 != 0:
    #    continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        #train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        train_rmse = torch.sqrt(loss_fn(y_pred, y_train))
        train_rmse = train_rmse.detach().cpu().numpy()
        rmse_train.append(train_rmse)
        y_pred = model(X_test)
        #test_rmse = np.sqrt(loss_fn(y_prd, y_test))
        test_rmse = torch.sqrt(loss_fn(y_pred, y_test))
        test_rmse = test_rmse.detach().cpu().numpy()
        rmse_test.append(test_rmse)

        #metric.update(y_pred, y_train)
        #train_r2 = metric.compute()
        #print(train_r2)

        #metric.update(y_pred, y_test)
        #test_r2 = metric.compute()
        #print(test_r2)

    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    end = time.time()
    print(end - start)