import pandas as pd
import os
import datetime
import preprocessing_module as ppm
import modeling_module as mm
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

dev = 'mps'
# dev = 'cpu'

# def create_dataset_ng(**kwargs):
#     kwargs_keys = kwargs.keys()
#     results = []
#
#     for key in kwargs_keys:
#         output = []
#         arr = kwargs[key]
#         n, m = kwargs[key].shape
#         for i in range(0, n):
#             output.append(np.transpose(arr[i, :]).reshape([m, 1]))
#
#         output = np.asarray(output)
#         output = torch.tensor(output).float()
#         results.append(output)
#
#     return results


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=2, num_layers=1, batch_first=True)
        # self.norm1 = torch.nn.LayerNorm([1, 16,16,1])
        # self.lstm_1 = nn.LSTM(32, 32, 1)
        # self.lstm_2 = nn.LSTM(8, 8, 1)
        # self.lstm_3 = nn.LSTM(8, 8, 1)
        # self.lstm_4 = nn.LSTM(8, 8, 1)
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x, _ = self.norm1(x)
        # x, _ = self.lstm_1(x)
        # x, _ = self.lstm_2(x)
        # x, _ = self.lstm_3(x)
        # x, _ = self.lstm_4(x)
        x = self.linear(x)
        return x



katse_nr = 4
order = 2

path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/'
# path = 'C:/Users/Sven/Puu/Data_files/AIRSCS/wave/data_v2/'
winx = 512

path = path + "katse_0" + str(katse_nr) + "/"
# path = path + "/katse_0" + str(katse_nr)

# initial_data_file_1 = Path(path + '\sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_co_clean.csv')
# initial_data_file_2 = path + '\sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_cro_clean.csv'
# target_data_file = path + '\wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'

initial_data_file_1 = path + '/sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_co_clean.csv'
initial_data_file_2 = path + '/sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_cro_clean.csv'
target_data_file = path + '/wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'

initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
target_data = pd.read_csv(target_data_file, sep=',')

input_data_1_train, input_data_2_train, target_data_train, input_data_1_test, input_data_2_test, target_data_test, \
    test_data_indexes = ppm.time_based_splitter(initial_data_1, initial_data_2, target_data)

input_data_1_train, input_data_2_train, target_data_train, input_data_1_valid, input_data_2_valid, target_data_valid, \
    valid_data_indexes = ppm.time_based_splitter(input_data_1_train, input_data_2_train, target_data_train)

input_data_1_train, input_data_2_train, target_data_train = ppm.initial_formatting_old_data(input_data_1_train,
                                                                                            input_data_2_train,
                                                                                            target_data_train)
input_data_1_test, input_data_2_test, target_data_test = ppm.initial_formatting_old_data(input_data_1_test,
                                                                                         input_data_2_test,
                                                                                         target_data_test)
input_data_1_valid, input_data_2_valid, target_data_valid = ppm.initial_formatting_old_data(input_data_1_valid,
                                                                                            input_data_2_valid,
                                                                                            target_data_valid)

device = torch.device(dev)

X_train_1 = input_data_1_train.values
X_train_2 = input_data_2_train.values
y_train = target_data_train.values
X_test_1 = input_data_1_test.values
X_test_2 = input_data_2_test.values
y_test = target_data_test.values
X_valid_1 = input_data_1_valid.values
X_valid_2 = input_data_2_valid.values
y_valid = target_data_valid.values

[X_train_1, X_train_2, y_train, X_valid_1, X_valid_2, y_valid, X_test_1, X_test_2, y_test] = \
    ppm.create_dataset_ng(X_train_1=X_train_1, X_train_2=X_train_2, y_train=y_train,
                      X_valid_1=X_valid_1, X_valid_2=X_valid_2, y_valid=y_valid,
                      X_test_1=X_test_1, X_test_2=X_test_2, y_test=y_test)

X_train_1, X_train_2, y_train = X_train_1.to(dev), X_train_2.to(dev), y_train.to(dev)
X_valid_1, X_valid_2, y_valid = X_valid_1.to(dev), X_valid_2.to(dev), y_valid.to(dev)
X_test_1, X_test_2, y_test = X_test_1.to(dev), X_test_2.to(dev), y_test.to(dev)


X_train, y_train = ppm.combine_polarisations_div_tensor(X_train_1, X_train_2, y_train, dev)
X_valid, y_valid = ppm.combine_polarisations_div_tensor(X_valid_1, X_valid_2, y_valid, dev)
X_test, y_test = ppm.combine_polarisations_div_tensor(X_test_1, X_test_2, y_test, dev)

# n = len(X_test_plot_tr)
# fig_1 = plt.figure()
# ax_1_1 = fig_1.add_subplot(1, 4, 1)
# ax_1_2 = fig_1.add_subplot(1, 4, 2)
# ax_1_3 = fig_1.add_subplot(1, 4, 3)
# ax_1_4 = fig_1.add_subplot(1, 4, 4)

# for i in range(0, n):
#    if i % 100 != 0:
#        ax_1_1.plot(X_test_plot[i, :], color='blue', linewidth=0.2)
#        ax_1_2.plot(y_test_plot[i, :], color='gold', linewidth=0.2)
#        ax_1_3.plot(X_test_plot_tr[i, :], color='blue', linewidth=0.2)
#        ax_1_4.plot(y_test_plot_tr[i, :], color='gold', linewidth=0.2)
# plt.show()

model = AirModel()
model.to(dev)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.MSELoss()
print(type(X_train), torch.Tensor.size(X_train), type(y_train), torch.Tensor.size(y_train))
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=200)
metric = R2Score()

rmse_train = []
rmse_test = []

n_epochs = 1000

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
    # if epoch % 100 != 0:
    #    continue
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)
        # train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        train_rmse = torch.sqrt(loss_fn(y_pred_train, y_train))
        train_rmse = train_rmse.detach().cpu().numpy()
        rmse_train.append(train_rmse)
        y_pred_valid = model(X_test)
        # test_rmse = np.sqrt(loss_fn(y_prd, y_test))
        test_rmse = torch.sqrt(loss_fn(y_pred_valid, y_test))
        test_rmse = test_rmse.detach().cpu().numpy()
        rmse_test.append(test_rmse)

        # metric.update(y_pred, y_train)
        # train_r2 = metric.compute()
        # print(train_r2)

        # metric.update(y_pred, y_test)
        # test_r2 = metric.compute()
        # print(test_r2)

    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    end = time.time()
    print(end - start)

y_pred_train = y_pred_train.detach().cpu().numpy()[:, :, 0]
y_train = y_train.detach().cpu().numpy()[:, :, 0]

y_pred_valid = y_pred_valid.detach().cpu().numpy()[:, :, 0]
y_valid = y_test.detach().cpu().numpy()[:, :, 0]

y_ampl = np.amax(y_valid) - np.amin(y_valid)
residuals = np.sqrt((y_pred_valid - y_valid) ** 2)
residuals_norm = residuals / np.amax(y_valid)
n, m = y_pred_valid.shape

print(np.sqrt(((y_pred_valid - y_valid) ** 2).mean()))

fig_1 = plt.figure()
ax_1_1 = fig_1.add_subplot(1, 4, 1)
ax_1_2 = fig_1.add_subplot(1, 4, 2)
ax_1_3 = fig_1.add_subplot(1, 4, 3)
ax_1_4 = fig_1.add_subplot(1, 4, 4)

for i in range(0, n):
    ax_1_1.plot(y_valid[i, :], color='blue', linewidth=0.2)
    ax_1_2.plot(y_pred_valid[i, :], color='gold', linewidth=0.2)
    ax_1_3.plot(residuals[i, :], color='darkviolet', linewidth=0.2)
    ax_1_4.plot(residuals_norm[i, :], color='green', linewidth=0.2)

plt.show()

fig_2 = plt.figure()
plt.plot(rmse_test, color='green', linewidth=0.5)
plt.plot(rmse_train, color='darkviolet', linewidth=0.5)
plt.show()

fig_3 = plt.figure()
for i in range(0, n):
    plt.plot(residuals_norm[i, :], color='darkviolet', linewidth=0.2)

plt.show()


goodness_descriptors = []

for i in range(0, len(y_test)):
    mse, rho, max_test, max_hat, delta_max_val, delta_max_loc = mm.goodness_descriptor(y_valid[i, :], y_pred_valid[i, :])
    goodness_descriptors.append([int(test_data_indexes[i]), mse, rho, max_test, max_hat, delta_max_val, delta_max_loc])

goodness_descriptors = np.array(goodness_descriptors)
columns = ['index', 'mse', 'rho', 'max_test', 'max_hat', 'delta_max_val', 'delta_max_loc']
print(columns)
print("Average values are: ", np.average(goodness_descriptors, axis=0))
goodness_descriptors = pd.DataFrame(goodness_descriptors, columns=columns)
goodness_descriptors.to_csv('Output.csv', index = False)
# input_tensor_1_train = torch.tensor(input_data_1_train.values).float()
# input_tensor_2_train = torch.tensor(input_data_2_train.values).float()
# target_tensor_train = torch.tensor(target_data_train.values).float()

# input_tensor_1_test = torch.tensor(input_data_1_test.values).float()
# input_tensor_2_test = torch.tensor(input_data_2_test.values).float()
# target_tensor_test = torch.tensor(target_data_test.values).float()

# input_tensor_1_valid = torch.tensor(input_data_1_valid.values).float()
# input_tensor_2_valid = torch.tensor(input_data_2_valid.values).float()
# target_tensor_valid = torch.tensor(target_data_valid.values).float()


# def create_dataset(X_train, y_train):
#     X, y = [], []
#     nx, mx = X_train.shape
#     ny, my = y_train.shape
#     for i in range(0, nx):
#         ax = np.transpose(X_train[i, :]).reshape([mx, 1])
#         ay = np.transpose(y_train[i, :]).reshape([my, 1])
#         X.append(ax)
#         y.append(ay)
#
#     X = np.asarray(X)
#     y = np.asarray(y)
#     return torch.tensor(X).float(), torch.tensor(y).float()

# X_test_plot = X_test.detach().cpu().numpy()
# y_test_plot = y_test.detach().cpu().numpy()

# X_train = F.normalize(torch.log(X_train))
# y_train = F.normalize(torch.log(y_train))

# X_valid = F.normalize(torch.log(X_valid))
# y_valid = F.normalize(torch.log(y_valid))

# X_test = F.normalize(torch.log(X_test))
# y_test = F.normalize(torch.log(y_test))

# X_test_plot_tr = X_test.detach().cpu().numpy()
# y_test_plot_tr = y_test.detach().cpu().numpy()
