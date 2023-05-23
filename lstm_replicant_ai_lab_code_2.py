import pandas as pd
import pickle as pkl

import torch,ipdb
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import platform
import time
import matplotlib.pyplot as plt
import torchviz
import hiddenlayer as hl
from torcheval.metrics import R2Score
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import load_save_module as lsm
import modeling_module as mm


print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "mps" if getattr(torch,'has_mps',False) \
    else "gpu" if torch.cuda.is_available() else "cpu"
print(f"PyTorch Version: {torch.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")


def create_dataset(X_train, y_train):
    X, y = [], []
    nx, mx = X_train.shape
    ny, my = y_train.shape
    for i in range(0, nx):
        ax = np.transpose(X_train[i, :]).reshape([mx, 1])
        ay = np.transpose(y_train[i, :]).reshape([my, 1])
        X.append(ax)
        y.append(ay)
    return torch.tensor(X).float(), torch.tensor(y).float()


path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/alternative_processing/'
fname_train = 'katse_03_winx_256_ldn_train.pkl'
fname_test = 'katse_03_winx_256_ldn_test.pkl'

initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index, input_data_test, \
    target_data_test, test_data_indexes = lsm.data_loader(path, fname_train, fname_test)

print(input_data_test.shape, target_data_test.shape)
print(initial_data_valid.shape, target_data_valid.shape)
print(initial_data_train.shape, target_data_train.shape)


X_train = initial_data_train
y_train = target_data_train

X_valid = initial_data_valid
y_valid = target_data_valid

X_test = input_data_test
y_test = target_data_test

print(X_test.shape, y_test.shape)
print(X_valid.shape, y_valid.shape)
print(X_train.shape, y_train.shape)

X_train, y_train = create_dataset(X_train, y_train)
X_test, y_test = create_dataset(X_test, y_test)
X_valid, y_valid = create_dataset(X_valid, y_valid)

X_train, y_train = X_train.to('mps'), y_train.to('mps')
X_test, y_test = X_test.to('mps'), y_test.to('mps')
X_valid, y_valid = X_valid.to('mps'), y_valid.to('mps')

model = mm.AirModel()
model.to('mps')
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.MSELoss()
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
    if epoch % 100 != 0:
        continue
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

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(X_valid) * np.nan
    y_pred = model(X_valid)

    #train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    #test_plot = np.ones_like(timeseries) * np.nan
    #test_plot[train_size + lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot

y_pred = y_pred.detach().cpu().numpy()[:,:,0]
X_valid = X_valid.detach().cpu().numpy()[:,:,0]
y_valid = y_valid.detach().cpu().numpy()[:,:,0]

residuals = y_pred - y_valid
n, m = y_pred.shape


fig_1 = plt.figure()
ax_1_1 = fig_1.add_subplot(1, 3, 1)
ax_1_2 = fig_1.add_subplot(1, 3, 2)
ax_1_3 = fig_1.add_subplot(1, 3, 3)

for i in range(0, n):
    ax_1_1.plot(y_valid[i, :], color='blue', linewidth=0.2)
    ax_1_2.plot(y_pred[i, :], color='gold', linewidth=0.2)
    ax_1_3.plot(residuals[i, :], color='darkviolet', linewidth=0.2)

plt.show()

fig_2 = plt.figure()
plt.plot(rmse_test, color = 'green', linewidth = 0.5)
plt.plot(rmse_train, color = 'darkviolet', linewidth = 0.5)
plt.show()

def goodness_descriptor(test_y, hat_y):
    mse = mean_squared_error(test_y, hat_y)
    rho = np.corrcoef(test_y, hat_y)[0, 1]
    max_test = test_y.argmax(axis=0)
    max_hat = hat_y.argmax(axis=0)
    delta_max_val = np.max(test_y) - np.max(hat_y)
    delta_max_loc = max_test - max_hat

    return mse, rho, max_test, max_hat, delta_max_val, delta_max_loc

goodness_descriptors = []

for i in range(0, len(y_pred)):
    mse, rho, max_test, max_hat, delta_max_val, delta_max_loc = goodness_descriptor(y_valid[i, :], y_pred[i, :])
    goodness_descriptors.append([int(test_data_indexes[i]), mse, rho, max_test, max_hat, delta_max_val, delta_max_loc])

goodness_descriptors = np.array(goodness_descriptors)
columns = ['index', 'mse', 'rho', 'max_test', 'max_hat', 'delta_max_val', 'delta_max_loc']
print(columns)
print("Average values are: ", np.average(goodness_descriptors, axis=0))
goodness_descriptors = pd.DataFrame(goodness_descriptors, columns=columns)