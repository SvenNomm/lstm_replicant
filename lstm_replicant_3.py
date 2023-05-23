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
import onnx
import torch.onnx

from dataset import StudentsPerformanceDataset
dataset = StudentsPerformanceDataset('exams')


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


path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/processed_datasets/'

with open(path + 'katse_3_4_training_data_04_19_2023_15_29_28.pkl', 'rb') as f:
    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = pkl.load(f)

with open(path + 'katse_3_4_testing_data_04_19_2023_15_29_28.pkl', 'rb') as f:
    input_data_test, target_data_test, test_data_indexes = pkl.load(f)

X_train = initial_data_train.values
y_train = target_data_train.values

X_test = initial_data_valid.values
y_test = target_data_valid.values

X_valid = input_data_test.values
y_valid = target_data_test.values



#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('mps')
def create_dataset(X_train, y_train):
    X, y = [], []
    nx, mx = X_train.shape
    ny, my = y_train.shape
    for i in range(len(X_train)):
        ax = np.transpose(X_train[i,:]).reshape([mx, 1])
        ay = np.transpose(y_train[i, :]).reshape([my, 1])
        X.append(ax)
        y.append(ay)
    return torch.tensor(X).float(), torch.tensor(y).float()

X_train, y_train = create_dataset(X_train, y_train)
X_test, y_test = create_dataset(X_test, y_test)
X_valid, y_valid = create_dataset(X_valid, y_valid)

#X_train, y_train = X_train.to('mps'), y_train.to('mps')
#X_test, y_test = X_test.to('mps'), y_test.to('mps')


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=5, batch_first=True)
        #self.lstm_1 = nn.LSTM(64, 64, 1)
        #self.lstm_2 = nn.LSTM(64, 64, 1)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        #x, _ = self.lstm_1(x)
        #x, _ = self.lstm_2(x)
        x = self.linear(x)
        return x


model = AirModel()
#torch.save(model, 'model_1.pt')
torch.onnx.export(model, X_train, 'model_3.onnx', export_params=False)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=200)
metric = R2Score()

rmse_train = []
rmse_test = []


n_epochs = 10
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
        #continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        rmse_train.append(train_rmse)
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
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
#torch.save(model, 'model_1.pt')

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




