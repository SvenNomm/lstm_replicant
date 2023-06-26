import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error


device = torch.device('mps')


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


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        # self.norm1 = torch.nn.LayerNorm([1, 16,16,1])
        # self.lstm_1 = nn.LSTM(32, 32, 1)
        # self.lstm_2 = nn.LSTM(8, 8, 1)
        # self.lstm_3 = nn.LSTM(8, 8, 1)
        # self.lstm_4 = nn.LSTM(8, 8, 1)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x, _ = self.norm1(x)
        # x, _ = self.lstm_1(x)
        # x, _ = self.lstm_2(x)
        # x, _ = self.lstm_3(x)
        # x, _ = self.lstm_4(x)
        x = self.linear(x)
        return x


def goodness_descriptor(test_y, hat_y):
    mse = mean_squared_error(test_y, hat_y)
    rho = np.corrcoef(test_y, hat_y)[0, 1]
    max_test = test_y.argmax(axis=0)
    max_hat = hat_y.argmax(axis=0)
    delta_max_val = np.max(test_y) - np.max(hat_y)
    delta_max_loc = max_test - max_hat

    return mse, rho, max_test, max_hat, delta_max_val, delta_max_loc