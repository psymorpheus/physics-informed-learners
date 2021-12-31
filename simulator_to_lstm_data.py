import numpy as np
import torch

np.random.seed(2)

train_data = np.genfromtxt('Data/simulation_10_20/train.csv', delimiter=',')
train_data = np.array(train_data, dtype=np.double)

test_data = np.genfromtxt('Data/simulation_10_20/test.csv', delimiter=',')
test_data = np.array(test_data, dtype=np.double)

vrange = train_data[0, 1:]
trange = train_data[1:, 0]

train_data = train_data[1:, 1:].T   # So that rows of time series data
test_data = test_data[1:, 1:].T

L = trange.shape[0]
N = vrange.shape[0]

# change in displacements data
# for i in range(train_data.shape[1]-1, 0, -1):
#     train_data[:, i] = train_data[:, i] - train_data[:, (i-1)]

# for i in range(test_data.shape[1]-1, 0, -1):
#     test_data[:, i] = test_data[:, i] - test_data[:, (i-1)]

torch.save(train_data, open('LSTM/traindata.pt', 'wb'))
torch.save(test_data, open('LSTM/testdata.pt', 'wb'))
