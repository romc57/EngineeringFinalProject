from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import sys

DIM = 50
NUM_MODEL = 0


class SimpleKNN:
    def __int__(self, num_neighbors):
        self._neighbors = num_neighbors
        self.model = KNeighborsClassifier(self._neighbors)

    def train_model(self, train_set, train_tags):
        return self.model.fit(train_set, train_tags)

    def predict(self, data):
        return self.model.predict(data)


class LinearNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(DIM, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        return self(x)


class DataManager:
    def __init__(self, tensors, labels):
        data = list()
        for i in range(len(tensors)):
            data.append((tensors[i], labels[i]))
        self.__data = torch.tensor(data)

    def get_data_iterator(self):
        return self.__data


def train_epoch(model, data_iterator, optimizer, criterion):
    loss = float(0)
    for i, data in enumerate(data_iterator):

        input, label = data
        optimizer.zero_grad()
        output = model(input)
        lost_calc = criterion(output.reshape(output.shape[0]), label)
        lost_calc.backward()
        optimizer.step()
        loss += lost_calc.item()
        data_iterator.set_postfix(lost=loss / (i + 1))
    return loss / (i + 1), model


def train_model(model, data_manager, n_epochs, learning_rate,model_to_read = None, start_from_zero = False, weight_decay=0, ):
    if not start_from_zero:
        loaded_file = open(f'model_number_{model_to_read}.pickle', 'rb')
        model = pickle.load(loaded_file)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.L1Loss()
    loss_lst = list()
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(iterable=data_manager.get_data_iterator())
        loss, model = train_epoch(model, pbar, optimizer, loss_func)
        loss_lst.append(loss)
    model_file = open(f'model_number_{NUM_MODEL}.pickle', 'wb')
    pickle.dump(model, model_file)
    model_file.close()



def evaluate(model, data_iterator, criterion):
    lost = list()
    for data in data_iterator:
        input, label = data
        output = model(input)
        lost_calc = criterion(output.reshape(output.shape[0]), label)
        lost.append(lost_calc.item())
    return np.mean(np.array(lost))


