from abc import ABC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset

NUM_MODEL_NET = '1_3d_net'
NUM_MODEL_KNN = '4_3d_knn'


class SimpleKNN:
    def __init__(self, num_neighbors, dim):
        self.__dim = dim
        self.__neighbors = num_neighbors
        self.model = KNeighborsClassifier(self.__neighbors)

    def train_model(self, train_set, train_tags):
        self.model = self.model.fit(train_set, train_tags)
        model_file = open(f'model_number_{NUM_MODEL_KNN}.pickle', 'wb')
        pickle.dump(self.model, model_file)
        model_file.close()
        return self.model

    def predict(self, data):
        return self.model.predict(data)


class LinearNeuralNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.__dim = dim
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, 1),
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
        self.__data = data

    def get_data_iterator(self, batch_size=2):
        data_loader = torch.utils.data.DataLoader(MyIterableDataset(self.__data), batch_size=batch_size)
        return data_loader


class MyIterableDataset(torch.utils.data.IterableDataset, ABC):
    def __init__(self, data):
        super(MyIterableDataset).__init__()
        self.dataset = data

    def __iter__(self):
        for frame, label in self.dataset:
            yield frame, label


def train_epoch(model, data_iterator, optimizer, criterion):
    loss = float(0)
    for i, data in enumerate(data_iterator):
        input, label = data
        input = input.float()
        optimizer.zero_grad()
        output = model(input)
        lost_calc = criterion(output.reshape(output.shape[0]), label)
        lost_calc.backward()
        optimizer.step()
        loss += lost_calc.item()
        data_iterator.set_postfix(lost=loss / (i + 1))
    return loss / (i + 1), model


def train_model(model, data_iter, n_epochs, learning_rate, model_to_read=None, start_from_zero=True, weight_decay=0, ):
    if not start_from_zero:
        loaded_file = open(f'model_number_{model_to_read}.pickle', 'rb')
        model = pickle.load(loaded_file)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.L1Loss()
    loss_lst = list()
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(iterable=data_iter.get_data_iterator())
        loss, model = train_epoch(model, pbar, optimizer, loss_func)
        loss_lst.append(loss)
    model_file = open(f'model_number_{NUM_MODEL_NET}.pickle', 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def evaluate(model, data_iterator, criterion):
    lost = list()
    for data in data_iterator:
        input, label = data
        input = input.float()
        output = model(input)
        lost_calc = criterion(output.reshape(output.shape[0]), label)
        lost.append(lost_calc.item())
    return np.mean(np.array(lost))


def evaluate_knn(train_set, train_tags, test_set, test_tags, model, method):
    model.train_model(train_set, train_tags)
    y_hat_train = model.predict(train_set)
    y_hat_test = model.predict(test_set)
    if method == 'accuracy':
        acc_train = np.sum(y_hat_train == train_tags) / len(train_tags)
        acc_test = np.sum(y_hat_test == test_tags) / len(test_tags)
        print(f'train accuracy = {acc_train}, test accuracy =  {acc_test}')


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    round_preds = -torch.ones(preds.shape[0])
    round_y = -torch.ones(y.shape[0])
    round_preds[preds >= 0.6] = 1
    round_y[y >= 0.6] = 1
    round_preds[preds <= 0.4] = 0
    round_y[y <= 0.4] = 0
    accuracy = round_preds == round_y
    return torch.mean(accuracy.float())

def split_data_train_test(data, labels, test_size=0.2):
    train_x, test_x,train_y, test_y = train_test_split(data, labels, test_size=test_size)
    return train_x, train_y, test_x, test_y