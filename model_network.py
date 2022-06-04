'''
@authors: Rom Cohen, Royi Schossberger.
@brief: The next py file is implementation of our learning models part final engineering project.
'''

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
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.autograd import Variable

# Constants:
NUM_MODEL_NET = '0_3d_net'
NUM_MODEL_KNN = 'multi_0_3d_knn'


# -----------------------classes---------------------------------


class SimpleKNN:
    """
    The next class is a wrapper class for knn model of sklearn.
    """

    def __init__(self, num_neighbors, dim):
        """
        Constructor.
        :param num_neighbors: The number of neighbors the knn uses for prediction.
        :param dim: what is the dimension of the data.
        """
        self.__dim = dim
        self.__neighbors = num_neighbors
        self.model = KNeighborsClassifier(self.__neighbors)

    def train_model(self, train_set, train_tags):
        """
        Train method - preform training using fit function of sklearn knn and save the model.
        :param train_set: train set data.
        :param train_tags: train labels of the same data.
        :return: trained knn model.
        """
        self.model = self.model.fit(train_set, train_tags)
        model_file = open(f'model_number_{NUM_MODEL_KNN}.pickle', 'wb')
        pickle.dump(self, model_file)
        model_file.close()
        return self.model

    def predict(self, data):
        """
        predict method return the predicted in a shape consistent with the input.
        :param data: data to predict labels on.
        :return:predictions.
        """
        return self.model.predict(data)

    def get_dim(self):
        """
        A getter method for __dim attribute.
        :return: dim integer.
        """
        return self.__dim


class LinearNeuralNet(nn.Module):
    """
    The next class in implementation of a linear neural network.
    """

    def __init__(self, dim, out_dim):
        """
        Model constructor.
        :param dim: flatten dim of the data.
        """
        super().__init__()
        self.__dim = dim
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Feed forward method of the model
        :param x: data point.
        :return: the model layers output for input x. before activation.
        """
        return self.layers(x)

    def predict(self, x):
        """
        Predict method on the model.
        :param x: data point.
        :return: the model output of x.
        """
        return self(x)

    def get_dim(self):
        """
        A getter for dim attribute.
        :return: dim.
        """
        return self.__dim


class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.hid1 = nn.Linear(dim, 50)  # 6-(10-10)-3
        self.hid2 = nn.Linear(50, 30)
        self.oupt = nn.Linear(30, 5)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        z = torch.tanh(self.hid2(z))
        z = self.oupt(z)  # no softmax: CrossEntropyLoss()
        return z

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_of_classes=2):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size_2, num_of_classes)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size_2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        flatten = torch.flatten(x)
        hidden = self.fc1(flatten)
        relu = self.relu(hidden)
        hidden_2 = self.fc2(relu)
        relu_2 = self.relu(hidden_2)
        output = self.fc3(relu_2)
        output = self.sigmoid(output)
        return output

    def predict(self, x):
        self(x)


class DataManager:
    """
    A data handler class.
    """

    def __init__(self, tensors, labels):
        """
        Constructor.
        :param tensors: data points.
        :param labels: labels consist with data points.
        """
        data = list()
        for i in range(len(tensors)):
            data.append((tensors[i], labels[i]))
        self.__data = data

    def get_data_iterator(self, batch_size=5):
        """
        A getter for data iterator.
        :param batch_size:
        :return: iterator.
        """
        data_loader = torch.utils.data.DataLoader(MyIterableDataset(self.__data), batch_size=batch_size)
        return data_loader


class MyIterableDataset(torch.utils.data.IterableDataset, ABC):
    """
    A class used to create a data iterator for our data.
    """

    def __init__(self, data):
        """
        Constructor.
        :param data: our data set.
        """
        super(MyIterableDataset).__init__()
        self.dataset = data

    def __iter__(self):
        """
        Implementation of an iterator.
        :return: yield data point, label.
        """
        for frame, label in self.dataset:
            yield frame, label


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    A method for training an epoch.
    :param model: chosen model object.
    :param data_iterator: the iterator for the data.
    :param optimizer: an optimizer method.
    :param criterion: loss function.
    :return: current loss and trained model.
    """
    loss = float(0)
    for i, data in enumerate(data_iterator):
        input, label = data
        input = input.float()
        optimizer.zero_grad()
        output = model(input)
        loss_calc = criterion(torch.argmax(output).float(), label)
        loss_calc = Variable(loss_calc.data, requires_grad=True)
        loss_calc.backward()
        optimizer.step()
        loss += loss_calc.item()
        data_iterator.set_postfix(lost=loss / (i + 1))
    return loss / (i + 1), model


def train_model(model, data_iter, n_epochs, learning_rate, model_to_read=None, start_from_zero=True, weight_decay=0):
    """
     A method for training the model.
    :param model: chosen model object.
    :param data_iter: the iterator for the data.
    :param n_epochs: The number of epochs to train the data.
    :param learning_rate: Chosen learning rate.
    :param model_to_read: if there is a saved model provide it name here.
    :param start_from_zero: boolean, True if not provided any model.
    :param weight_decay: Chosen weight decay.
    :return: None
    """
    if not start_from_zero:
        loaded_file = open(f'model_number_{model_to_read}.pickle', 'rb')
        model = pickle.load(loaded_file)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.L1Loss()
    loss_lst = list()
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(iterable=data_iter.get_data_iterator())
        loss, model = train_epoch(model, pbar, optimizer, loss_func)
        loss_lst.append(loss)
    model_file = open(f'model_number_{NUM_MODEL_NET}.pickle', 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def evaluate(model, data_iterator, criterion):
    """
    An evaluation method.
    :param model: The chosen model.
    :param data_iterator: Iterator for the data.
    :param criterion: loss function
    :return: average of loss.
    """
    lost = list()
    for data in data_iterator:
        input, label = data
        input = input.float()
        output = model(input)
        output = torch.argmax(output)
        lost_calc = criterion(output, label)
        lost.append(lost_calc.item())
    return np.mean(np.array(lost))


def evaluate_knn(train_set, train_tags, test_set, test_tags, model, method):
    """
    An evaluation method for knn model.
    :param train_set: data points for train set.
    :param train_tags: labels for train set.
    :param test_set: data points for test set.
    :param test_tags:labels for test set.
    :param model: the knn model object.
    :param method: evaluation method string.
    :return:
    """
    model.train_model(train_set, train_tags)
    y_hat_train = model.predict(train_set)
    y_hat_test = model.predict(test_set)
    if method == 'accuracy':
        acc_train = np.sum(y_hat_train == train_tags) / len(train_tags)
        acc_test = np.sum(y_hat_test == test_tags) / len(test_tags)
        print(f'train accuracy = {acc_train}, test accuracy =  {acc_test}')
        return y_hat_test, test_tags,


def calculate_accuracy(y_pred, y):
    return np.mean(y_pred == y)


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
    """
    A splitter for the data set.
    :param data: data points.
    :param labels: labels consist with data points.
    :param test_size: float for size of the test from data.
    :return:
    """
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=test_size)
    return train_x, train_y, test_x, test_y


def plot_confusion_matrix(labels, pred_labels):

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(4))
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.show()
