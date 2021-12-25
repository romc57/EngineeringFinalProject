from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset

DIM = 50


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


def train_epoch():
    pass


def train_model():
    pass


def evaluate():
    pass

