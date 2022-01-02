import numpy as np
import torch

from utils import  get_data_set, normalize_data_len
from model_network import *
# import tensorflow as tf


if __name__ == '__main__':
    data, tags = get_data_set(r'training_data_set', r'three_d_points')
    data = normalize_data_len(data)
    data_knn = np.array(data)
    train_x, train_y, test_x, test_y = split_data_train_test(data_knn, tags, 0.2)

    # runs knn
    model = SimpleKNN(1)
    train_x_knn = train_x.reshape(len(train_x) , -1)
    test_x_knn = test_x.reshape(len(test_x) , -1)
    evaluate_knn(train_x_knn, train_y, test_x_knn, test_y, model, 'accuracy')

    # runs network
    data_manager = DataManager(torch.tensor(train_x), torch.tensor(train_y))
    data_net = data_manager.get_data_iterator()
    data_test_manager = DataManager(torch.tensor(test_x), torch.tensor(test_y))
    model_net = LinearNeuralNet()
    train_model(model_net, data_manager, 20, 0.01,weight_decay=0.01)
    print(evaluate(model_net, data_test_manager.get_data_iterator(), binary_accuracy))
