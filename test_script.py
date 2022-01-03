from utils import  get_data_set, normalize_data_len
from model_network import *

TEST_SET_SIZE = 0.2
EPOCHS = 2
LR = 0.01
WD = 0.01
TRAINING_DATA = r'training_data_set'
THREE_DIM = r'three_d_points'
TWO_DIM = r'centered_points'

if __name__ == '__main__':
    # data, tags = get_data_set(TRAINING_DATA, THREE_DIM)
    # print(f'Number of samples in data : {len(tags)}, train set size = {(1 - TEST_SET_SIZE) * len(tags)}')
    # data = normalize_data_len(data)
    # data_knn = np.array(data)
    # train_x, train_y, test_x, test_y = split_data_train_test(data_knn, tags, TEST_SET_SIZE)
    # dim = train_x.shape[1] * train_x.shape[2] * train_x.shape[3]
    # # runs knn:
    # model = SimpleKNN(1, dim)
    # train_x_knn = train_x.reshape(len(train_x) , -1)
    # test_x_knn = test_x.reshape(len(test_x) , -1)
    # evaluate_knn(train_x_knn, train_y, test_x_knn, test_y, model, 'accuracy')

    # runs knn multi:
    data_multi, tags_multi = get_data_set(TRAINING_DATA, TWO_DIM, True)
    print(f'Number of samples in data : {len(tags_multi)}, train set size = {(1 - TEST_SET_SIZE) * len(tags_multi)}')
    data_multi = normalize_data_len(data_multi)
    data_knn_multi = np.array(data_multi)
    train_x, train_y, test_x, test_y = split_data_train_test(data_knn_multi, tags_multi, TEST_SET_SIZE)
    dim = train_x.shape[1] * train_x.shape[2] * train_x.shape[3]
    model = SimpleKNN(1, dim)
    train_x_knn = train_x.reshape(len(train_x), -1)
    test_x_knn = test_x.reshape(len(test_x), -1)
    evaluate_knn(train_x_knn, train_y, test_x_knn, test_y, model, 'accuracy')

    # runs network:
    # data_manager = DataManager(torch.tensor(train_x), torch.tensor(train_y))
    # data_net = data_manager.get_data_iterator()
    # data_test_manager = DataManager(torch.tensor(test_x), torch.tensor(test_y))
    #
    # model_net = LinearNeuralNet(dim)
    # train_model(model_net, data_manager, EPOCHS, learning_rate=LR,weight_decay=WD)
    # print(evaluate(model_net, data_test_manager.get_data_iterator(), binary_accuracy))
