from utils import  get_data_set, normalize_data_len
from model_network import *
from network_model import *

TEST_SET_SIZE = 0.1
EPOCHS = 100
LR = 0.000001
WD = 0.0005
TRAINING_DATA = r'training_data_set'
THREE_DIM = r'three_d_points'
TWO_DIM = r'centered_points'
APP_TRAINING = r'D:\projects_files\final project'
FOLDER = r'LabeledSquatPose'

def embed_data(dataset, embeder):
    out_dataset = []
    for X in dataset:
        new_X = []
        for frame in X:
            embedded = embeder(np.array(frame))
            new_X.append(embedded)
        out_dataset.append(new_X)
    return np.array(out_dataset)

if __name__ == '__main__':
    # runs knn:
    embedded = FullBodyPoseEmbedder()
    data_multi, tags_multi = get_data_set(APP_TRAINING, FOLDER)
    data_multi, tags_multi = normalize_data_len(data_multi, tags_multi)
    tags_multi = np.array(tags_multi)
    data_multi = embed_data(data_multi, embedded)
    print(f'Number of samples in data : {len(tags_multi)}, train set size = {int((1 - TEST_SET_SIZE) * len(tags_multi))}')
    data_knn_multi = np.array(data_multi)
    train_x, train_y, test_x, test_y = split_data_train_test(data_knn_multi, tags_multi, TEST_SET_SIZE)
    dim = train_x.shape[1] * train_x.shape[2] * train_x.shape[3]

    train_x_knn = train_x.reshape(len(train_x), -1)
    test_x_knn = test_x.reshape(len(test_x), -1)
    for i in range(1, 11):
        model = SimpleKNN(i, dim)
        print(f'KNN with {i} neighbors:')
        y_hat, test_y = evaluate_knn(train_x_knn, train_y, test_x_knn, test_y, model, 'accuracy')
        # plot_confusion_matrix(y_hat, test_y)

    # runs network:
    # data_manager = DataManager(torch.tensor(train_x), torch.tensor(train_y))
    # data_net = data_manager.get_data_iterator(batch_size=5)
    # data_test_manager = DataManager(torch.tensor(test_x), torch.tensor(test_y))
    #
    # model_net = Net(3)
    # train_model(model_net, data_manager, EPOCHS, learning_rate=LR, weight_decay=WD)
    # print(f'NN for {TWO_DIM} return accuracy of '
    #       f'{evaluate(model_net, data_test_manager.get_data_iterator(), calculate_accuracy)} \nfor the test set.')
