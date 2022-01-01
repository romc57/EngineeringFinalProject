from utils import  get_data_set
from model_network import *
import tensorflow as tf







if __name__ == '__main__':
    data = get_data_set(r'run_dir', 'centered_points')

    data_knn = np.array(data)
    model = SimpleKNN(1)
    data_knn = data_knn.reshape(len(data_knn) , -1)
    fit = model.train_model(data_knn, [1, 1])
    tag = model.predict(data_knn)
    print(tag)

    data_net = np.array(data)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(30,14, 2)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics='accuracy')
    model.fit(data_net,np.array([1,1]), epochs=50)
    test_loss, test_acc = model.evaluate(data_net, np.array([1,1]))
    print(test_loss)
    print(test_acc)

    print('debug!')
