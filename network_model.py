import numpy as np
import time
from utils import *
import os
import torch as T

device = T.device("cpu")


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            # 'left_elbow', 'right_elbow',
            # 'left_wrist', 'right_wrist',
            # 'left_pinky_1', 'right_pinky_1',
            # 'left_index_1', 'right_index_1',
            # 'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            # self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            # self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            # self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            # self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            # self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            # self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            # self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            # self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            # self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            # self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            # self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            # self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


class SquatsDataSet(T.utils.data.Dataset):
    # sex units   state   test_score  major
    # -1  0.395   0 0 1   0.5120      1
    #  1  0.275   0 1 0   0.2860      2
    # -1  0.220   1 0 0   0.3350      0
    # sex: -1 = male, +1 = female
    # state: maryland, nebraska, oklahoma
    # major: finance, geology, history

    def __init__(self, root_dir, data_folder, folders=APP_MULTI, train=True, test_size=0.2):
        self.embeder = FullBodyPoseEmbedder()
        X, Y = get_data_set(root_dir, data_folder, folders, multi=True)
        self.x_data, self.y_data = normalize_data_len(X, Y)
        self.x_data = self.embed_data(self.x_data)
        if train:
            self.x_data, self.y_data = self.x_data[:int((1 - test_size) * len(self.x_data))], \
                                       self.y_data[:int((1 - test_size) * len(self.y_data))]
        else:
            self.x_data, self.y_data = self.x_data[int((1 - test_size) * len(self.x_data)):], \
                                       self.y_data[int((1 - test_size) * len(self.y_data)):]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        trgts = self.y_data[idx]
        sample = {
            'predictors': preds,
            'targets': trgts
        }
        return sample

    def embed_data(self, dataset):
        out_dataset = []
        for X in dataset:
            new_X = []
            for frame in X:
                embedded = self.embeder(np.array(frame))
                new_X.append(embedded)
            out_dataset.append(new_X)
        return torch.tensor(out_dataset)


# -----------------------------------------------------------

class Net(T.nn.Module):
    def __init__(self, dim=660):
        super(Net, self).__init__()
        self.flatten = T.nn.Flatten()
        self.hid1 = T.nn.Linear(dim, 60)  # 6-(10-10)-3
        self.hid2 = T.nn.Linear(60, 30)
        self.oupt = T.nn.Linear(30, 10)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.relu(self.flatten(x))
        z = T.relu(self.hid1(z))
        z = T.relu(self.hid2(z))
        z = self.oupt(z)  # no softmax: CrossEntropyLoss()
        return z


# -----------------------------------------------------------

class Net2(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = T.nn.Conv2d(20, 6, 3)
        self.pool = T.nn.MaxPool2d(2, 2)
        self.conv2 = T.nn.Conv2d(6, 48, 1)
        self.fc1 = T.nn.Linear(432, 120)
        self.fc2 = T.nn.Linear(120, 84)
        self.fc3 = T.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -----------------------------------------------------------

def accuracy(model, ds):
    # assumes model.eval()
    # granular but slow approach
    n_correct = 0;
    n_wrong = 0
    for i in range(len(ds)):
        X = ds[i]['predictors'].reshape(1, 20, 11, 3)
        Y = ds[i]['targets']  # [0] [1] or [2]
        with T.no_grad():
            oupt = model(X)  # logits form

        big_idx = T.argmax(oupt)  # [0] [1] or [2]
        if big_idx == Y:
            n_correct += 1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc


# -----------------------------------------------------------

def accuracy_quick(model, dataset):
    # assumes model.eval()
    # en masse but quick
    n = len(dataset)
    X = dataset[0:n]['predictors']
    Y = T.flatten(dataset[0:n]['targets'])  # 1-D

    with T.no_grad():
        oupt = model(X)
    # (_, arg_maxs) = T.max(oupt, dim=1)  # old style
    arg_maxs = T.argmax(oupt, dim=1)  # collapse cols
    num_correct = T.sum(Y == arg_maxs)
    acc = (num_correct * 1.0 / len(dataset))
    return acc.item()


# -----------------------------------------------------------

def main():
    # 0. get started
    print("\nBegin predict squat training \n")
    np.random.seed(1)
    T.manual_seed(1)

    # 1. create DataLoader objects
    print("Creating Datasets ")

    train_file = r'D:\projects_files\final project'
    train_folder = r'LabeledSquatPose'

    train_ds = SquatsDataSet(train_file, train_folder, train=True)

    test_file = ".\\Data\\students_test.txt"
    test_ds = SquatsDataSet(train_file, train_folder, train=False)  # all 40 rows

    bat_size = 10
    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=bat_size, shuffle=True)

    # 2. create network
    net = Net2().float().to(device)
    print(net.type(torch.float64))
    # 3. train model
    max_epochs = 3000
    ep_log_interval = 100
    lrn_rate = 0.001

    # -----------------------------------------------------------

    loss_func = T.nn.CrossEntropyLoss()  # apply log-softmax()
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)

    print("\nbat_size = %3d " % bat_size)
    print("loss = " + str(loss_func))
    print("optimizer = SGD")
    print("max_epochs = %3d " % max_epochs)
    print("lrn_rate = %0.3f " % lrn_rate)

    print("\nStarting train with saved checkpoints")
    net.train()
    for epoch in range(0, max_epochs):
        T.manual_seed(1 + epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']  # inputs
            Y = batch['targets']
            # shape [10,3] (!)
            optimizer.zero_grad()
            if len(X) != 10:
                continue
            oupt = net(X)  # shape [10] (!)

            loss_val = loss_func(oupt, Y)  # avg loss in batch
            epoch_loss += loss_val.item()  # a sum of averages
            loss_val.backward()
            optimizer.step()

        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % \
                  (epoch, epoch_loss))

            # checkpoint after 0-based epoch 100, 200, etc.
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            fn = ".\\Log\\" + str(dt) + str("-") + \
                 str(epoch) + "_checkpoint.pt"
            # -----------------------------------------------------------

            info_dict = {
                'epoch': epoch,
                'numpy_random_state': np.random.get_state(),
                'torch_random_state': T.random.get_rng_state(),
                'net_state': net.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            # T.save(info_dict, fn)

    print("Training complete ")

    # 4. evaluate model accuracy
    print("\nComputing model accuracy")
    net.eval()
    acc_train = accuracy(net, train_ds)  # item-by-item
    print("Accuracy on training data = %0.4f" % acc_train)
    acc_test = accuracy(net, test_ds)  # en masse
    # acc_test = accuracy_quick(net, test_ds)  # en masse
    print("Accuracy on test data = %0.4f" % acc_test)

    # # 5. make a prediction
    # print("\nPredicting for (M  30.5  oklahoma  543): ")
    # inpt = np.array([[-1, 0.305,  0,0,1,  0.543]],
    #                 dtype=np.float32)
    # inpt = T.tensor(inpt, dtype=T.float32).to(device)
    # with T.no_grad():
    #     logits = net(inpt)      # values do not sum to 1.0
    # probs = T.softmax(logits, dim=1)  # tensor
    # probs = probs.numpy()  # numpy vector prints better
    # np.set_printoptions(precision=4, suppress=True)
    # print(probs)

    # 6. save model (state_dict approach)
    # print("\nSaving trained model ")
    fn = ".\\Models\\student_model.pth"
    # T.save(net.state_dict(), fn)

    # saved_model = Net()
    # saved_model.load_state_dict(T.load(fn))
    # use saved_model to make prediction(s)

    print("\nEnd predict major demo")


if __name__ == "__main__":
    main()
