import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import os
import random
import pickle
import copy
import torch

"""
Utils file for common functions along the system
"""

X = 0
Y = 1
Z = 2

MULTI_LABELS = ['high_waste', 'knee_collapse', 'lifting_heels', 'good']
BINARY_LABELS = ['bad', 'good']

BODY_PARTS_LIST_CLASS = ['Nose', 'REye_c', 'LEye_c', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee',
                         'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RTows', 'LTows']  # Relevant body parts for the system

BODY_PARTS_LIST = ['Nose', 'REye_l', 'REye_c', 'REye_r', 'LEye_r', 'LEye_c', 'LEye_l', 'RCheek', 'LCheek', 'RLip',
                   'LLip', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RHand_r', 'LHand_l',
                   'RHand_c', 'LHand_c', 'RHand_l', 'LHand_r', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle',
                   'RHeel', 'LHeel', 'RTows', 'LTows']  # All body parts the cv returns

POSE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15),
              (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
              (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
              (30, 32), (27, 31), (28, 32)]  # Pairs or body parts that are connected

# Map a string to index
BODY_PARTS_MAP_INDEX = {'Nose': 0, 'REye_l': 1, 'REye_c': 2, 'REye_r': 3, 'LEye_r': 4, 'LEye_c': 5, 'LEye_l': 6,
                        'RCheek': 7, 'LCheek': 8, 'RLip': 9, 'LLip': 10, 'RShoulder': 11, 'LShoulder': 12, 'RElbow': 13,
                        'LElbow': 14, 'RWrist': 15, 'LWrist': 16, 'RHand_r': 17, 'LHand_l': 18, 'RHand_c': 19,
                        'LHand_c': 20, 'RHand_l': 21, 'LHand_r': 22, 'RHip': 23, 'LHip': 24, 'RKnee': 25, 'LKnee': 26,
                        'RAnkle': 27, 'LAnkle': 28, 'RHeel': 29, 'LHeel': 30, 'RTows': 31, 'LTows': 32}

# Key to instruction for the UI - Not all instructions are implemented
INSTRUCTIONS = {
    'keep_still': "Keep still",
    'heels_not_horizontal': "Make sure your are faced to the camera standing straight\nand the camera is positioned "
                            "horizontal to the ground",
    'heels_not_horizontal_in_motion': 'Make sure you are facing straight to the camera and the camera is positioned '
                                      'horizontal to the floor',
    'calibrated': 'Done',
    'invalid_points': 'Points are not right',
    'standing_line': 'Please stand on the standing line',
    'missing_points': 'Missing some needed points',
    'lifting_heels': 'Keep your feet flat on the floor',
    'high_waste': 'Bend forward less',
    'knee_collapse': 'Keep knees aligned with your feet',
    'but_forward': 'Imagine sitting down on a chair',
    'head_position': 'Look straight ahead',
    'shoulder_stress': 'Release your shoulders',
    'round_back': 'Engage your core',
    'partial_motion': 'Try to go down more'
}


def get_ignored_indexes(full_list, class_list):
    """
    Return the indexes of body parts the system should ignore
    :param full_list:
    :param class_list:
    :return:
    """
    output = list()
    for idx, part in enumerate(full_list):
        if part not in class_list:
            output.append(idx)
    return output


def plot_points(point_list, label_list, title, out_dir, file_name, file_dir):
    """
    Plots points given on a graph and outputs it into the path given in params
    :param point_list: List of points to plot
    :param label_list: Labels representing the name of the body parts
    :param title: Title of the plot
    :param out_dir: Output directory of the files
    :param file_name: Name of the file to save
    :param file_dir: Directory to save the file inside of the out_dir
    """
    x = list()
    y = list()
    labels = list()
    min_x, min_y, max_x, max_y = None, None, None, None
    for i, point in enumerate(point_list):
        if point:
            min_x = min_x if min_x != None and min_x < point[0] else point[0]
            max_x = max_x if max_x != None and max_x > point[0] else point[0]
            x.append(point[0])
            min_y = min_y if min_y != None and min_y < point[1] else point[1]
            max_y = max_y if max_y != None and max_y > point[1] else point[1]
            y.append(point[1])
            labels.append(label_list[i])
    plt.title(title)
    plt.scatter(np.array(x), np.array(y))
    # plt.xlim((int(min_x) - 1, int(max_x) + 1))
    # plt.ylim((int(min_y) - 1, int(max_y) + 1))
    plt.xlim((-4, 14))
    plt.ylim((-1, 14))
    for i, txt in enumerate(labels):
        plt.annotate("{}".format(txt), (x[i], y[i]))
    file_path = os.path.join(out_dir, file_dir, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_profile_points(point_list, label_list, title, out_dir, file_name, file_dir):
    """
    Plot 3d points at the profile of the user
    :param point_list: List of 3d points
    :param label_list: List of labels for each point representing its body part name
    :param title: Title of the plot
    :param out_dir: Output directory
    :param file_name: Name of the file to save
    :param file_dir: Directory to save the file inside the out_dir
    :return:
    """
    x = list()
    y = list()
    labels = list()
    min_x, min_y, max_x, max_y = None, None, None, None
    for i, point in enumerate(point_list):
        if point:
            min_x = min_x if min_x != None and min_x < point[Z] else point[Z]
            max_x = max_x if max_x != None and max_x > point[Z] else point[Z]
            x.append(point[Z])
            min_y = min_y if min_y != None and min_y < point[Y] else point[Y]
            max_y = max_y if max_y != None and max_y > point[Y] else point[Y]
            y.append(point[Y])
            labels.append(label_list[i])
    plt.scatter(np.array(x), np.array(y))
    plt.xlim((int(min_x) - 100, int(max_x) + 100))
    plt.ylim((int(min_y) - 100, int(max_y) + 100))
    for i, txt in enumerate(labels):
        if txt[0] == 'L':
            plt.annotate("{}".format(txt), (x[i], y[i] + 6))
        else:
            plt.annotate("{}".format(txt), (x[i], y[i] - 10))
    file_path = os.path.join(out_dir, file_dir, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    plt.clf()


def print_points(point_list, body_part_list):
    """
    Print a point list in a certain format
    :param point_list: List of points to print
    :param body_part_list: List of body part, index to string
    """
    output = 'Detected: '
    for i in range(len(point_list)):
        output += "{}: {}, ".format(body_part_list[i], point_list[i])
    print(output)


def create_run_dir():
    """
    Create a run directory for a certain run, named with the date and time of the run and inside of the directory there
    are sub directory for each kind of output. For testing and creating datasets
    :return: The path to the new directory
    """
    output_dir = 'run_dir'
    cur_string_time = "".join(str(datetime.datetime.now()).split(" ")[1].split(".")[0].split(":"))
    cur_string_date = "".join(str(datetime.datetime.now()).split(" ")[0].split("-"))
    name = "run_{}_{}".format(cur_string_date, cur_string_time)
    run_dir_path = os.path.join(output_dir, name)
    os.mkdir(run_dir_path)
    output_directories = ['centered_plots', 'img_samples', 'three_d_points', 'centered_points', 'profile_plots']
    for out_dir in output_directories:
        path = os.path.join(run_dir_path, out_dir)
        os.mkdir(path)
    return run_dir_path


def rotate_point(point, radians):
    """
    Rotate a point around the (0,0)
    :param point: Body point
    :param radians: Radians to rotate
    :return: Rotated point
    """
    x, y = point
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)
    return round(xx, 3), round(yy, 3)


def get_2d_distance(point1, point2):
    """
    Get the distance between two points
    :param point1: First point
    :param point2: Second point
    :return:
    """
    return math.sqrt(math.pow(int(point1[X]) - int(point2[X]), 2) + math.pow(int(point1[Y]) - int(point2[Y]), 2))


def get_z_coordinate(point, distance, relative_p):
    """
    Calculate where the z point will given the old distance of its 2d place from the relative point
    :param point: Point to get z coordinate
    :param distance: Initial distance between reference point and given point on the 2d dim
    :param relative_p: Relative point
    :return: The z coordinate of the point
    """
    temp = math.pow(distance, 2) - math.pow((point[X] - relative_p[X]), 2) - math.pow((point[Y] - relative_p[Y]), 2)
    temp = round(temp, 2)
    if temp < 0:
        temp = 0
    return math.floor(math.sqrt(temp))


def write_to_txt_points(points, run_dir, file_name, dir_name):
    """
    Write points into a text file
    :param points: A list of points to write into the text
    :param run_dir: Run directory to save the file
    :param file_name: Name of the file to be saved
    :param dir_name: Name of the directory to save the file inside the run_dir
    """
    file_path = os.path.join(run_dir, dir_name, file_name)
    with open(file_path, 'w') as writer:
        writer.write(str(points))


def load_txt_point(run_dir, file_name, dir_name):
    """
    Load points from a text file
    :param run_dir: Path to the run directory
    :param file_name: Name of the points file
    :param dir_name: Name of the directory of the points inside the run_dir
    :return: A list of points
    """
    file_path = os.path.join(run_dir, dir_name, file_name)
    with open(file_path, 'r') as reader:
        text = reader.read()
        text = text.replace('\n', ',')
        points = eval(text)
    return points


def load_random_squat(squat_type='good'):
    """
    Load a random set of points representing a squat
    :param squat_type: Type of squat, the kind of classification it got
    :return: Tuple of lists of points
    """
    working_dir = 'training_data_set'
    point_types = ['centered_points', 'three_d_points']
    points = list()
    for point_type in point_types:
        path = os.path.join(working_dir, point_type, squat_type)
        squat_files = os.listdir(path)
        file_idx = random.randint(0, len(squat_files) - 1)
        points.append(list(load_txt_point(working_dir, squat_files[file_idx], os.path.join(point_type, squat_type))))
    return tuple(points)


def get_standing_line(width, height, height_fraction, line_fraction):
    """
    Given the width and height of an image return and the standing line specs, height on the image and width of the line
    The function will return the coordinates the standing line should be presented on
    :param width: Width of the image
    :param height: Height of the image
    :param height_fraction: Fraction of the height of the image where the standing line will be positioned
    :param line_fraction: Width of the line at the ration of the image width
    :return: Standing line coordinates
    """
    middle_frame = width / 2
    height_position = height - (height * height_fraction)
    fraction_width = width * line_fraction
    f_point = (int(math.floor(middle_frame - (fraction_width / 2))), int(math.floor(height_position)))
    s_point = (int(math.floor(middle_frame + (fraction_width / 2))), int(math.floor(height_position)))
    return f_point, s_point


def get_data_set(directory, folder_name, multi_labels=MULTI_LABELS, binary_labels=BINARY_LABELS, multi=True):
    """
    Get a data set from a folder
    :param multi_labels:
    :param directory: Path to dataset directory
    :param folder_name: Name of the containing folder
    :param multi: Should the dataset be multiclass or binary
    :return: The dataset and labels
    """
    output_data = list()
    output_labels = list()
    if multi:
        folder_list = multi_labels
    else:
        folder_list = binary_labels
    path = os.path.join(directory, folder_name)
    for i, folder in enumerate(folder_list):
        for file in os.listdir(os.path.join(path, folder)):
            curr_data = load_txt_point(path, file, folder)
            output_data.append(curr_data)
            label = [0] * len(folder_list)
            label[i] = 1
            output_labels.append(torch.tensor(label))
    return output_data, output_labels


def get_data_set_multi(directory, folder_name):
    output_data = list()
    output_labels = list()
    path = os.path.join(directory, folder_name)
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            curr_data = load_txt_point(path, file, folder)
            output_data.append(curr_data)
            if folder == 'bad':
                output_labels.append((1, 0))
            elif folder == 'good':
                output_labels.append((0, 1))
    return output_data, output_labels


def find_min_len(data_set):
    """
    Find the smallest frame count in a dataset
    :param data_set: dataset object
    :return: Minimum frame samples in the dataset
    """
    min_len = 1000
    for video_frames in data_set:
        curr_len = len(video_frames)
        if curr_len < min_len:
            min_len = curr_len
    return min_len


def find_min_y_index(frames):
    head_y_position = frames[:, 0, 1]
    return np.where(head_y_position == np.amin(head_y_position))[0]


def convert_list_to_np(list_obj):
    return np.array(list_obj)


def find_slicing_indices(wanted_len, min_move, frames_len):
    if wanted_len % 2 == 0:
        move_down_size = np.floor(wanted_len / 2).astype(np.int64)
        move_up_size = move_down_size
    else:
        move_up_size = np.floor(wanted_len / 2).astype(np.int64)
        move_down_size = wanted_len - move_up_size
    indices_down = np.linspace(0, min_move, num=move_down_size).astype(np.int64)
    indices_up = np.linspace(min_move + 1, frames_len - 1, num=move_up_size).astype(np.int64)
    return np.concatenate([indices_down, indices_up])


def normalize_data_len(data_set):
    output_data = list()
    min_len = find_min_len(data_set)
    for video_frames in data_set:
        curr_len = len(video_frames)
        video_frames = np.array(video_frames)
        if curr_len == min_len:
            output_data.append(video_frames)
            continue
        if min_len < curr_len:
            min_index = find_min_y_index(video_frames)
            slicing_indices = find_slicing_indices(min_len, min_index[0], curr_len)
            video_frames_new = video_frames[slicing_indices]
            output_data.append(video_frames_new)
    return output_data


def load_models(model_paths):
    output_models = list()
    for model_path in model_paths:
        loaded_file = open(f'{model_path}', 'rb')
        model = pickle.load(loaded_file)
        output_models.append(model)
    return output_models
