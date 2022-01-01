import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import os

X = 0
Y = 1

points_example = [(333, 10), (333, 73), (278, 83), (389, 73), (361, 271), (306, 396), (417, 302), (292, 260),
                  (361, 396), (417, 302), (333, 0), (347, 0), (306, 10), (361, 10)]

BODY_PARTS_LIST = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
                   'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']


def get_2d_distance(point1, point2):
    return math.sqrt(math.pow(int(point1[X]) - int(point2[X]), 2) + math.pow(int(point1[Y]) - int(point2[Y]), 2))


def plot_points(point_list, label_list, title, out_dir, file_name, file_dir):
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
    plt.xlim((int(min_x) - 100, int(max_x) + 100))
    plt.ylim((int(min_y) - 100, int(max_y) + 100))
    for i, txt in enumerate(labels):
        if 'L' in txt:
            plt.annotate("{}-{}".format(txt, (x[i], y[i])), (x[i] - 100, y[i]))
        else:
            plt.annotate("{}-{}".format(txt, (x[i], y[i])), (x[i], y[i]))
    file_path = os.path.join(out_dir, file_dir, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    # plt.show()
    plt.clf()


def print_points(point_list, body_part_list):
    output = 'Detected: '
    for i in range(len(point_list)):
        output += "{}: {}, ".format(body_part_list[i], point_list[i])
    print(output)


def create_run_dir():
    output_dir = 'run_dir'
    cur_string_time = "".join(str(datetime.datetime.now()).split(" ")[1].split(".")[0].split(":"))
    cur_string_date = "".join(str(datetime.datetime.now()).split(" ")[0].split("-"))
    name = "run_{}_{}".format(cur_string_date, cur_string_time)
    run_dir_path = os.path.join(output_dir, name)
    os.mkdir(run_dir_path)
    output_directories = ['centered_plots', 'vision_plots', 'img_samples', 'centered_points']
    for out_dir in output_directories:
        path = os.path.join(run_dir_path, out_dir)
        os.mkdir(path)
    return run_dir_path


def rotate_point(point, radians):
    x, y = point
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)
    return round(xx, 3), round(yy, 3)


def write_to_txt_points(points, run_dir, file_name):
    file_path = os.path.join(run_dir, 'centered_points', file_name)
    with open(file_path, 'w') as writer:
        writer.write(str(points))


def load_txt_point(run_dir, file_name):
    file_path = os.path.join(run_dir, 'centered_points', file_name)
    with open(file_path, 'r') as reader:
        points = eval(reader.read())
    return points


def get_data_set(run_dir, folder_name):
    data = list()
    for directory in os.listdir(run_dir):
        curr_data = list()
        path = os.path.join(run_dir, directory)
        for file in os.listdir(os.path.join(run_dir, directory, folder_name)):
            curr_data.append(load_txt_point(path, file))
        data.append(curr_data)
    return data


def get_data_for_knn(run_dir, folder_name):
    data = get_data_set(run_dir, folder_name)
    data_knn = np.array(data)
    data_knn = data_knn.reshape(len(data_knn), -1)
    return data_knn




