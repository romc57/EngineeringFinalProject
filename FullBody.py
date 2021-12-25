from utils import get_2d_distance, plot_points, print_points, rotate_point, write_to_txt_points
import copy
import math

X = 0
Y = 1

BODY_PARTS_LIST = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
                   'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']

BODY_PARTS_LIST_CLASS = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee',
                         'LAnkle', 'REye', 'LEye', 'REar', 'LEar']

BODY_PARTS_MAP_INDEX = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

INSTRUCTIONS = {
    'keep_still': "Keep still",
    'ankles_not_horizontal': "Make sure your are faced to the camera standing straight\nand the camera is positioned "
                             "horizontal to the ground",
    'ankles_not_horizontal_in_motion': 'Make sure you are facing straight to the camera and the camera is positioned '
                                       'horizontal to the floor',
    'calibrated': 'Done',
    'invalid_points': 'Points are not right'
}


class Body:

    def __init__(self, height, run_dir):
        self.run_dir = run_dir
        self.__ready = False
        self.__valid_points = False
        self.body_points = None
        self.class_body_points = None
        self.class_body_points_centered = None
        self.ignore_indexes = [BODY_PARTS_MAP_INDEX['RElbow'], BODY_PARTS_MAP_INDEX['RWrist'],
                               BODY_PARTS_MAP_INDEX['LElbow'], BODY_PARTS_MAP_INDEX['LWrist'],
                               BODY_PARTS_MAP_INDEX['Background']]
        self.pixel_delta_from_horizontal = 2
        self.height = int(height)
        self.norm_body_lengths = None
        self.init_norm_body_lengths = None
        self.__side_corrections = ['Ankle', 'Knee', 'Hip', 'Shoulder', 'Eye', 'Ear']
        self.frame_index = None

    def is_ready(self):
        return self.__ready

    def got_valid_points(self):
        return self.__valid_points

    def check_body_points(self, points, frame_index):
        self.frame_index = frame_index
        self.__valid_points = False
        self.class_body_points = dict()
        for i, point in enumerate(points):
            if i in self.ignore_indexes:
                continue
            else:
                if not point:
                    return INSTRUCTIONS['invalid_points']
                else:
                    self.class_body_points[BODY_PARTS_LIST[i]] = point
        self.center_r_ankle()
        self.rotate_points_around_r_ankle()
        if not self.__check_ankle_alignment(self.class_body_points['LAnkle'], self.class_body_points['RAnkle']):
            return INSTRUCTIONS['ankles_not_horizontal_in_motion']
        if not self.__side_corrector():
            return INSTRUCTIONS['invalid_points']
        self.__valid_points = True
        self.output_points()

    def calibrate_class(self, body_points, frame_index):
        self.frame_index = frame_index
        self.class_body_points = dict()
        for i, point in enumerate(body_points):
            if i in self.ignore_indexes:
                continue
            else:
                if not point:
                    return INSTRUCTIONS['keep_still']
                else:
                    self.class_body_points[BODY_PARTS_LIST[i]] = point
        if not self.__check_ankle_alignment(self.class_body_points['LAnkle'], self.class_body_points['RAnkle']):
            return INSTRUCTIONS['ankles_not_horizontal']
        self.__ready = True
        self.__normalize_body_lengths()
        self.init_norm_body_lengths = copy.deepcopy(self.norm_body_lengths)
        self.center_r_ankle()
        self.rotate_points_around_r_ankle()
        self.output_points()
        if not self.__side_corrector():
            return INSTRUCTIONS['keep_still']
        return INSTRUCTIONS['calibrated']

    def __side_corrector(self):
        for body_part in self.__side_corrections:
            left = self.class_body_points["L{}".format(body_part)]
            right = self.class_body_points["R{}".format(body_part)]
            if abs(left[X] - right[X]) < 3:
                return False
            if left[X] > right[X]:
                temp = copy.deepcopy(self.class_body_points["L{}".format(body_part)])
                self.class_body_points["L{}".format(body_part)] = self.class_body_points["R{}".format(body_part)]
                self.class_body_points["R{}".format(body_part)] = temp
        return True

    def __check_ankle_alignment(self, l_ankle, r_ankle):
        return abs(int(l_ankle[Y]) - int(r_ankle[Y])) <= self.pixel_delta_from_horizontal

    def __normalize_body_lengths(self):
        self.norm_body_lengths = dict()
        self.norm_body_lengths['ankle_knee'] = (get_2d_distance(self.class_body_points['RAnkle'],
                                                                self.class_body_points['RAnkle']) +
                                                get_2d_distance(self.class_body_points['LAnkle'],
                                                                self.class_body_points['LAnkle'])) / \
                                               (2 * self.height)
        self.norm_body_lengths['knee_hip'] = (get_2d_distance(self.class_body_points['RKnee'],
                                                              self.class_body_points['RHip']) +
                                              get_2d_distance(self.class_body_points['LKnee'],
                                                              self.class_body_points['LHip'])) / \
                                             (2 * self.height)
        self.norm_body_lengths['hips'] = get_2d_distance(self.class_body_points['LHip'],
                                                         self.class_body_points['RHip']) / self.height
        self.norm_body_lengths['shoulders'] = get_2d_distance(self.class_body_points['LShoulder'],
                                                              self.class_body_points['RShoulder']) / self.height
        middle_hip = (self.class_body_points['LHip'][X] + ((self.class_body_points['RHip'][X] -
                                                            self.class_body_points['LHip'][X]) / 2),
                      self.class_body_points['LHip'][Y])
        self.norm_body_lengths['mid_hip_neck'] = get_2d_distance(middle_hip, self.class_body_points['Neck'])/self.height
        self.norm_body_lengths['neck_nose'] = get_2d_distance(self.class_body_points['Neck'],
                                                              self.class_body_points['Nose']) / self.height

    def get_network_point_format(self):
        output = list()
        for i, body_part in enumerate(BODY_PARTS_LIST):
            if i in self.ignore_indexes:
                output.append(None)
            else:
                output.append(self.class_body_points[body_part])
        return output

    def get_class_points_list(self):
        output = list()
        for body_part in BODY_PARTS_LIST_CLASS:
            output.append(self.class_body_points_centered[body_part])
        return output

    def output_points(self):
        point_list = self.get_class_points_list()
        print_points(point_list, BODY_PARTS_LIST_CLASS)
        write_to_txt_points(point_list, self.run_dir, 'centered_points_id_{}'.format(self.frame_index))
        plot_points(point_list, BODY_PARTS_LIST_CLASS, 'Centered body points - frame={}'.format(self.frame_index),
                    self.run_dir, 'centered_idx_{}'.format(self.frame_index), 'centered_plots')

    def center_r_ankle(self):
        self.class_body_points_centered = dict()
        neck_point = self.class_body_points['RAnkle']
        move_x = neck_point[X]
        move_y = neck_point[Y]
        for body_part in self.class_body_points:
            if not self.class_body_points[body_part]:
                continue
            self.class_body_points_centered[body_part] = ((self.class_body_points[body_part][X] - move_x),
                                                          (self.class_body_points[body_part][Y] - move_y))

    def rotate_points_around_r_ankle(self):
        r_ankle_coord = self.class_body_points_centered['RAnkle']
        if r_ankle_coord != (0, 0):
            print('problem')
            return
        for body_part in self.class_body_points_centered:
            if body_part == 'RAnkle':
                continue
            self.class_body_points_centered[body_part] = rotate_point(self.class_body_points_centered[body_part],
                                                                      math.pi)