from utils import get_2d_distance, plot_points, print_points, rotate_point, write_to_txt_points, get_z_coordinate, \
    plot_profile_points
import copy
import math

X = 0
Y = 1
Z = 2

BODY_PARTS_LIST = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
                   'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']

BODY_PARTS_LIST_CLASS = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee',
                         'LAnkle', 'REye', 'LEye', 'REar', 'LEar']

POSE_PAIRS_CLASS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["Neck", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"],
                    ["Nose", "REye"], ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

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

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.__standing_line = None
        self.__user_on_standing_line = False
        self.__frame_sizes = None
        self.__ready = False
        self.__valid_points = False
        self.body_points = None
        self.class_body_points = None
        self.class_body_points_centered = None
        self.class_body_points_3d = None
        self.init_class_body_points_centered = None
        self.ignore_indexes = [BODY_PARTS_MAP_INDEX['RElbow'], BODY_PARTS_MAP_INDEX['RWrist'],
                               BODY_PARTS_MAP_INDEX['LElbow'], BODY_PARTS_MAP_INDEX['LWrist'],
                               BODY_PARTS_MAP_INDEX['Background']]
        self.pixel_delta_from_horizontal = 5
        self.body_lengths = None
        self.init_body_lengths = None
        self.__side_corrections = {'Ankle': 15, 'Knee': 15, 'Hip': 15, 'Shoulder': 15, 'Eye': 5, 'Ear': 5}
        self.__symmetry_corrections = ['Ankle', 'Knee', 'Hip']
        self.frame_index = None

    def is_ready(self):
        return self.__ready

    def set_standing_line(self, points, frame_sizes):
        self.__standing_line = points
        self.__frame_sizes = frame_sizes

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
        if not self.__check_ankle_alignment(self.class_body_points['LAnkle'], self.class_body_points['RAnkle']):
            return INSTRUCTIONS['ankles_not_horizontal_in_motion']
        if not self.__side_corrector():
            return INSTRUCTIONS['invalid_points']
        #self.__fix_symmetry()
        self.center_r_ankle()
        self.rotate_points_around_r_ankle()
        self.create_3d_points()
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
        if not self.__side_corrector():
            return INSTRUCTIONS['keep_still']
        self.__fix_symmetry()
        self.__ready = True
        self.center_r_ankle()
        self.rotate_points_around_r_ankle()
        self.init_class_body_points_centered = copy.deepcopy(self.class_body_points_centered)
        self.__get_body_lengths()
        self.create_3d_points()
        self.output_points()
        return INSTRUCTIONS['calibrated']

    def __side_corrector(self):
        for body_part in self.__side_corrections:
            left = self.class_body_points["L{}".format(body_part)]
            right = self.class_body_points["R{}".format(body_part)]
            if abs(left[X] - right[X]) < self.__side_corrections[body_part]:
                return False
            if left[X] < right[X]:
                self.class_body_points["L{}".format(body_part)] = right
                self.class_body_points["R{}".format(body_part)] = left
        return True

    def __check_ankle_alignment(self, l_ankle, r_ankle):
        return abs(int(l_ankle[Y]) - int(r_ankle[Y])) <= self.pixel_delta_from_horizontal

    def __fix_symmetry(self):
        for body_part in self.__symmetry_corrections:
            left = self.class_body_points["L{}".format(body_part)]
            right = self.class_body_points["R{}".format(body_part)]
            y_avg = right[Y] + (right[Y] - left[Y]) / 2
            self.class_body_points["L{}".format(body_part)] = (left[X], int(math.ceil(y_avg)))
            self.class_body_points["R{}".format(body_part)] = (right[X], int(math.ceil(y_avg)))

    def __set_user_on_standing_line(self):
        left_ankle = self.class_body_points['LAnkle']
        right_ankle = self.class_body_points['RAnkle']
        pass

    def __get_body_lengths_not_in_use(self):
        self.body_lengths = dict()
        for edge in POSE_PAIRS_CLASS:
            key = "{}_{}".format(edge[0], edge[1])
            p1 = self.class_body_points_centered[edge[0]]
            p2 = self.class_body_points_centered[edge[1]]
            self.body_lengths[key] = get_2d_distance(p1, p2)

    def __get_body_lengths(self):
        self.init_body_lengths = dict()
        self.init_body_lengths['ankle_knee'] = (get_2d_distance(self.class_body_points['RAnkle'],
                                                                self.class_body_points['RKnee']) +
                                                get_2d_distance(self.class_body_points['LAnkle'],
                                                                self.class_body_points['LKnee'])) / 2
        self.init_body_lengths['knee_hip'] = (get_2d_distance(self.class_body_points['RKnee'],
                                                              self.class_body_points['RHip']) +
                                              get_2d_distance(self.class_body_points['LKnee'],
                                                              self.class_body_points['LHip'])) / 2
        self.init_body_lengths['hips'] = get_2d_distance(self.class_body_points['LHip'],
                                                         self.class_body_points['RHip'])
        self.init_body_lengths['shoulders'] = get_2d_distance(self.class_body_points['LShoulder'],
                                                              self.class_body_points['RShoulder'])
        middle_hip = (self.class_body_points['RHip'][X] + ((self.class_body_points['LHip'][X] -
                                                            self.class_body_points['RHip'][X]) / 2),
                      self.class_body_points['RHip'][Y])
        self.init_body_lengths['mid_hip_neck'] = get_2d_distance(middle_hip, self.class_body_points['Neck'])
        self.init_body_lengths['neck_nose'] = get_2d_distance(self.class_body_points['Neck'],
                                                              self.class_body_points['Nose'])

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

    def get_class_3d_points_list(self):
        output = list()
        for body_part in BODY_PARTS_LIST_CLASS:
            output.append(self.class_body_points_3d[body_part])
        return output

    def output_points(self):
        point_list = self.get_class_points_list()
        print_points(point_list, BODY_PARTS_LIST_CLASS)
        write_to_txt_points(point_list, self.run_dir, 'centered_points_id_{}.txt'.format(self.frame_index),
                            'centered_points')
        three_d_point_list = self.get_class_3d_points_list()
        write_to_txt_points(three_d_point_list, self.run_dir, 'three_d_points_id_{}.txt'.format(self.frame_index),
                            'three_d_points')
        plot_profile_points(three_d_point_list, BODY_PARTS_LIST_CLASS,
                            'Profile body points - frame={}'.format(self.frame_index), self.run_dir,
                            'profile_idx_{}'.format(self.frame_index), 'profile_plots')
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

    def create_3d_points(self):
        self.class_body_points_3d = dict()
        cur = copy.deepcopy(self.class_body_points_centered['RAnkle'])
        self.class_body_points_3d['RAnkle'] = cur + tuple([0])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['RKnee'])
        self.class_body_points_3d['RKnee'] = cur + tuple([self.class_body_points_3d['RAnkle'][Z] +
                                                    get_z_coordinate(cur, self.init_body_lengths['ankle_knee'], prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['RHip'])
        self.class_body_points_3d['RHip'] = cur + tuple([self.class_body_points_3d['RKnee'][Z] -
                                                   get_z_coordinate(cur, self.init_body_lengths['knee_hip'], prev)])
        cur = copy.deepcopy(self.class_body_points_centered['LAnkle'])
        self.class_body_points_3d['LAnkle'] = cur + tuple([0])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LKnee'])
        self.class_body_points_3d['LKnee'] = cur + tuple([self.class_body_points_3d['LAnkle'][Z] +
                                                    get_z_coordinate(cur, self.init_body_lengths['ankle_knee'], prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LHip'])
        self.class_body_points_3d['LHip'] = cur + tuple([self.class_body_points_3d['LKnee'][Z] -
                                                   get_z_coordinate(cur, self.init_body_lengths['knee_hip'], prev)])
        middle_hip = (self.class_body_points_3d['RHip'][X] + ((self.class_body_points_3d['LHip'][X] -
                                                               self.class_body_points_3d['RHip'][X]) / 2),
                      self.class_body_points_3d['RHip'][Y], self.class_body_points_3d['RHip'][Z])
        cur = copy.deepcopy(self.class_body_points_centered['Neck'])
        self.class_body_points_3d['Neck'] = cur + tuple([middle_hip[Z] +
                                                   get_z_coordinate(cur, self.init_body_lengths['mid_hip_neck'],
                                                                    middle_hip)])
        prev = cur
        self.class_body_points_3d['RShoulder'] = self.class_body_points_centered['RShoulder'] + \
                                                 tuple([self.class_body_points_3d['Neck'][Z]])
        self.class_body_points_3d['LShoulder'] = self.class_body_points_centered['LShoulder'] + \
                                                 tuple([self.class_body_points_3d['Neck'][Z]])
        cur = copy.deepcopy(self.class_body_points_centered['Nose'])
        self.class_body_points_3d['Nose'] = cur + tuple([self.class_body_points_3d['Neck'][Z] +
                                                   get_z_coordinate(cur, self.init_body_lengths['neck_nose'], prev)])
        self.class_body_points_3d['REye'] = self.class_body_points_centered['REye'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['LEye'] = self.class_body_points_centered['LEye'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['REar'] = self.class_body_points_centered['REar'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['LEar'] = self.class_body_points_centered['LEar'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])


