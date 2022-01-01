from utils import get_2d_distance, plot_points, print_points, rotate_point, write_to_txt_points, get_z_coordinate, \
    plot_profile_points, BODY_PARTS_LIST, BODY_PARTS_LIST_CLASS, INSTRUCTIONS, X, Y, Z, get_ignored_indexes
import copy
import math


class Body:

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.__standing_line_points = None
        self.__user_on_standing_line = False
        self.__frame_sizes = None
        self.__ready = False
        self.__valid_points = False
        self.class_body_points = None
        self.ignore_indexes = get_ignored_indexes(BODY_PARTS_LIST, BODY_PARTS_LIST_CLASS)
        self.pixel_delta_from_horizontal = 30
        self.body_lengths = None
        self.init_body_lengths = None
        self.frame_index = None
        self.__user_height_in_pixel = None
        self.valid_points_list = list()
        self.calibrate_mode = True
        self.__init_pose = None
        self.__lowest_squat_point = None
        self.sample_squat_mode = False
        self.__cur_squat_frames = list()
        self.__cur_lowest_squat_point = None

    def is_ready(self):
        return self.__ready

    def set_standing_line(self, points, frame_sizes):
        self.__standing_line_points = points
        self.__frame_sizes = frame_sizes

    def on_standing_line(self):
        return self.__user_on_standing_line

    def got_valid_points(self):
        return self.__valid_points

    def calibrate_class(self, body_points, frame_index):
        self.frame_index = frame_index
        if not self.__insert_points(body_points):
            self.__ready = False
            return INSTRUCTIONS['missing_points']
        if not self.__set_user_on_standing_line():
            self.__ready = False
            return INSTRUCTIONS['standing_line']
        self.__calculate_user_height()
        self.__init_pose = copy.deepcopy(self.class_body_points)
        self.__ready = True
        return INSTRUCTIONS['calibrated']

    def init_squat_ref(self):
        cur_nose_height = self.__frame_sizes[Y] - self.class_body_points['Nose'][Y]
        init_nose_height = self.__frame_sizes[Y] - self.__init_pose['Nose'][Y]
        if init_nose_height * 0.75 < cur_nose_height and not self.__lowest_squat_point:
            return False
        if not self.__lowest_squat_point:
            self.__lowest_squat_point = cur_nose_height
        if cur_nose_height < self.__lowest_squat_point:
            self.__lowest_squat_point = cur_nose_height
            return False
        if abs(cur_nose_height - init_nose_height) < 10:
            print('Finished: {}'.format(self.__lowest_squat_point / self.__user_height_in_pixel * 100))
            return True

    def check_body_points(self, points, frame_index):
        self.frame_index = frame_index
        self.__valid_points = False
        if not self.__insert_points(points):
            return INSTRUCTIONS['missing_points']
        if not self.__set_user_on_standing_line():
            return INSTRUCTIONS['standing_line']
        self.__valid_points = True

    def squat(self):
        cur_height = self.__frame_sizes[Y] - self.class_body_points['Nose'][Y]
        if len(self.__cur_squat_frames) == 0:
            if abs(cur_height - self.__user_height_in_pixel) < 10:
                self.__cur_squat_frames.append(copy.deepcopy(self.class_body_points))
                self.__cur_lowest_squat_point = self.__frame_sizes[Y] - self.class_body_points['Nose'][Y]
                return False
            else:
                return False
        if abs(cur_height - self.__user_height_in_pixel) < 50:  # End or beginning
            if abs(self.__cur_lowest_squat_point - self.__lowest_squat_point) < 30:  # End
                self.__cur_lowest_squat_point = None
                self.valid_points_list.append(copy.deepcopy(self.__cur_squat_frames))
                self.__cur_squat_frames = list()
                return True
            else:
                self.__cur_lowest_squat_point = cur_height
                self.__cur_squat_frames = list()
        self.__cur_lowest_squat_point = cur_height if cur_height < self.__cur_lowest_squat_point else \
            self.__cur_lowest_squat_point
        self.__cur_squat_frames.append(copy.deepcopy(self.class_body_points))
        return False

    def __insert_points(self, body_points):
        self.class_body_points = dict()
        for i, point in enumerate(body_points):
            if i in self.ignore_indexes:
                continue
            else:
                if not point:
                    return False
                else:
                    self.class_body_points[BODY_PARTS_LIST[i]] = point
        return True

    # Not in use
    def __side_corrector(self):
        if not self.class_body_points:
            return
        for body_part in self.__side_corrections:
            left = self.class_body_points["L{}".format(body_part)]
            right = self.class_body_points["R{}".format(body_part)]
            if abs(left[X] - right[X]) < self.__side_corrections[body_part]:
                print('side corrector: {}: diff {} needed diff {}'.format(body_part, left[X] - right[X],
                                                                          self.__side_corrections[body_part]))
                return False
        return True

    # Not in use
    def __check_heel_alignment(self, l_heel, r_heel):
        if not abs(int(l_heel[Y]) - int(r_heel[Y])) <= self.pixel_delta_from_horizontal:
            print('check ankle alignment: diff {} diff needed {}'.format(abs(int(l_heel[Y]) - int(r_heel[Y])),
                                                                         self.pixel_delta_from_horizontal))
            return False

    # Not in use
    def __fix_symmetry(self):
        if not self.class_body_points:
            return
        for body_part in self.__symmetry_corrections:
            left = self.class_body_points["L{}".format(body_part)]
            right = self.class_body_points["R{}".format(body_part)]
            y_avg = right[Y] + (right[Y] - left[Y]) / 2
            self.class_body_points["L{}".format(body_part)] = (left[X], int(math.ceil(y_avg)))
            self.class_body_points["R{}".format(body_part)] = (right[X], int(math.ceil(y_avg)))

    def __set_user_on_standing_line(self):
        if not self.class_body_points or not self.__standing_line_points:
            return
        left_heel = self.class_body_points['LHeel']
        right_heel = self.class_body_points['RHeel']
        if not (self.__standing_line_points[0][X] < left_heel[X] < self.__standing_line_points[1][X] and
                self.__standing_line_points[0][X] < right_heel[X] < self.__standing_line_points[1][X]):
            self.__user_on_standing_line = False
            return False
        dist_lef = abs(self.__standing_line_points[0][Y] - left_heel[Y])
        dist_right = abs(self.__standing_line_points[0][Y] - left_heel[Y])
        if (dist_lef > 10) or (dist_right > 10):
            self.__user_on_standing_line = False
            return False
        self.__user_on_standing_line = True
        return True

    def __calculate_user_height(self):
        if not self.class_body_points or not self.__standing_line_points:
            return
        nose_y = self.class_body_points['Nose'][Y]
        height = abs(nose_y - self.__standing_line_points[0][Y])
        print('User height: {}'.format(height))
        self.__user_height_in_pixel = height

    def __get_body_lengths(self):
        self.init_body_lengths = dict()
        self.init_body_lengths['ankle_knee'] = (get_2d_distance(self.__init_pose['RAnkle'],
                                                                self.__init_pose['RKnee']) +
                                                get_2d_distance(self.__init_pose['LAnkle'],
                                                                self.__init_pose['LKnee'])) / 2
        self.init_body_lengths['knee_hip'] = (get_2d_distance(self.__init_pose['RKnee'],
                                                              self.__init_pose['RHip']) +
                                              get_2d_distance(self.__init_pose['LKnee'],
                                                              self.__init_pose['LHip'])) / 2
        middle_hip = (self.__init_pose['RHip'][X] + ((self.__init_pose['LHip'][X] -
                                                            self.__init_pose['RHip'][X]) / 2),
                      self.__init_pose['RHip'][Y])
        mid_shoulder = (self.__init_pose['RShoulder'][X] + ((self.__init_pose['RShoulder'][X] -
                                                                   self.__init_pose['RShoulder'][X]) / 2),
                        self.__init_pose['RShoulder'][Y])
        self.init_body_lengths['mid_hip_neck'] = get_2d_distance(middle_hip, mid_shoulder)
        self.init_body_lengths['neck_nose'] = get_2d_distance(self.__init_pose['Neck'],
                                                              self.__init_pose['Nose'])

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

    def center_r_heel(self):
        self.class_body_points_centered = dict()
        heel_point = self.class_body_points['RHeel']
        move_x = heel_point[X]
        move_y = heel_point[Y]
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
                                                          get_z_coordinate(cur, self.init_body_lengths['ankle_knee'],
                                                                           prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['RHip'])
        self.class_body_points_3d['RHip'] = cur + tuple([self.class_body_points_3d['RKnee'][Z] -
                                                         get_z_coordinate(cur, self.init_body_lengths['knee_hip'],
                                                                          prev)])
        cur = copy.deepcopy(self.class_body_points_centered['LAnkle'])
        self.class_body_points_3d['LAnkle'] = cur + tuple([0])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LKnee'])
        self.class_body_points_3d['LKnee'] = cur + tuple([self.class_body_points_3d['LAnkle'][Z] +
                                                          get_z_coordinate(cur, self.init_body_lengths['ankle_knee'],
                                                                           prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LHip'])
        self.class_body_points_3d['LHip'] = cur + tuple([self.class_body_points_3d['LKnee'][Z] -
                                                         get_z_coordinate(cur, self.init_body_lengths['knee_hip'],
                                                                          prev)])
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
                                                         get_z_coordinate(cur, self.init_body_lengths['neck_nose'],
                                                                          prev)])
        self.class_body_points_3d['REye'] = self.class_body_points_centered['REye'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['LEye'] = self.class_body_points_centered['LEye'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['REar'] = self.class_body_points_centered['REar'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['LEar'] = self.class_body_points_centered['LEar'] + \
                                            tuple([self.class_body_points_3d['Nose'][Z]])
