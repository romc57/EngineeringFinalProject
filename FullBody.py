from utils import get_2d_distance, plot_points, rotate_point, write_to_txt_points, get_z_coordinate, \
    plot_profile_points, BODY_PARTS_LIST, BODY_PARTS_LIST_CLASS, INSTRUCTIONS, X, Y, Z, get_ignored_indexes
import copy
import math
import datetime


class Body:

    def __init__(self, run_dir, training_dir=None):
        self.run_dir = run_dir
        self.__standing_line_points = None
        self.__user_on_standing_line = False
        self.__frame_sizes = None
        self.__ready = False
        self.__valid_points = False
        self.class_body_points = None
        self.class_body_points_centered = None
        self.class_body_points_3d = None
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
        self.training_dir = training_dir

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
        self.__get_body_lengths()
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

    def get_squat(self, squat_idx=-1):
        if len(self.valid_points_list) == 0:
            return  False
        elif squat_idx != -1 and squat_idx >= len(self.valid_points_list):
            return False
        else:
            squat_class_type = self.valid_points_list[squat_idx]
            three_d, centered = list(), list()
            for frame in squat_class_type:
                self.class_body_points = copy.deepcopy(frame)
                self.center_r_heel()
                self.rotate_points_around_r_heel()
                self.create_3d_points()
                centered.append(self.get_class_points_list())
                three_d.append(self.get_class_3d_points_list())
            return centered, three_d

    def squat(self):
        cur_height = self.__frame_sizes[Y] - self.class_body_points['Nose'][Y]  # Current user height
        if len(self.__cur_squat_frames) == 0:  # The beginning of the squat
            if self.__user_height_in_pixel - 5 < cur_height:  # If the user is standing - beginning of movement
                self.__cur_squat_frames.append(copy.deepcopy(self.class_body_points))  # Add capture to squat list
                self.__cur_lowest_squat_point = self.__frame_sizes[Y] - self.class_body_points['Nose'][Y]  # Set min
                return False  # Continue movement
            else:  # If the user is not standing
                return False  # Don't create a squat set yet
        print('User height ratio {}'.format(abs(cur_height - self.__user_height_in_pixel)))
        if self.__user_height_in_pixel - 5 < cur_height:  # End or beginning of movement
            print('User lowest point ratio {}'.format(abs(self.__cur_lowest_squat_point - self.__lowest_squat_point)))
            if (self.__cur_lowest_squat_point - self.__lowest_squat_point) < 10:  # if End - got the low already
                self.__cur_lowest_squat_point = None  # Reset the lowest point
                self.__cur_squat_frames.append(copy.deepcopy(self.class_body_points))  # Add capture to list
                self.valid_points_list.append(copy.deepcopy(self.__cur_squat_frames))  # Add squat list to list of squats
                self.__cur_squat_frames = list()  # Reset the current squat list
                return True  # Return end of movement
        self.__cur_lowest_squat_point = cur_height if cur_height < self.__cur_lowest_squat_point else \
            self.__cur_lowest_squat_point  # Set the lowest point in the movement
        self.__cur_squat_frames.append(copy.deepcopy(self.class_body_points))  # Add capture to list
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
        self.class_body_points['frame_idx'] = self.frame_index
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
        if (dist_lef > 15) or (dist_right > 15):
            self.__user_on_standing_line = False
            return False
        self.__user_on_standing_line = True
        return True

    def __calculate_user_height(self):
        if not self.class_body_points or not self.__standing_line_points:
            return
        nose_y = self.class_body_points['Nose'][Y]
        height = abs(nose_y - self.__standing_line_points[0][Y])
        self.__user_height_in_pixel = height

    def __get_body_lengths(self):
        self.init_body_lengths = dict()
        self.init_body_lengths['r_heel_ankle'] = get_2d_distance((self.__init_pose['RAnkle'][X],
                                                                  self.__init_pose['RHeel'][Y]),
                                                                 self.__init_pose['RAnkle'])
        self.init_body_lengths['l_heel_ankle'] = get_2d_distance((self.__init_pose['LAnkle'][X],
                                                                  self.__init_pose['LHeel'][Y]),
                                                                 self.__init_pose['LAnkle'])
        self.init_body_lengths['r_ankle_knee'] = get_2d_distance(self.__init_pose['RAnkle'], self.__init_pose['RKnee'])
        self.init_body_lengths['l_ankle_knee'] = get_2d_distance(self.__init_pose['LAnkle'], self.__init_pose['LKnee'])
        self.init_body_lengths['r_knee_hip'] = get_2d_distance(self.__init_pose['RKnee'], self.__init_pose['RHip'])
        self.init_body_lengths['l_knee_hip'] = get_2d_distance(self.__init_pose['LKnee'], self.__init_pose['LHip'])
        self.init_body_lengths['r_hip_shoulder'] = get_2d_distance(self.__init_pose['RShoulder'],
                                                                   self.__init_pose['RHip'])
        self.init_body_lengths['l_hip_shoulder'] = get_2d_distance(self.__init_pose['LShoulder'],
                                                                   self.__init_pose['LHip'])
        mid_shoulder = (self.__init_pose['LShoulder'][X] + ((self.__init_pose['RShoulder'][X] -
                                                             self.__init_pose['LShoulder'][X]) / 2),
                        (self.__init_pose['RShoulder'][Y] + self.__init_pose['LShoulder'][Y]) / 2)
        self.init_body_lengths['shoulder_nose'] = get_2d_distance(mid_shoulder, self.__init_pose['Nose'])

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
        if not self.run_dir:
            return False
        self.__get_body_lengths()
        for squat_points in self.valid_points_list:
            three_d, centered = list(), list()
            start_frame, end_frame = squat_points[0]['frame_idx'], squat_points[-1]['frame_idx']
            for points in squat_points:
                self.class_body_points = copy.deepcopy(points)
                self.center_r_heel()
                self.rotate_points_around_r_heel()
                self.create_3d_points()
                centered.append(self.get_class_points_list())
                three_d.append(self.get_class_3d_points_list())

                plot_profile_points(three_d[-1], BODY_PARTS_LIST_CLASS,
                                    'Profile body points - frame={}'.format(points['frame_idx']), self.run_dir,
                                    'profile_idx_{}'.format(points['frame_idx']), 'profile_plots')
                plot_points(centered[-1], BODY_PARTS_LIST_CLASS,
                            'Centered body points - frame={}'.format(points['frame_idx']), self.run_dir,
                            'centered_idx_{}'.format(points['frame_idx']), 'centered_plots')
            write_to_txt_points('\n'.join(str(dots) for dots in centered), self.run_dir,
                                'centered_points_id_{}-{}.txt'.format(start_frame, end_frame), 'centered_points')
            write_to_txt_points('\n'.join(str(dots) for dots in three_d), self.run_dir,
                                'three_d_points_id_{}-{}.txt'.format(start_frame, end_frame), 'three_d_points')

    def output_data_set_points(self):
        self.__get_body_lengths()
        for squat_points in self.valid_points_list:
            three_d = list()
            centered = list()
            start_frame = squat_points[-1]['frame_idx']
            for points in squat_points:
                self.class_body_points = copy.deepcopy(points)
                self.center_r_heel()
                self.rotate_points_around_r_heel()
                self.create_3d_points()
                centered.append(self.get_class_points_list())
                three_d.append(self.get_class_3d_points_list())
            cur_string_time = "".join(str(datetime.datetime.now()).split(" ")[1].split(".")[0].split(":"))
            cur_string_date = "".join(str(datetime.datetime.now()).split(" ")[0].split("-")) + cur_string_time
            write_to_txt_points('\n'.join(str(dots) for dots in centered), 'training_data_set',
                                'centered_points_id_{}_d_{}.txt'.format(start_frame, cur_string_date),
                                'centered_points/{}'.format(self.training_dir))
            write_to_txt_points('\n'.join(str(dots) for dots in three_d), 'training_data_set',
                                'three_d_points_id_{}_d_{}.txt'.format(start_frame, cur_string_date),
                                'three_d_points/{}'.format(self.training_dir))
        self.class_body_points = None
        self.class_body_points_3d = None
        self.class_body_points_centered = None
        self.valid_points_list = list()

    def center_r_heel(self):
        self.class_body_points_centered = dict()
        heel_point = self.class_body_points['RHeel']
        move_x = heel_point[X]
        move_y = heel_point[Y]
        for body_part in BODY_PARTS_LIST_CLASS:
            self.class_body_points_centered[body_part] = ((self.class_body_points[body_part][X] - move_x),
                                                          (self.class_body_points[body_part][Y] - move_y))

    def rotate_points_around_r_heel(self):
        r_ankle_coord = self.class_body_points_centered['RHeel']
        if r_ankle_coord != (0, 0):
            return
        for body_part in self.class_body_points_centered:
            if body_part == 'RHeel':
                continue
            self.class_body_points_centered[body_part] = rotate_point(self.class_body_points_centered[body_part],
                                                                      math.pi)

    def create_3d_points(self):
        self.class_body_points_3d = dict()
        cur = copy.deepcopy(self.class_body_points_centered['RHeel'])
        self.class_body_points_3d['RHeel'] = cur + tuple([0])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['RAnkle'])
        self.class_body_points_3d['RAnkle'] = cur + tuple([self.class_body_points_3d['RHeel'][Z] +
                                                           get_z_coordinate(cur, self.init_body_lengths['r_heel_ankle'],
                                                                            prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['RKnee'])
        self.class_body_points_3d['RKnee'] = cur + tuple([self.class_body_points_3d['RAnkle'][Z] +
                                                          get_z_coordinate(cur, self.init_body_lengths['r_ankle_knee'],
                                                                           prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['RHip'])
        self.class_body_points_3d['RHip'] = cur + tuple([self.class_body_points_3d['RKnee'][Z] -
                                                         get_z_coordinate(cur, self.init_body_lengths['r_knee_hip'],
                                                                          prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['RShoulder'])
        self.class_body_points_3d['RShoulder'] = cur + tuple([self.class_body_points_3d['RHip'][Z] +
                                                              get_z_coordinate(cur,
                                                                               self.init_body_lengths['r_hip_shoulder'],
                                                                               prev)])
        cur = copy.deepcopy(self.class_body_points_centered['LHeel'])
        self.class_body_points_3d['LHeel'] = cur + tuple([0])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LAnkle'])
        self.class_body_points_3d['LAnkle'] = cur + tuple([self.class_body_points_3d['LHeel'][Z] +
                                                           get_z_coordinate(cur, self.init_body_lengths['l_heel_ankle'],
                                                                            prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LKnee'])
        self.class_body_points_3d['LKnee'] = cur + tuple([self.class_body_points_3d['LAnkle'][Z] +
                                                          get_z_coordinate(cur, self.init_body_lengths['l_ankle_knee'],
                                                                           prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LHip'])
        self.class_body_points_3d['LHip'] = cur + tuple([self.class_body_points_3d['LKnee'][Z] -
                                                         get_z_coordinate(cur, self.init_body_lengths['l_knee_hip'],
                                                                          prev)])
        prev = cur
        cur = copy.deepcopy(self.class_body_points_centered['LShoulder'])
        self.class_body_points_3d['LShoulder'] = cur + tuple([self.class_body_points_3d['LHip'][Z] +
                                                              get_z_coordinate(cur,
                                                                               self.init_body_lengths['l_hip_shoulder'],
                                                                               prev)])
        mid_shoulder = (self.class_body_points_3d['RShoulder'][X] + ((self.class_body_points_3d['LShoulder'][X] -
                                                                      self.class_body_points_3d['RShoulder'][X]) / 2),
                        (self.class_body_points_3d['RShoulder'][Y] + self.class_body_points_3d['LShoulder'][Y]) / 2,
                        (self.class_body_points_3d['RShoulder'][Z] + self.class_body_points_3d['LShoulder'][Z] / 2))
        prev = mid_shoulder
        cur = copy.deepcopy(self.class_body_points_centered['Nose'])
        self.class_body_points_3d['Nose'] = cur + tuple([mid_shoulder[Z] +
                                                         get_z_coordinate(cur, self.init_body_lengths['shoulder_nose'],
                                                                          prev)])
        self.class_body_points_3d['REye_c'] = self.class_body_points_centered['REye_c'] + \
                                              tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['LEye_c'] = self.class_body_points_centered['LEye_c'] + \
                                              tuple([self.class_body_points_3d['Nose'][Z]])
        self.class_body_points_3d['RTows'] = (self.class_body_points_centered['RTows'][X], 0) + \
                                             tuple([get_z_coordinate((self.class_body_points_centered['RTows'][X], 0),
                                                                     self.__user_height_in_pixel * 0.12,
                                                                     self.class_body_points_centered['RHeel'])])
        self.class_body_points_3d['LTows'] = (self.class_body_points_centered['LTows'][X], 0) + \
                                             tuple([get_z_coordinate((self.class_body_points_centered['LTows'][X], 0),
                                                                     self.__user_height_in_pixel * 0.12,
                                                                     self.class_body_points_centered['LHeel'])])
